"""
Femtotron 训练入口脚本(支持 3D parallelism: PP + DP + TP + ZeRO + AC)
============================================================

用法:
    # 8卡 DP+TP(老用法,无 PP)
    torchrun --nproc_per_node=8 scripts/train.py --config configs/tiny_llama_debug.yaml --dp 4 --tp 2

    # 8卡 3D parallel(PP=2, DP=2, TP=2)
    torchrun --nproc_per_node=8 scripts/train.py --config configs/tiny_llama_debug.yaml \\
        --pp 2 --dp 2 --tp 2 --num_microbatches 4 --pp_schedule 1f1b

    # 4卡纯 PP(开发常用,验证 PP 路径)
    torchrun --nproc_per_node=4 scripts/train.py --config configs/tiny_llama_debug.yaml \\
        --pp 4 --dp 1 --tp 1 --num_microbatches 4
"""

import os
import sys
import argparse
import yaml
import time
import torch
import torch.distributed as dist
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path

from femtotron.data.data_source import PreprocessedDataset
from femtotron.sharding.factory import create_sharding_strategy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from femtotron.parallel_context import ParallelContext
from femtotron.model.parallel_plan import get_llama_parallel_plan
from femtotron.model.model_loader import ModelLoader
from femtotron.model.llama import build_llama_model
from femtotron.training.mixed_precision_manager import MixedPrecisionManager
from femtotron.training.optimizer import get_param_groups
from femtotron.training.lr_schedule import create_lr_schedule
from femtotron.training.trainer import Trainer
from femtotron.training.train_config import TrainConfig, PipelineConfig
from femtotron.data.data_loader import DistributedDataLoader
from femtotron.data.preprocess import preprocess, PreprocessConfig
from femtotron.data.collator import Collator, simple_pretrain_collator, PadSftCollator, stack_collator

# ─── DP 相关 import ───
from femtotron.parallel.data_parallel.ddp import DataParallelGradSync
from femtotron.parallel.data_parallel.gradient_synchronizer import GradientSynchronizer
# ─── PP 相关 import ───
from femtotron.parallel.pipeline_parallel.partition import partition_layers
from femtotron.parallel.pipeline_parallel.stage import PipelineStage
from femtotron.parallel.pipeline_parallel.comm_ops import PipelineComm
from femtotron.parallel.pipeline_parallel.runner import PipelineRunner


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=int(os.environ.get("WORLD_SIZE", 1)))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(msg):
    if dist.get_rank() == 0:
        print(msg)


def _reset_rotary_inv_freq(model: torch.nn.Module, model_config, device: torch.device):
    """
    `to_empty()` 不初始化 persistent=False 的 buffer。
    LlamaRotaryEmbedding 的 `inv_freq` 就是这种 buffer,需要显式重算。
    
    随机初始化路径必须调用;ModelLoader 路径下生产实现应当自己处理。
    """
    rotary_emb = getattr(model.model, "rotary_emb", None)
    if rotary_emb is None:
        return  # 极少数变体可能没有,容错
    
    base = getattr(model_config, "rope_theta", 10000.0)
    dim = (
        getattr(model_config, "head_dim", None)
        or (model_config.hidden_size // model_config.num_attention_heads)
    )
    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, dim, 2, dtype=torch.int64)
            .to(device=device, dtype=torch.float)
            / dim
        )
    )
    rotary_emb.inv_freq.copy_(inv_freq.to(rotary_emb.inv_freq.dtype))
    if hasattr(rotary_emb, "original_inv_freq"):
        rotary_emb.original_inv_freq.copy_(
            inv_freq.to(rotary_emb.original_inv_freq.dtype)
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Femtotron Training")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径")

    # 并行配置
    parser.add_argument("--dp", type=int, default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--pp", type=int, default=None)

    # ─── PP 专属配置 ───
    parser.add_argument("--num_microbatches", type=int, default=None,
                        help="PP 一个 step 切多少个 microbatch(pp_size>1 时生效)")
    parser.add_argument("--pp_schedule", type=str, default=None,
                        choices=["gpipe", "1f1b"],
                        help="PP schedule: gpipe (memory-heavy, simple) or 1f1b (production)")
    parser.add_argument("--pp_partition", type=str, default=None,
                        choices=["uniform"],
                        help="PP 层分配策略(目前仅 uniform)")

    # ZeRO 配置
    parser.add_argument("--zero_stage", type=int, default=None,
                        help="ZeRO stage: 0=baseline / 1 / 2 / 3")
    parser.add_argument("--zero_wrap_policy", type=str, default=None)

    # AC 配置
    parser.add_argument("--ac_enabled", type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=None, help="启用 activation checkpointing")
    parser.add_argument("--ac_policy", type=str, default=None)

    # 模型配置
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)

    # 训练超参
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None,
                        help="DataLoader 每次返回的 batch 大小(PP 下会被 num_microbatches 切分)")
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)

    # 数据
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)

    # Logging & Checkpoint
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def load_config(args) -> dict:
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    cli_overrides = {k: v for k, v in vars(args).items()
                     if v is not None and k != "config"}
    for key, value in cli_overrides.items():
        config[key] = value
    return config


def _build_model(config, model_config, parallel_ctx, device, plan,
                  *, use_pp_aware: bool, layer_range, model_name):
    """构造模型 — PP-aware 与否的两条路径在这里收口。"""
    with torch.device("meta"):
        if use_pp_aware:
            model = build_llama_model(
                model_config, parallel_ctx,
                use_pp_aware=True, layer_range=layer_range,
            )
        else:
            model = build_llama_model(model_config, parallel_ctx)
    
    if model_name:
        # 真实预训练权重路径
        loader = ModelLoader(parallel_ctx)
        # 假设 ModelLoader 已经支持 PP partial(传 layer_range 让它只 load 对应层)
        loader.load_and_distribute(
            model, model_name, parallel_plan=plan, device=device,
        )
    else:
        # 随机初始化
        model = model.to_empty(device=device)
        # ── 关键:to_empty 后必须重算 rotary inv_freq ──
        _reset_rotary_inv_freq(model, model_config, device)
        for p in model.parameters():
            if p.requires_grad:
                nn_init_scale = config.get("init_std", 0.02)
                torch.nn.init.normal_(p, mean=0.0, std=nn_init_scale)
    
    return model.bfloat16()


def build_all(config: dict):
    """根据配置构建所有组件。"""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # ─── 1. 并行拓扑 ───
    tp_size = config.get("tp", 1)
    pp_size = config.get("pp", 1)
    dp_size = config.get("dp", world_size // (tp_size * pp_size))

    assert dp_size * tp_size * pp_size == world_size, (
        f"dp({dp_size}) × tp({tp_size}) × pp({pp_size}) "
        f"= {dp_size*tp_size*pp_size} != world_size({world_size})"
    )

    parallel_ctx = ParallelContext(OrderedDict([
        ("pp", pp_size),
        ("dp", dp_size),
        ("tp", tp_size),
    ]))

    log(f"并行拓扑: PP={pp_size}, DP={dp_size}, TP={tp_size}")
    log(f"  本 rank: pp={parallel_ctx.pp_rank}, "
        f"dp={parallel_ctx.dp_rank}, tp={parallel_ctx.tp_rank}")

    # ─── 2. PP 配置 + 校验 ───
    num_microbatches = config.get("num_microbatches", 1)
    pp_schedule = config.get("pp_schedule", "1f1b")
    pp_partition_strategy = config.get("pp_partition", "uniform")
    
    if pp_size > 1:
        assert num_microbatches >= pp_size, (
            f"num_microbatches({num_microbatches}) 应当 ≥ pp_size({pp_size}),"
            f"否则流水线会有空 stage(bubble 占满)"
        )
        log(f"PP 配置: schedule={pp_schedule}, num_microbatches={num_microbatches}, "
            f"partition={pp_partition_strategy}")
    elif num_microbatches > 1:
        log(f"  注意: pp_size=1 但 num_microbatches={num_microbatches},将被忽略")
        num_microbatches = 1

    # ─── 3. 模型配置 ───
    model_name = config.get("model_name", None)
    plan = get_llama_parallel_plan()

    if model_name:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model_name)
        if config.get("num_hidden_layers"):
            model_config.num_hidden_layers = config["num_hidden_layers"]
        tokenizer_name = model_name
        log(f"模型: {model_name}")
    else:
        from transformers import AutoConfig, AutoTokenizer
        tokenizer_name = config.get("tokenizer", "meta-llama/Llama-2-7b-hf")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        log(f"Tokenizer: {tokenizer_name} (vocab_size={tokenizer.vocab_size})")
        model_config = AutoConfig.for_model(
            "llama",
            hidden_size=config.get("hidden_size", 256),
            intermediate_size=config.get("intermediate_size", 512),
            num_attention_heads=config.get("num_attention_heads", 8),
            num_key_value_heads=config.get("num_key_value_heads", 4),
            num_hidden_layers=config.get("num_hidden_layers", 4),
            max_position_embeddings=config.get("max_position_embeddings", 512),
            vocab_size=tokenizer.vocab_size,
            rms_norm_eps=1e-5,
            hidden_act="silu",
            tie_word_embeddings=config.get("tie_word_embeddings", False),
        )
        log(f"模型: Tiny LLaMA (随机初始化)")
    
    log(f"  层数: {model_config.num_hidden_layers}, "
        f"Hidden: {model_config.hidden_size}, "
        f"Vocab: {model_config.vocab_size}")
    
    # 3D 拓扑下的 model config 兼容性检查
    if tp_size > 1:
        assert model_config.num_attention_heads % tp_size == 0, (
            f"num_attention_heads({model_config.num_attention_heads}) "
            f"必须能被 tp_size({tp_size}) 整除"
        )
        assert model_config.num_key_value_heads % tp_size == 0, (
            f"num_key_value_heads({model_config.num_key_value_heads}) "
            f"必须能被 tp_size({tp_size}) 整除"
        )
        assert model_config.vocab_size % tp_size == 0, (
            f"vocab_size({model_config.vocab_size}) 必须能被 tp_size({tp_size}) 整除"
        )
    
    if pp_size > 1:
        assert model_config.num_hidden_layers >= pp_size, (
            f"num_hidden_layers({model_config.num_hidden_layers}) "
            f"必须 ≥ pp_size({pp_size}),否则某 stage 没有层"
        )

    # ─── 4. PP layer partition ───
    if pp_size > 1:
        partitions = partition_layers(
            model_config.num_hidden_layers, pp_size,
            strategy=pp_partition_strategy,
        )
        my_layers = partitions[parallel_ctx.pp_rank]
        layer_range = range(my_layers[0], my_layers[-1] + 1)
        log(f"  本 rank 持有层: {list(layer_range)}")
        use_pp_aware = True
    else:
        layer_range = None
        use_pp_aware = False

    # ─── 5. 构造模型 ───
    model = _build_model(
        config, model_config, parallel_ctx, device, plan,
        use_pp_aware=use_pp_aware, layer_range=layer_range,
        model_name=model_name,
    )
    num_params = sum(p.numel() for p in model.parameters())
    log(f"  参数量(本 rank): {num_params:,}")

    # ─── 6. 训练配置 ───
    train_config = TrainConfig(
        master_dtype=torch.float32,
        grad_clip=config.get("grad_clip", 1.0),
        grad_accum_steps=config.get("grad_accum_steps", 1),
        train_steps=config.get("train_steps", 500),
        log_interval=config.get("log_interval", 10),
        checkpoint_interval=config.get("checkpoint_interval", 100),
        checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
        warmup_steps=config.get("warmup_steps", 50),
        min_lr_ratio=config.get("min_lr_ratio", 0.1),
        pipeline_config=PipelineConfig(
            num_microbatches=num_microbatches,
            schedule=pp_schedule,
        ),
    )

    # ─── 7. ZeRO strategy ───
    from femtotron.sharding.factory import create_sharding_strategy
    from femtotron.parallel.data_parallel.gradient_synchronizer import create_grad_synchronizer
    from femtotron.sharding.zero_config import ZeROConfig
    from femtotron.scripts.presets import get_wrap_policy, get_ac_policy

    zero_stage = config.get("zero_stage", 0)
    if zero_stage == 3:
        wrap_policy_name = config.get("zero_wrap_policy")
        if not wrap_policy_name:
            raise ValueError("zero_stage=3 但没指定 zero_wrap_policy")
        wrap_policy = get_wrap_policy(wrap_policy_name)
    else:
        wrap_policy = None

    zero_config = ZeROConfig(stage=zero_stage, wrap_policy=wrap_policy)
    strategy = create_sharding_strategy(parallel_ctx, zero_config)
    log(f"ZeRO: stage={zero_stage}, strategy={type(strategy).__name__}")

    # ─── 8. MixedPrecisionManager + Optimizer ───
    lr = config.get("lr", 3e-4)
    weight_decay = config.get("weight_decay", 0.01)
    compute_param_groups = get_param_groups(model, weight_decay=weight_decay)

    mp_manager = MixedPrecisionManager(
        model=model,
        sharding_strategy=strategy,
        parallel_ctx=parallel_ctx,
        parallel_plan=plan,
        config=train_config,
        inner_optimizer_cls=torch.optim.AdamW,
        inner_optimizer_kwargs={
            "lr": lr,
            "betas": (config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.95)),
            "eps": config.get("adam_eps", 1e-8),
        },
        compute_param_groups=compute_param_groups,
    )
    
    if hasattr(strategy, "prepare_for_backward"):
        strategy.prepare_for_backward(mp_manager.groups)

    # ─── 9. AC(在 mp_manager 之后) ───
    if config.get("ac_enabled", False):
        from femtotron.training.activation_ckpt import apply_activation_checkpointing
        ac_policy_name = config.get("ac_policy")
        if not ac_policy_name:
            raise ValueError("ac_enabled=True 但没指定 ac_policy")
        ac_policy = get_ac_policy(ac_policy_name)
        n_wrapped = apply_activation_checkpointing(
            model, ac_policy,
            use_reentrant=config.get("ac_use_reentrant", False),
            preserve_rng_state=config.get("ac_preserve_rng_state", False),
        )
        log(f"  AC: wrapped {n_wrapped} modules")
    else:
        log(f"  AC: disabled")

    log(f"训练配置:")
    log(f"  LR={lr}, WD={weight_decay}, Warmup={train_config.warmup_steps}, "
        f"Clip={train_config.grad_clip}")
    log(f"  Grad accum steps: {train_config.grad_accum_steps}")
    log(f"  精度: BF16 compute + FP32 master")

    # ─── 10. LR Schedule ───
    scheduler = create_lr_schedule(
        mp_manager.inner,
        warmup_steps=train_config.warmup_steps,
        total_steps=train_config.train_steps,
        min_lr_ratio=train_config.min_lr_ratio,
    )

    # ─── 11. 数据 ───
    seq_len = config.get("seq_len", 128)
    micro_batch_size = config.get("micro_batch_size", 1)       # PP 最小单位
    num_microbatches = config.get("num_microbatches", 1)
    
    # 用 dataclass 的 step batch(可选,但用着方便)
    step_batch_size = micro_batch_size * num_microbatches
    
    log(f"  micro_batch_size:     {micro_batch_size}")
    log(f"  num_microbatches:     {num_microbatches}")
    log(f"  step_batch_size/DP:   {step_batch_size}")
    
    # ─── 数据缓存路径(包含 task/packing 自证)───
    task    = config.get("task", "pretrain")
    packing = config.get("packing", "concat" if task == "pretrain" else "ffd")
    
    dataset_name = config.get("dataset", "roneneldan/TinyStories")
    data_dir = config.get("data_dir", "./data")
    safe_name = dataset_name.replace("/", "_")
    safe_tok = tokenizer_name.replace("/", "_")
    cache_path = os.path.join(
        data_dir,
        f"{safe_name}_{safe_tok}_seqlen{seq_len}_{task}_{packing}.pt"
    )
    
    # ─── Rank 0 预处理,其他 rank 等 ───
    if dist.get_rank() == 0:
        if os.path.exists(cache_path):
            log(f"数据: 使用缓存 {cache_path}")
        else:
            log(f"数据: 缓存不存在,开始预处理 (task={task}, packing={packing})...")
            os.makedirs(data_dir, exist_ok=True)
            preprocess_config = PreprocessConfig(
                dataset_name=dataset_name,
                tokenizer_name=tokenizer_name,
                output_path=cache_path,
                seq_len=seq_len,
                task=task,
                packing=packing,
                num_proc=config.get("preprocess_num_proc", 16),
            )
            preprocess(preprocess_config)
            log(f"数据: 预处理完成")
    dist.barrier()
    
    # ─── 所有 rank 加载(mmap)───
    train_dataset = PreprocessedDataset(
        cache_path,
        mmap=True,
        expect_task=task,   # ← meta 自证
    )
    log(f"数据: {len(train_dataset)} 条样本, "
        f"seq_len={train_dataset.input_ids.shape[1]}, "
        f"task={train_dataset.meta['task']}, "
        f"packing={train_dataset.meta['packing']}")
    
    dataloader = DistributedDataLoader(
        dataset=train_dataset,
        parallel_ctx=parallel_ctx,
        micro_batch_size=micro_batch_size,
        collator=stack_collator,
        sampler=None,
        num_workers=config.get("num_workers", 2),
    )
    
    tokens_per_step = (
        micro_batch_size
        * num_microbatches
        * seq_len
        * dp_size
        * train_config.grad_accum_steps
    )
    log(f"  Tokens/optimizer step (全局): {tokens_per_step:,}")

    # ─── 12. Grad sync ───
    grad_sync = create_grad_synchronizer(mp_manager.groups, parallel_ctx, strategy)
    log(f"  Grad sync: {type(grad_sync).__name__}")

    # ─── 13. PipelineRunner(仅 pp_size > 1) ───
    pp_runner = None
    if pp_size > 1:
        per_mb_size = micro_batch_size // num_microbatches
        # loss_scale = 1 / (grad_accum × num_microbatches),让 sum(per-mb-grad) = batch_mean
        loss_scale = 1.0 / (train_config.grad_accum_steps * num_microbatches)
        
        stage = PipelineStage(model, parallel_ctx, loss_scale=loss_scale)
        comm = PipelineComm(parallel_ctx, seq_len, model_config.hidden_size, dtype=torch.bfloat16)
        recv_shape = (per_mb_size, seq_len, model_config.hidden_size)
        pp_runner = PipelineRunner(
            stage, comm,
            schedule_name=pp_schedule,
            num_microbatches=num_microbatches,
            recv_shape=recv_shape,
            recv_dtype=torch.bfloat16,
        )
        log(f"  PipelineRunner: schedule={pp_schedule}, M={num_microbatches}, "
            f"recv_shape={recv_shape}, loss_scale={loss_scale:.6f}")

    # ─── 14. Trainer ───
    trainer = Trainer(
        model=model,
        mp_manager=mp_manager,
        scheduler=scheduler,
        dataloader=dataloader,
        grad_sync=grad_sync,
        parallel_ctx=parallel_ctx,
        train_config=train_config,
        pp_runner=pp_runner,  # ← PP 路径开关:None=standard, 否则走 PP path
    )

    return trainer


def main():
    args = parse_args()
    config = load_config(args)

    local_rank = init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("=" * 60)
        print("  Femtotron Training")
        print(f"  World size: {world_size}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 60)
    dist.barrier()

    trainer = build_all(config)

    if args.resume:
        log(f"\n从 checkpoint 恢复: {args.resume}")
        trainer._load_checkpoint(args.resume)

    log(f"\n{'=' * 60}")
    log(f"  开始训练")
    log(f"{'=' * 60}\n")

    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  训练完成")
        print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"{'=' * 60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()