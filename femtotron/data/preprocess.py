# preprocess.py
"""
Unified preprocess for pretrain / SFT / future tasks.
离线把原始数据 tokenize + pack 成定长序列，存到磁盘。

设计原则:
1. tokenization × labeling × packing 三轴正交,任意 sane 组合都合法
2. 输出格式自证类型(meta 字段),training 加载时校验
3. Trainer 完全不知道 task 类型,collator 是 stack
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal
import itertools
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

Task    = Literal["pretrain", "sft"]
Packing = Literal["concat", "ffd", "pad"]


# ─── Stage 1: tokenization ─────────────────────────────────────────────────
# 每个 tokenize_* 函数都产出标准化的 {"input_ids": [...], "labels": [...]}
# labels 已经按该 task 的策略 mask 好

def _tokenize_pretrain(
    raw, tokenizer,
    text_field: str = "text",
    add_eos: bool = True,
    num_proc: int = 16,
):
    """
    预训练 tokenization:每条文本 tokenize 后 labels = input_ids。
    可选在末尾加 EOS(标记文档边界)。
    """
    eos_id = tokenizer.eos_token_id
    
    def fn(examples):
        outputs = tokenizer(examples[text_field], add_special_tokens=False)
        input_ids = outputs["input_ids"]
        if add_eos:
            input_ids = [ids + [eos_id] for ids in input_ids]
        return {"input_ids": input_ids, "labels": input_ids}
    
    return raw.map(
        fn, batched=True, num_proc=num_proc,
        remove_columns=raw.column_names,
        desc="Tokenizing (pretrain)",
    )


def _tokenize_sft(
    raw, tokenizer,
    messages_field: str = "messages",
    ignore_index: int = -100,
    num_proc: int = 16,
):
    """
    SFT tokenization:增量渲染对话,只让 assistant 段的 token 进入 loss。
    
    每个 message 依次渲染,diff 出新增的 tokens:
      - role == "assistant":new_ids 进入 labels
      - role != "assistant":new_ids 在 labels 里填 ignore_index
    
    天然支持 multi-turn(多个 assistant 段都计算 loss),
    天然支持 system prompt(归在非 assistant 段,被 mask)。
    
    时间复杂度 O(N_messages²) on chat_template render,但 N 通常 < 20。
    """
    def fn(example):
        messages = example[messages_field]
        input_ids: list[int] = []
        loss_mask:    list[int] = []
        cum_ids:   list[int] = []
        
        for i, msg in enumerate(messages):
            text = tokenizer.apply_chat_template(
                messages[:i+1],
                tokenize=False,
                add_generation_prompt=False,
            )
            new_cum_ids = tokenizer(text, add_special_tokens=False).input_ids
            new_ids = new_cum_ids[len(cum_ids):]
            
            input_ids.extend(new_ids)
            # loss_mask:assistant 段 True,其他 False
            is_assistant = msg["role"] == "assistant"
            loss_mask.extend([is_assistant] * len(new_ids))
            
            cum_ids = new_cum_ids
        
        return {"input_ids": input_ids, "loss_mask": loss_mask}
    
    return raw.map(
        fn, batched=False, num_proc=num_proc,
        remove_columns=raw.column_names,
        desc="Tokenizing (sft)",
    )


# ─── Stage 2: packing ──────────────────────────────────────────────────────
# 每个 _pack_* 函数都接受 tokenized dataset(有 input_ids, labels)
# 产出统一的 dict[str, Tensor]

def _pack_concat(tokenized, seq_len: int) -> dict[str, torch.Tensor]:
    """
    自由拼接(预训练):
    所有样本首尾拼成一个长流,切成 seq_len 块,丢弃末尾不足部分。
    """
    all_ids = list(itertools.chain.from_iterable(tokenized["input_ids"]))
    all_lbl = list(itertools.chain.from_iterable(tokenized["labels"]))
    assert len(all_ids) == len(all_lbl)
    
    n_full = len(all_ids) // seq_len
    truncated = n_full * seq_len
    
    return {
        "input_ids": torch.tensor(all_ids[:truncated], dtype=torch.long).view(n_full, seq_len),
        "labels":    torch.tensor(all_lbl[:truncated], dtype=torch.long).view(n_full, seq_len),
    }


def _pack_ffd(
    tokenized, seq_len: int, pad_token_id: int,
    max_subseqs: int = 32,
    drop_oversize: bool = True,
    shuffle_seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    First-Fit-Decreasing packing(SFT):
    样本按长度降序,贪心装进 bin。每个子样本 position_ids 从 0 重置。
    """
    # 收集 + 过滤
    samples = []
    for ids, lbl in zip(tokenized["input_ids"], tokenized["labels"]):
        n = len(ids)
        if n > seq_len:
            if drop_oversize:
                continue
            raise ValueError(f"sample length {n} > seq_len {seq_len}")
        samples.append((ids, lbl, n))
    
    # FFD
    samples.sort(key=lambda x: -x[2])
    bins: list[dict] = []
    for ids, lbl, n in samples:
        placed = False
        for b in bins:
            if b["used"] + n <= seq_len and len(b["samples"]) < max_subseqs:
                b["samples"].append((ids, lbl, n))
                b["used"] += n
                placed = True
                break
        if not placed:
            bins.append({"samples": [(ids, lbl, n)], "used": n})
    
    random.Random(shuffle_seed).shuffle(bins)
    
    # 构造 tensors
    N = len(bins)
    input_ids    = torch.full((N, seq_len), pad_token_id, dtype=torch.long)
    loss_mask = torch.zeros((N, seq_len), dtype=torch.bool)   # 默认全 False = 不算 loss
    position_ids = torch.zeros((N, seq_len), dtype=torch.int32)
    seqlens      = torch.zeros((N, max_subseqs), dtype=torch.int32)
    
    for i, b in enumerate(bins):
        offset = 0
        for j, (ids, mask, n) in enumerate(b["samples"]):
            input_ids[i, offset:offset+n]    = torch.tensor(ids, dtype=torch.long)
            loss_mask[i, :n] = torch.tensor(mask, dtype=torch.bool)
            position_ids[i, offset:offset+n] = torch.arange(n)
            seqlens[i, j] = n
            offset += n
        # Pad section 作为最后一个 "doc"
        pad_n = seq_len - offset
        if pad_n > 0:
            position_ids[i, offset:] = torch.arange(pad_n)   # 任意,反正被 attention mask 掉
            seqlens[i, len(b["samples"])] = pad_n
    
    # 计算 packing 效率(写进 meta 方便诊断)
    total_real = sum(b["used"] for b in bins)
    efficiency = total_real / (N * seq_len)
    print(f"  Packing efficiency: {efficiency:.1%} ({N} bins, {total_real:,}/{N * seq_len:,} tokens)")
    
    return {
        "input_ids":    input_ids,
        "loss_mask":    loss_mask,
        "position_ids": position_ids,
        "seqlens":      seqlens,
    }


def _pack_pad(
    tokenized, seq_len: int, pad_token_id: int,
    drop_oversize: bool = True,
) -> dict[str, torch.Tensor]:
    """
    无 packing,每条 pad 到 seq_len(SFT 简单模式)。
    """
    samples = []
    for ids, lbl in zip(tokenized["input_ids"], tokenized["labels"]):
        n = len(ids)
        if n > seq_len:
            if drop_oversize:
                continue
            raise ValueError(f"sample length {n} > seq_len {seq_len}")
        samples.append((ids, lbl, n))
    
    N = len(samples)
    input_ids = torch.full((N, seq_len), pad_token_id, dtype=torch.long)
    loss_mask = torch.zeros((N, seq_len), dtype=torch.bool)   # 默认全 False = 不算 loss
    for i, (ids, mask, n) in enumerate(samples):
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        loss_mask[i, :n] = torch.tensor(mask, dtype=torch.bool)
    
    return {"input_ids": input_ids, "loss_mask": loss_mask}


# ─── Stage 3: orchestrator ─────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    # I/O
    dataset_name: str
    tokenizer_name: str
    output_path: str
    
    # Common
    seq_len: int = 4096
    num_proc: int = 16
    num_samples: int | None = None
    ignore_index: int = -100
    
    # Task selection
    task: Task = "pretrain"
    
    # Pretrain-specific
    text_field: str = "text"
    add_eos: bool = True
    
    # SFT-specific
    messages_field: str = "messages"
    
    # Packing
    packing: Packing = "concat"   # 默认与 task 配套
    max_subseqs: int = 32
    drop_oversize: bool = True
    shuffle_seed: int = 42
    
    def __post_init__(self):
        # 校验 task / packing 组合
        valid = {
            ("pretrain", "concat"),
            ("pretrain", "pad"),     # 罕见但合法
            ("sft", "ffd"),
            ("sft", "pad"),
        }
        if (self.task, self.packing) not in valid:
            raise ValueError(
                f"Invalid (task={self.task}, packing={self.packing}). "
                f"Valid combos: {valid}"
            )


def preprocess(cfg: PreprocessConfig) -> dict:
    """
    统一预处理入口。
    """
    # 1. Load
    split = "train" if cfg.num_samples is None else f"train[:{cfg.num_samples}]"
    raw = load_dataset(cfg.dataset_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        print(f"[warn] tokenizer has no pad_token_id, falling back to eos_id={pad_id}")
    
    # 2. Tokenize(按 task 分派)
    tokenize_fn = {
        "pretrain": lambda: _tokenize_pretrain(
            raw, tokenizer,
            text_field=cfg.text_field,
            add_eos=cfg.add_eos,
            num_proc=cfg.num_proc,
        ),
        "sft": lambda: _tokenize_sft(
            raw, tokenizer,
            messages_field=cfg.messages_field,
            ignore_index=cfg.ignore_index,
            num_proc=cfg.num_proc,
        ),
    }[cfg.task]
    tokenized = tokenize_fn()
    
    # 3. Pack(按 packing 分派)
    pack_fn = {
        "concat": lambda: _pack_concat(tokenized, cfg.seq_len),
        "ffd": lambda: _pack_ffd(
            tokenized, cfg.seq_len, pad_id,
            max_subseqs=cfg.max_subseqs,
            drop_oversize=cfg.drop_oversize,
            shuffle_seed=cfg.shuffle_seed,
        ),
        "pad": lambda: _pack_pad(
            tokenized, cfg.seq_len, pad_id,
            drop_oversize=cfg.drop_oversize,
        ),
    }[cfg.packing]
    packed = pack_fn()
    
    # 4. Save with meta
    output = {
        **packed,
        "meta": {
            **{k: v for k, v in asdict(cfg).items() if k not in ("output_path", "num_samples")},
            "requested_num_samples": cfg.num_samples,
            "num_samples":  int(packed["input_ids"].shape[0]),
            "total_tokens": int(packed["input_ids"].numel()),
            "pad_token_id": pad_id,
        },
    }
    torch.save(output, cfg.output_path)
    print(f"Saved {output['meta']['num_samples']} samples "
          f"({output['meta']['total_tokens']:,} tokens) to {cfg.output_path}")
    return output

if __name__ == "__main__":
    preprocess_config = PreprocessConfig(
        dataset_name="HuggingFaceFW/fineweb-edu",
        tokenizer_name="meta-llama/Meta-Llama-3-8B",
        output_path="packed_4k.pt",
        seq_len=4096,
        num_proc=16,
    )
    preprocess(preprocess_config)

    # # SFT(packed,生产用)
    # cfg = PreprocessConfig(
    #     dataset_name="HuggingFaceH4/ultrachat_200k",
    #     tokenizer_name="meta-llama/Llama-3-8B-Instruct",
    #     output_path="data/sft_packed.pt",
    #     seq_len=4096,
    #     task="sft",
    #     packing="ffd",
    #     max_subseqs=32,
    # )
    # preprocess(cfg)


    # # SFT(padded,教学/调试用)
    # cfg = PreprocessConfig(
    #     dataset_name="...",
    #     tokenizer_name="...",
    #     output_path="data/sft_pad.pt",
    #     seq_len=2048,
    #     task="sft",
    #     packing="pad",
    # )
    # preprocess(cfg)