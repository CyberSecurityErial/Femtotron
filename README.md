# Femtotron

**从零实现的分布式 LLM 训练框架**

> *Femto (10⁻¹⁵)* — 比 [Picotron](https://github.com/huggingface/picotron) 更小一点。

Femtotron 是一个教育导向的分布式训练框架项目。从零开始实现了 3D 并行（TP + DP + PP）、ZeRO Stage 1/2/3、混合精度训练、Activation Checkpointing 和 SFT 微调的完整训练栈。目标是以清晰的代码和完备的测试，帮助开发者深入理解大模型训练的每一个底层细节。

配套项目：[Pico-vLLM](https://github.com/Koas-W/Pico-vLLM)（从零实现的推理引擎）

<!-- TODO: 如果有博客链接，加在这里 -->
<!-- 📖 [开发博客](https://koas-w.github.io/) -->

---

## ✨ Feature 一览

### 已实现

| 类别 | Feature | 说明 |
|------|---------|------|
| **Tensor Parallel** | ColumnParallelLinear / RowParallelLinear | 带自定义 autograd 通信算子（CopyToTP / GatherFromTP / ReduceFromTP / ScatterToTP） |
| | VocabParallelEmbedding | vocab 维度切分，all-reduce 聚合 |
| **Data Parallel** | DDP 梯度同步 | all-reduce + 梯度累积 |
| **Pipeline Parallel** | 1F1B 调度 | 模型按层切分、跨 stage P2P 通信、warmup/steady/cooldown 三阶段调度 |
| **ZeRO** | Stage 1 | Optimizer states 分片 |
| | Stage 2 | + 梯度 reduce-scatter |
| | Stage 3 | + 参数分片 + forward/backward hook 以层为粒度按需 all-gather |
| **混合精度** | BF16 compute + FP32 master weights | 精度管理、梯度搬运、权重同步一体化 |
| **Activation Checkpointing** | 可配置的 Selective 模式 | 工厂类设计，可扩展 |
| **模型支持** | LLaMA 架构 | HuggingFace 兼容，ParallelPlan 驱动的自动并行化 |
| **权重加载** | 分布式加载模式 | Meta device 创建，支持 safetensors |
| **训练** | 完整训练循环 | LR warmup + cosine decay、gradient clipping、checkpoint save/load |
| **SFT** | 监督微调 | Chat template 数据处理、loss masking |

### 与其他教育框架对比

| Feature | Femtotron | [Picotron](https://github.com/huggingface/picotron) | [Nanotron](https://github.com/huggingface/nanotron) |
|---------|:---------:|:-------:|:--------:|
| Tensor Parallel | ✅ | ✅ | ✅ |
| Data Parallel | ✅ | ✅ | ✅ |
| Pipeline Parallel | ✅ (1F1B, Interleaved 1F1B) | ✅ (1F1B) | ✅ (1F1B) |
| ZeRO Stage 1 | ✅ | ❌ | ✅ |
| ZeRO Stage 2 | ✅ | ❌ | ❌ |
| **ZeRO Stage 3** | **✅** | **❌** | **❌** |
| Activation Checkpointing | ✅ | ✅ | ✅ |
| SFT | ✅ | ❌ | ✅ |
| Context Parallel | 🔜 | ✅ | ❌ |
| Interleaved 1F1B | ✅ | ❌ | ❌ |

<!-- TODO: 如有需要可以更新对比表 -->

### 开发中 / 计划中

| Feature | 状态 |
|---------|------|
| Zero Bubble 调度 | 🔜 计划中 |
| Sequence Parallel | 🔜 计划中 |
| 通信-计算 Overlap | 🔜 计划中 |
| LoRA 微调 | 🔜 计划中 |
| FP8 训练 | 🔜 计划中 |

---

## 🏗️ 项目结构

```
femtotron/
├── __init__.py
├── parallel_context.py                      # N 维并行网格，管理 DP/TP/PP/CP/EP 的 ProcessGroup
│
├── data/
│   ├── __init__.py
│   ├── collator.py                          # DataCollator
│   ├── data_loader.py                       # 分布式 DataLoader，按 DP rank 分片
│   ├── data_source.py                       # 数据源抽象（PackedDataset 等）
│   ├── distributed_sampler.py               # DP 感知的 DistributedSampler
│   └── preprocess.py                        # 离线预处理：tokenize + Pretrain / SFT + Pad / Pack + loss mask
│
├── model/
│   ├── __init__.py
│   ├── base.py                              # 模型基类 / 通用接口定义
│   ├── layer_factory.py                     # 根据 ParallelRule 创建对应的并行层实例
│   ├── llama.py                             # LlamaForTraining（完整模型，无 PP 时使用，不再主力使用）
│   ├── llama_causal.py                      # PP 感知的完整模型
│   ├── llama_partial_model.py               # PP 感知的部分模型（只持有本 stage 的层）
│   ├── model_loader.py                      # 分布式的参数加载脚本
│   ├── parallel_module_builder.py           # 模型构建入口，串联 parallelize + load
│   ├── parallel_plan.py                     # ParallelRule + ParallelPlan + get_llama_parallel_plan()
│   ├── parallelize_model.py                 # 遍历模型，按 plan 替换 nn.Linear → TP 版本
│   └── shard_loader.py                      # PP 分片加载（只加载本 stage 需要的层权重）
│
├── parallel/
│   ├── tensor_parallel/
│   │   ├── __init__.py
│   │   ├── comm_ops.py                      # CopyToTP / GatherFromTP / ReduceFromTP / ScatterToTP
│   │   ├── embedding.py                     # VocabParallelEmbedding
│   │   └── linear.py                        # ColumnParallelLinear / RowParallelLinear
│   │
│   ├── data_parallel/
│   │   ├── __init__.py
│   │   ├── ddp.py                           # DataParallelGradSync（all-reduce 梯度）
│   │   └── gradient_synchronizer.py         # 工厂函数，根据 sharding strategy 选择 sync 方式
│   │
│   └── pipeline_parallel/
│       ├── action.py                        # PP 调度动作定义（Forward / Backward / SendRecv）
│       ├── comm_ops.py                      # P2P 通信封装（send/recv activation 和 gradient）
│       ├── microbatch.py                    # Microbatch 切分和管理
│       ├── partition.py                     # 模型层到 PP stage 的均匀或自定义分配
│       ├── pipeline_config.py               # PP 配置（num_microbatches, schedule 名称等）
│       ├── runner.py                        # PipelineRunner，编排一个完整 step 的 PP 执行
│       ├── schedule.py                      # baseline 和 1F1B 调度逻辑（warmup / steady / cooldown）
│       ├── interleaved_partition.py         # 模型层到 PP stage 的 Interleaved 均匀或自定义分配
│       ├── interleaved_runner.py            # PipelineRunner，编排一个完整 step 的 PP 执行
│       ├── interleaved_schedule.py          # Interleaved 1F1B 调度逻辑（warmup / steady / cooldown）
│       └── stage.py                         # PipelineStage，本 rank 的 forward/backward 执行
│
├── sharding/
│   ├── __init__.py
│   ├── sharding_strategy.py                 # ShardingStrategy Protocol 定义
│   ├── sharding_spec.py                     # ShardingSpec dataclass（描述单个参数的分片信息）
│   ├── cluster_sharding_spec.py             # 多参数聚合的分片规格（flatten 后的全局视图）
│   ├── param_group_cluster.py               # 参数聚合管理（将多个参数打包用于批量通信和各参数的视图控制）
│   ├── no_shard.py                          # NoShardStrategy（Stage 0，纯 DDP）
│   ├── zero1.py                             # ZeRO1Strategy（optimizer states 分片）
│   ├── zero2.py                             # ZeRO2Strategy（+ 梯度 reduce-scatter）
│   ├── zero3.py                             # ZeRO3Strategy（+ 参数分片 + forward/backward hook）
│   ├── zero_config.py                       # ZeROConfig dataclass
│   ├── wrap_policy.py                       # ZeRO-3 的层级 wrap 策略（决定哪些层注册 hook）
│   └── factory.py                           # create_sharding_strategy 工厂函数
│
├── training/
│   ├── __init__.py
│   ├── param_group.py                       # ParamGroup dataclass（compute / master / spec 映射）
│   ├── mixed_precision_manager.py           # BF16 compute + FP32 master，集成 optimizer
│   ├── grad_accumulator.py                  # 梯度累积 buffer 管理
│   ├── grad_transform.py                    # GradTransform（ClipGradNorm 等梯度变换）
│   ├── optimizer.py                         # get_param_groups（weight decay 分组）
│   ├── lr_schedule.py                       # Linear warmup + cosine decay
│   ├── activation_ckpt.py                   # Activation Checkpointing 包装类
│   ├── ckpt_config.py                       # Activation Checkpointing 配置
│   ├── ckpt_policy.py                       # Activation Checkpointing 策略
│   ├── train_config.py                      # TrainConfig 总配置
│   └── trainer.py                           # 训练循环主逻辑
│
├── scripts/
│   ├── __init__.py
│   ├── presets.py                           # 预设配置（AC policy、wrap policy 等便捷工厂）
│   ├── train.py                             # 训练入口脚本
│   └── verify_training.py                   # 训练结果验证（生成文本 + validation loss）
│
└── test/
    ├── control/
    │   └── ckpt_baseline.py                 # Activation Checkpointing 对照实验
    ├── unit/
    │   ├── test_tp_linear.py                # TP 线性层 forward/backward 正确性
    │   ├── test_model_loading.py            # 模型封装层：替换、shape、权重、forward/backward
    │   ├── test_mixed_precision.py          # 混合精度：master 创建、梯度搬运、同步、FP32 对比
    │   ├── test_cluster_sharding_spec.py    # ClusterShardingSpec 的分片计算正确性
    │   ├── test_param_group_cluster.py      # ParamGroupCluster 的聚合/拆分正确性
    │   ├── test_partial_model_equivalence.py # PP partial model 和完整模型的输出等价性
    │   ├── test_pipeline_comm.py            # PP send/recv 通信正确性
    │   ├── test_pipeline_stage.py           # PipelineStage 的 forward/backward 正确性
    │   ├── test_pp_aware_vs_old.py          # PP-aware 模型 vs 原始模型的等价性
    │   ├── test_pp_basics.py                # PP 基础功能冒烟测试
    │   ├── test_pp_runner.py                # PipelineRunner 端到端正确性
    │   └── test_pp_schedule.py              # 1F1B 调度逻辑的时序正确性
    └── integration/
        ├── test_dp_training.py              # DP 配置间 loss 一致性
        ├── test_gradient_accum.py           # 梯度累积 loss bit-exact
        ├── test_zero1_2_3.py                # ZeRO 全 stage loss 一致性 + 显存验证
        ├── test_zero_ac.py                  # ZeRO × AC 显存和 loss 数学等价
        ├── test_pp_trainer.py               # 3D 并行（PP=2×DP=2×TP=2）完整验证
        └── diagnose_mismatch.py             # 调试工具
```

<!-- TODO: 如果目录结构有变化，在这里更新 -->

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.10
- PyTorch ≥ 2.3（需要 NCCL 支持）
- CUDA ≥ 12.0
- 多卡 GPU（完整功能需要 ≥ 2 卡，3D 并行测试需要 8 卡）

### 安装

```bash
git clone https://github.com/Koas-W/Femtotron.git
cd femtotron
pip install -e .
```

### 数据预处理

```bash
# 处理预训练数据（TinyStories）
python femtotron/data/preprocess.py
```

### 训练

```bash
# 8 卡预训练（DP=4, TP=2）
torchrun --nproc_per_node=8 femtotron/scripts/train.py --config configs/tiny_llama_debug.yaml

# 2 卡纯 DP
torchrun --nproc_per_node=2 femtotron/scripts/train.py --config configs/tiny_llama_debug.yaml --dp 2 --tp 1

# 自定义并行配置（3D 并行）
torchrun --nproc_per_node=8 femtotron/scripts/train.py --config configs/tiny_llama_debug.yaml --dp 2 --tp 2 --pp 2
```

### 运行测试

```bash
# 全部测试（单元 + 集成）
make test

# 仅单元测试
make test-unit

# 仅集成测试
make test-integration
```

---

## 🧪 测试矩阵

所有并行组件都有独立的正确性测试，核心验证标准是 **loss 一致性**。不同并行配置在相同数据上应该产生一致的 loss 曲线。

| 测试 | 验证内容 | GPU 需求 |
|------|---------|---------|
| `test_tp_linear` | ColumnParallel / RowParallel / VocabEmbed 的 forward/backward、Column→Row 链路、TP=1 退化 | ≥ 2 |
| `test_model_loading` | 层替换类型、shape、权重分发、forward/backward 一致性、TP=1 退化、多步训练 | ≥ 1 |
| `test_mixed_precision` | Master weights 创建、梯度搬运、权重同步、grad clipping、BF16 vs FP32 趋势一致 | ≥ 1 |
| `test_cluster_sharding_spec` | ClusterShardingSpec 的分片边界计算、各 rank 分到的 numel 正确性 | ≥ 1 |
| `test_param_group_cluster` | ParamGroupCluster 的 flatten/unflatten、聚合梯度拆分回各参数 | ≥ 1 |
| `test_partial_model_equivalence` | PP partial model（只含部分层）和完整模型在相同层范围上的输出 bit-exact | ≥ 1 |
| `test_pipeline_comm` | PP 跨 stage 的 send/recv 数据完整性、shape/dtype 一致性 | ≥ 2 |
| `test_pipeline_stage` | PipelineStage 的单 microbatch forward/backward 正确性、梯度回传 | ≥ 2 |
| `test_pp_aware_vs_old` | PP-aware 模型构建 vs 原始 HuggingFace 模型的 forward 输出等价性 | ≥ 1 |
| `test_pp_basics` | PP 基础功能冒烟测试（模型切分、stage 分配、端到端正确性） | ≥ 2 |
| `test_pp_runner` | PipelineRunner 多 microbatch 端到端正确性、loss 聚合 | ≥ 2 |
| `test_pp_schedule` | 1F1B 调度的时序正确性（warmup/steady/cooldown 阶段的 F/B 次数和顺序） | ≥ 1 |

### 集成测试

| 测试 | 验证内容 | GPU 需求 |
|------|---------|---------|
| `test_dp_training` | DP=1/TP=N vs DP=N/TP=1 的 loss 一致性 | ≥ 2 |
| `test_gradient_accum` | mbs=8/accum=1 vs mbs=4/accum=2 vs mbs=2/accum=4 的 loss bit-exact | ≥ 2 |
| `test_zero1_2_3` | ZeRO 全 stage 的 loss 一致性、显存节省、master 分片、weight 一致性 | ≥ 4 |
| `test_zero_ac` | ZeRO × AC 的显存对比和 loss 数学等价性 | ≥ 4 |
| `test_pp_trainer` | **PP=2 × DP=2 × TP=2** 下 ZeRO-0/1/2/3 × AC 的完整 3D 并行验证 | 8 |

---

## 📊 验证结果

### 峰值显存（MB）— 纯 DP 场景

> PP=1, TP=1, DP=8 · H=1024, H_ff=2048, heads=16, kv_heads=4, layers=8, vocab=1024 · 变化 seq_len

| Config | seq=16 | seq=32 | seq=1024 |
|--------|-------:|-------:|---------:|
| baseline | 1653 | 1653 | 3631 |
| baseline + AC | 1653 | 1653 | 1669 |
| ZeRO-1 | 460 | 471 | 2913 |
| ZeRO-1 + AC | 460 | 460 | 901 |
| ZeRO-2 | 443 | 444 | 2776 |
| ZeRO-2 + AC | 443 | 444 | 765 |
| ZeRO-3 | 384 | 419 | 2840 |
| **ZeRO-3 + AC** | **261** | **261** | **711** |

### 峰值显存（MB）— 3D 并行场景

> PP=2, DP=2, TP=2 (8 卡) · H=2048, H_ff=8192, heads=32, kv_heads=8, vocab=40960, seq=32 · 变化 num_layers

| Config | N=8 (4/stage) | N=32 (16/stage) | N=64 (32/stage) |
|--------|-------:|-------:|-------:|
| ZeRO-0 | 3601 | 12558 | 24404 |
| ZeRO-0 + AC | 3601 | 11954 | 23091 |
| ZeRO-1 | 2136 | 9534 | 18595 |
| ZeRO-1 + AC | 2136 | 6993 | 13490 |
| ZeRO-2 | 2044 | 9030 | 17627 |
| ZeRO-2 + AC | 2044 | 6569 | 12602 |
| ZeRO-3 | 2233 | 10808 | 20829 |
| **ZeRO-3 + AC** | **2216** | **6417** | **12018** |】

<!-- TODO: 填入 SFT 的实际 benchmark 结果 -->

<!-- 
**SFT**（Qwen2.5-7B, MetaMathQA, 8 卡）：

```
           GSM8K     MATH
基座模型    55%       25%
SFT 后     78%       47%
```
-->

---

## 🗺️ 开发路线

- [x] **Phase 1：核心骨架组件**
  - [x] ParallelContext + TP + 模型封装 + 混合精度
  - [x] 训练循环 + DDP + 梯度累积
  - [x] ZeRO Stage 1 / 2 / 3
  - [x] Activation Checkpointing
  - [x] Pipeline Parallel (1F1B)
  - [x] 3D 并行组合验证
  - [x] SFT 兼容
- [ ] **Phase 2：增强组件**
  - [ ] Interleaved 1F1B / Zero Bubble 调度
  - [ ] Sequence Parallel
  - [ ] 通信-计算 Overlap
  - [ ] LoRA
  - [ ] FP8 训练
- [ ] **Phase 3：验证和实际训练产出**
  - [ ] Qwen2.5-7B + MetaMathQA SFT
  - [ ] GSM8K / MATH benchmark
  - [ ] 性能报告（MFU, tokens/sec/GPU）

---

## 📖 开发日志

<!-- TODO: 链接到你的博客文章 -->

1. [Femtotron开发日志系列](https://koas-w.github.io/categories/%E5%BC%80%E5%8F%91%E6%97%A5%E5%BF%97/)
2. [LLM学习日志系列](https://koas-w.github.io/categories/%E5%AD%A6%E4%B9%A0%E6%97%A5%E5%BF%97/)

---

## 致谢

Femtotron 的设计参考了以下项目的架构思路，但所有代码均为独立实现：

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — Tensor / Pipeline Parallel 的经典设计
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) — ZeRO 系列论文
- [Nanotron](https://github.com/huggingface/nanotron) — 简洁的训练框架参考
- [Picotron](https://github.com/huggingface/picotron) — 教育导向的 4D 并行实现

---

## License

MIT License