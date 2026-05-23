"""
femtotron/parallel/pipeline_parallel/interleaved_partition.py

Layer partitioning for Interleaved 1F1B (Megatron-style).

每个设备持有 V 个**不连续的** layer chunks(也叫 virtual stages / model chunks)。
全局 P*V 个 chunk 按 round-robin 分配:全局 chunk_id=k 落在 device (k % P) 上,
device 上的 chunk_id_in_dev = k // P。

例子 (num_layers=16, pp_size=4, virtual_stages=2):
    全局 chunk 边界(每个 2 层):
        chunk0=[0,1]  chunk1=[2,3]  chunk2=[4,5]  chunk3=[6,7]
        chunk4=[8,9]  chunk5=[10,11] chunk6=[12,13] chunk7=[14,15]
    设备分配(round-robin):
        Device 0: chunk_id_in_dev=0 → [0,1]  ; chunk_id_in_dev=1 → [8,9]
        Device 1: chunk_id_in_dev=0 → [2,3]  ; chunk_id_in_dev=1 → [10,11]
        Device 2: chunk_id_in_dev=0 → [4,5]  ; chunk_id_in_dev=1 → [12,13]
        Device 3: chunk_id_in_dev=0 → [6,7]  ; chunk_id_in_dev=1 → [14,15]

Forward 数据流:chunk0 → chunk1 → ... → chunk7,
设备路径:dev0 → dev1 → dev2 → dev3 → dev0 → dev1 → ...
所以 PP comm 必须是 **ring 拓扑**(不是 linear);wraparound (dev3 → dev0) 由
InterleavedPipelineRunner 通过一个 RingContextAdapter 让 PipelineComm 看到
ring 邻居。本文件只负责 layer 划分。
"""

from __future__ import annotations


def interleaved_partition_layers(
    num_layers: int,
    pp_size: int,
    virtual_stages: int,
) -> list[list[range]]:
    """Partition layers for interleaved 1F1B.

    Args:
        num_layers: 总层数 L。**必须能被 pp_size * virtual_stages 整除**——
            这是 Megatron interleaved 的标准约定(每个 chunk 同样大小,
            schedule 公式才成立)。若 L 不整除,显式报错而非默默不均。
        pp_size: pipeline 并行度 P。
        virtual_stages: 每设备 chunk 数 V(= Megatron 的 num_model_chunks)。

    Returns:
        长度 pp_size 的列表。result[r] 是 device r 持有的 V 个 layer ranges,
        按 chunk_id_in_dev 升序排列:result[r][c] 是 device r 上 chunk_id_in_dev=c
        对应的全局 layer 索引 range。

        全局 chunk 编号关系: global_chunk_id = c * pp_size + r。

    Raises:
        ValueError: 参数不合法,或 num_layers 不能被 P*V 整除。
    """
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")
    if virtual_stages < 1:
        raise ValueError(f"virtual_stages must be >= 1, got {virtual_stages}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    total_chunks = pp_size * virtual_stages
    if num_layers % total_chunks != 0:
        raise ValueError(
            f"num_layers ({num_layers}) must be divisible by "
            f"pp_size * virtual_stages ({pp_size} * {virtual_stages} = {total_chunks}). "
            f"Megatron interleaved 1F1B assumes uniform chunk size; "
            f"unbalanced partitions would require schedule formula changes."
        )

    layers_per_chunk = num_layers // total_chunks

    # result[r][c] = layer range for chunk_id_in_dev=c on device r
    result: list[list[range]] = [[] for _ in range(pp_size)]
    for global_chunk_id in range(total_chunks):
        device = global_chunk_id % pp_size
        # chunk_id_in_dev = global_chunk_id // pp_size (由 append 顺序保证)
        start = global_chunk_id * layers_per_chunk
        result[device].append(range(start, start + layers_per_chunk))

    # Sanity: 每个 device 都有刚好 V 个 chunk
    for r, chunks in enumerate(result):
        assert len(chunks) == virtual_stages, (
            f"internal: device {r} has {len(chunks)} chunks, expected {virtual_stages}"
        )

    return result