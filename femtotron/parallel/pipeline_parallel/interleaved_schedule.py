"""
femtotron/parallel/pipeline_parallel/interleaved_schedule.py

Interleaved 1F1B schedule (Megatron-LM style).

每个 device 持有 V 个 layer chunks(virtual stages)。
chunk_id_in_dev=c 对应全局 chunk c*P+r(round-robin 分配,见 interleaved_partition)。

核心数学:把单个 microbatch 想象成穿过 P*V 个 virtual stages(不是 P 个)。
Schedule 在每个 device 上是 warmup → steady → cooldown 三阶段,公式见 Megatron。

Decoding(给 virtual_id k):
    mb_in_group   = k % (P*V)
    chunk_forward = mb_in_group // P            # 0..V-1
    chunk_backward = V - 1 - chunk_forward      # 反向,backward 时 chunk 顺序倒过来
    real_mb_id    = (k // (P*V)) * P + (k % P)  # 实际 microbatch 索引 0..N-1

Warmup count for device r:
    num_warmup = min((P - r - 1) * 2 + (V - 1) * P, N * V)

Bubble:
    - vanilla 1F1B:   (P-1) / N
    - interleaved:    (P-1) / (N*V)        ← V 倍 bubble 减少
    - 代价:          P*V comm 频率(V 倍),buffer 也更多

PPAction 编码(零修改 action.py 的关键技巧):
    PPAction.mb_id 字段被复用为 "effective_mb_id":
        eff_mb_id = chunk_id_in_dev * num_microbatches + real_mb_id
    Schedule emit 时 encode,Runner dispatch 时 decode 选 stages[chunk_id]。
    既不增 action 字段,也不增新 action 类型。

Boundary 处理(在 schedule 层做,不留给 Runner):
    - chunk_id_in_dev=0 on rank=0:全局 first chunk → RF/SB 是 no-op,schedule 不 emit
    - chunk_id_in_dev=V-1 on rank=P-1:全局 last chunk → SF/RB 是 no-op,schedule 不 emit
    - 中间 chunks:全部 comm 都 emit(走 ring 拓扑,Runner 用 RingContextAdapter)
    - Combined op (SFRB/SBRF):若两边都需要 → 用 combined;一边 no-op → 拆成单向;
      两边都 no-op → 全 skip。

约束:num_microbatches % pp_size == 0(Megatron 公式假设 mb 整除 group 大小)。

V=1 注意:与原 one_f_one_b_schedule **不等价**(warmup 是 2x),仅为公式连续性保留。
V=1 请用 schedule.py 的 one_f_one_b_schedule。
"""

from __future__ import annotations

from .action import (
    PPAction,
    Forward, Backward,
    RecvForward, SendForward,
    RecvBackward, SendBackward,
    SendForwardRecvBackward, SendBackwardRecvForward,
)


# ════════════════════════════════════════════════════════════════
# Public encoding API(Runner 也要用)
# ════════════════════════════════════════════════════════════════

def encode_mb_id(chunk_id_in_dev: int, real_mb_id: int, num_microbatches: int) -> int:
    """Encode (chunk_id_in_dev, real_mb_id) → effective mb_id for PPAction。

    Inverse: decode_mb_id(eff_mb_id, num_microbatches) → (chunk, mb)。
    """
    return chunk_id_in_dev * num_microbatches + real_mb_id


def decode_mb_id(eff_mb_id: int, num_microbatches: int) -> tuple[int, int]:
    """Decode eff_mb_id → (chunk_id_in_dev, real_mb_id)。"""
    return eff_mb_id // num_microbatches, eff_mb_id % num_microbatches


# ════════════════════════════════════════════════════════════════
# Schedule
# ════════════════════════════════════════════════════════════════

def interleaved_one_f_one_b_schedule(
    num_microbatches: int,
    pp_size: int,
    pp_rank: int,
    virtual_stages: int,
) -> list[PPAction]:
    """Generate Interleaved 1F1B action stream for one device.

    Args:
        num_microbatches: N。**必须能被 pp_size 整除**(Megatron 公式约束)。
        pp_size: P。
        pp_rank: r ∈ [0, P)。
        virtual_stages: V。每设备 chunk 数。V=1 退化但**不等价于 vanilla 1F1B**
            (warmup 是 2x),V=1 请改用 schedule.one_f_one_b_schedule。

    Returns:
        Action 列表。每个 action 的 mb_id 字段是 eff_mb_id(chunk * N + mb)。
        Runner 解码后 dispatch 到对应 stage。

    Raises:
        ValueError: 参数不合法,或 num_microbatches % pp_size != 0(pp_size>1 时)。
    """
    # ── Validation ──
    if num_microbatches < 1:
        raise ValueError(f"num_microbatches must be >= 1, got {num_microbatches}")
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")
    if pp_rank < 0 or pp_rank >= pp_size:
        raise ValueError(f"pp_rank must be in [0, {pp_size}), got {pp_rank}")
    if virtual_stages < 1:
        raise ValueError(f"virtual_stages must be >= 1, got {virtual_stages}")
    if pp_size > 1 and num_microbatches % pp_size != 0:
        raise ValueError(
            f"num_microbatches ({num_microbatches}) must be divisible by "
            f"pp_size ({pp_size}) for interleaved 1F1B. "
            f"Megatron decoder assumes complete microbatch groups."
        )

    P, V, N = pp_size, virtual_stages, num_microbatches
    r = pp_rank
    total_vmb = N * V

    is_first_dev = (r == 0)
    is_last_dev = (r == P - 1)

    # ── Decoders (closures) ──
    def chunk_forward(vid: int) -> int:
        return (vid % (P * V)) // P

    def chunk_backward(vid: int) -> int:
        return V - 1 - (vid % (P * V)) // P

    def real_mb(vid: int) -> int:
        return (vid // (P * V)) * P + (vid % P)

    def eff(chunk: int, mb: int) -> int:
        return chunk * N + mb

    # ── Boundary checks(True = comm is real, False = no-op,跳过 emit) ──
    def sf_needed(chunk: int) -> bool:
        """SendForward 真发 iff F 不是全局 last chunk。"""
        return not (chunk == V - 1 and is_last_dev)

    def rb_needed(chunk: int) -> bool:
        """RecvBackward 真收 iff B 不是全局 last chunk(否则 grad 由 local loss 来)。"""
        return not (chunk == V - 1 and is_last_dev)

    def sb_needed(chunk: int) -> bool:
        """SendBackward 真发 iff B 不是全局 first chunk(否则 upstream 不存在)。"""
        return not (chunk == 0 and is_first_dev)

    def rf_needed(chunk: int) -> bool:
        """RecvForward 真收 iff F 不是全局 first chunk(否则 input 来自 dataloader)。"""
        return not (chunk == 0 and is_first_dev)

    # ── Phase counts(Megatron 公式) ──
    num_warmup = min((P - r - 1) * 2 + (V - 1) * P, total_vmb)
    num_steady = total_vmb - num_warmup
    num_cooldown = num_warmup  # symmetric

    actions: list[PPAction] = []

    # ════════════════════════════════════════════════════════════════
    # Phase 1: Warmup(pure forwards,fill the pipeline)
    # ════════════════════════════════════════════════════════════════
    for k in range(num_warmup):
        vid = k
        c = chunk_forward(vid)
        mb = real_mb(vid)
        e = eff(c, mb)

        if rf_needed(c):
            actions.append(RecvForward(mb_id=e))
        actions.append(Forward(mb_id=e))
        if sf_needed(c):
            actions.append(SendForward(mb_id=e))

    # ════════════════════════════════════════════════════════════════
    # Phase 2: Steady-state 1F1B(每 iter 一个 F + 一个 B,交错排放)
    # ════════════════════════════════════════════════════════════════
    for k in range(num_steady):
        fwd_vid = num_warmup + k
        bwd_vid = k

        fwd_c = chunk_forward(fwd_vid)
        bwd_c = chunk_backward(bwd_vid)
        fwd_m = real_mb(fwd_vid)
        bwd_m = real_mb(bwd_vid)
        fwd_e = eff(fwd_c, fwd_m)
        bwd_e = eff(bwd_c, bwd_m)

        is_first_steady = (k == 0)
        is_last_steady = (k == num_steady - 1)

        # First steady iter:fwd 的 input 还没收过,显式 RF
        # (后续 steady iter 的 RF 由上一次的 SBRF 完成)
        if is_first_steady and rf_needed(fwd_c):
            actions.append(RecvForward(mb_id=fwd_e))

        actions.append(Forward(mb_id=fwd_e))

        # SF + RB:两边都 real → combined;一边 no-op → 单向;两边 no-op → skip
        sf = sf_needed(fwd_c)
        rb = rb_needed(bwd_c)
        if sf and rb:
            actions.append(SendForwardRecvBackward(fwd_mb=fwd_e, bwd_mb=bwd_e))
        elif sf:
            actions.append(SendForward(mb_id=fwd_e))
        elif rb:
            actions.append(RecvBackward(mb_id=bwd_e))
        # else: 全 no-op,skip

        actions.append(Backward(mb_id=bwd_e))

        # SB + RF(next iter 的 fwd 提前收 input)
        sb = sb_needed(bwd_c)
        if is_last_steady:
            # No next F to receive for;just maybe SB
            if sb:
                actions.append(SendBackward(mb_id=bwd_e))
        else:
            next_fwd_vid = fwd_vid + 1
            next_fwd_c = chunk_forward(next_fwd_vid)
            next_fwd_m = real_mb(next_fwd_vid)
            next_fwd_e = eff(next_fwd_c, next_fwd_m)
            rf_next = rf_needed(next_fwd_c)

            if sb and rf_next:
                actions.append(SendBackwardRecvForward(bwd_mb=bwd_e, fwd_mb=next_fwd_e))
            elif sb:
                actions.append(SendBackward(mb_id=bwd_e))
            elif rf_next:
                actions.append(RecvForward(mb_id=next_fwd_e))
            # else: 全 no-op,skip

    # ════════════════════════════════════════════════════════════════
    # Phase 3: Cooldown(pure backwards,drain the pipeline)
    # ════════════════════════════════════════════════════════════════
    for j in range(num_cooldown):
        bwd_vid = num_steady + j
        c = chunk_backward(bwd_vid)
        mb = real_mb(bwd_vid)
        e = eff(c, mb)

        if rb_needed(c):
            actions.append(RecvBackward(mb_id=e))
        actions.append(Backward(mb_id=e))
        if sb_needed(c):
            actions.append(SendBackward(mb_id=e))

    return actions