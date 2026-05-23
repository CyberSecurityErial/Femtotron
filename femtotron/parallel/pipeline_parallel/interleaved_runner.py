"""
femtotron/parallel/pipeline_parallel/interleaved_runner.py

Interleaved 1F1B runner:持有 V 个 PipelineStage,按 schedule dispatch。

vs 原 PipelineRunner 的关键差异:
    1. 多 stage:self.stages: list[PipelineStage],长度 = V
       每个对应 chunk_id_in_dev=0..V-1 的 model partial。
    2. mb_id 解码:PPAction.mb_id 是 eff_mb_id,decode 后定位到 stages[chunk_id]。
       Stage 内部 dict 用 real_mb_id 作 key(各 stage 互不冲突)。
    3. Ring 拓扑 comm:内部 PipelineComm 用 _RingContextAdapter 构造,
       pp_prev_rank/pp_next_rank 永远是 ring 邻居((rank±1) % P 的全局 rank),
       不再有 None 边界——boundary no-op 由 schedule 通过不 emit action 处理。
    4. Inputs/Labels 路由:
       - 仅 rank=0 的 stages[0] 接 microbatch_inputs(全局 first chunk)
       - 仅 rank=P-1 的 stages[V-1] 接 microbatch_labels(全局 last chunk)

零修改:不改 PipelineComm / PipelineStage / 原 runner.py / action.py / schedule.py。
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .action import (
    PPAction,
    Forward, Backward,
    RecvForward, SendForward,
    RecvBackward, SendBackward,
    SendForwardRecvBackward, SendBackwardRecvForward,
)
from .stage import PipelineStage
from .comm_ops import PipelineComm
from .microbatch import split_microbatches
from .interleaved_schedule import (
    interleaved_one_f_one_b_schedule,
    decode_mb_id,
)


# ════════════════════════════════════════════════════════════════
# Ring-mode context adapter
# ════════════════════════════════════════════════════════════════

class _RingContextAdapter:
    """Wrap a ParallelContext so PipelineComm sees ring topology.

    PipelineComm 在 send_forward / recv_backward 等方法里检查
        self.ctx.pp_prev_rank is None  → first stage, no-op
        self.ctx.pp_next_rank is None  → last stage, no-op

    Interleaved 1F1B 需要 ring 拓扑((rank+1)%P 发,(rank-1)%P 收),且 boundary
    no-op 已经由 schedule 处理(boundary 的 comm action 根本不 emit),所以
    pp_prev_rank / pp_next_rank 永远不应该是 None。

    这个 adapter 覆盖这两个属性,其余属性透传到底层 ctx。
    无侵入,无新依赖,完全用现成 PipelineComm。
    """

    def __init__(self, parallel_ctx) -> None:
        self._ctx = parallel_ctx
        # 用真正的 global ranks 算 ring peers
        pp_group_ranks = dist.get_process_group_ranks(parallel_ctx.pp_group)
        pp_size = len(pp_group_ranks)
        pp_rank = parallel_ctx.pp_rank
        # ring 邻居 = group 内 (pp_rank ± 1) % pp_size 位置的 global rank
        self.pp_prev_rank = pp_group_ranks[(pp_rank - 1) % pp_size]
        self.pp_next_rank = pp_group_ranks[(pp_rank + 1) % pp_size]

    def __getattr__(self, name: str):
        # 所有其它属性(pp_group, world_rank, pp_size, pp_rank, ...)透传
        return getattr(self._ctx, name)


# ════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════

class InterleavedPipelineRunner:
    """Execute Interleaved 1F1B against V PipelineStages on one device.

    Args:
        stages: 长度 V 的 PipelineStage 列表,stages[c] 对应 chunk_id_in_dev=c。
            构造约束:
                - rank=0 上 stages[0].is_first 必须 True(全局 first chunk,持 embedding)
                - rank=P-1 上 stages[V-1].is_last 必须 True(全局 last chunk,持 lm_head)
                - 其余 stage 都不应该是 first/last
        parallel_ctx: 提供 pp_group / pp_rank / pp_size / world_rank。
        num_microbatches: N。
        virtual_stages: V,必须 == len(stages)。
        seq_len, hidden_size: 用于构造内部 ring-mode PipelineComm 的 buffer 描述。
        recv_shape: (mb_size, seq_len, hidden_size),用于 Runner 自己 alloc recv buffer
            (跟原 PipelineRunner 完全一样的 API)。
        recv_dtype: comm 用 dtype(默认 bf16)。

    生命周期对齐 PipelineRunner:
        runner.run_step(batch)  → 一个 global batch 的完整 forward+backward
        返回 dict[real_mb_id → scalar_loss],仅 last device 非空。
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        parallel_ctx,
        *,
        num_microbatches: int,
        virtual_stages: int,
        recv_shape: tuple[int, ...],
        seq_len: int,
        hidden_size: int,
        recv_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if len(stages) != virtual_stages:
            raise ValueError(
                f"len(stages)={len(stages)} != virtual_stages={virtual_stages}"
            )
        if virtual_stages < 1:
            raise ValueError(f"virtual_stages must be >= 1, got {virtual_stages}")

        self.stages = stages
        self.ctx = parallel_ctx
        self.num_microbatches = num_microbatches
        self.virtual_stages = virtual_stages
        self.recv_shape = recv_shape
        self.recv_dtype = recv_dtype

        self.is_first_dev = (parallel_ctx.pp_rank == 0)
        self.is_last_dev = (parallel_ctx.pp_rank == parallel_ctx.pp_size - 1)

        # ── 边界 stage 的 is_first / is_last 校验 ──
        if self.is_first_dev and not stages[0].is_first:
            raise ValueError(
                "On rank 0, stages[0] must have is_first=True "
                "(it's the global first chunk; needs embedding layer). "
                "Check model construction for chunk_id_in_dev=0."
            )
        if self.is_last_dev and not stages[virtual_stages - 1].is_last:
            raise ValueError(
                f"On rank P-1, stages[{virtual_stages-1}] must have is_last=True "
                f"(it's the global last chunk; needs lm_head + loss). "
                f"Check model construction for chunk_id_in_dev={virtual_stages-1}."
            )
        for c, st in enumerate(stages):
            is_global_first = (self.is_first_dev and c == 0)
            is_global_last = (self.is_last_dev and c == virtual_stages - 1)
            if st.is_first and not is_global_first:
                raise ValueError(
                    f"stages[{c}] (rank={parallel_ctx.pp_rank}) has is_first=True "
                    f"but it's not the global first chunk. Only chunk 0 on rank 0 "
                    f"should be is_first; check model construction."
                )
            if st.is_last and not is_global_last:
                raise ValueError(
                    f"stages[{c}] (rank={parallel_ctx.pp_rank}) has is_last=True "
                    f"but it's not the global last chunk. Only chunk V-1 on rank P-1 "
                    f"should be is_last; check model construction."
                )

        # ── 内部 ring-mode comm ──
        ring_ctx = _RingContextAdapter(parallel_ctx)
        self.comm = PipelineComm(
            ring_ctx, seq_len, hidden_size, dtype=recv_dtype,
        )

        # ── 一次性构造 action stream(每 step 复用) ──
        self.actions = interleaved_one_f_one_b_schedule(
            num_microbatches,
            parallel_ctx.pp_size,
            parallel_ctx.pp_rank,
            virtual_stages,
        )

        # 所有 stage 应该在同一 device;取第一个 stage 的 device
        self._device = next(stages[0].model.parameters()).device

    # ════════════════════════════════════════════════════════════════
    # High-level API: Trainer 调用
    # ════════════════════════════════════════════════════════════════

    def run_step(self, batch: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Run one global batch:forward + backward across all microbatches/chunks。"""
        inputs_dict = labels_dict = None
        if self.is_first_dev:
            input_mbs = split_microbatches(batch["input_ids"], self.num_microbatches)
            inputs_dict = {i: mb for i, mb in enumerate(input_mbs)}
        if self.is_last_dev:
            label_mbs = split_microbatches(batch["labels"], self.num_microbatches)
            labels_dict = {i: mb for i, mb in enumerate(label_mbs)}
        return self.run(
            self.actions,
            recv_shape=self.recv_shape,
            recv_dtype=self.recv_dtype,
            microbatch_inputs=inputs_dict,
            microbatch_labels=labels_dict,
        )

    # ════════════════════════════════════════════════════════════════
    # Low-level API: 测试 / 高级用法
    # ════════════════════════════════════════════════════════════════

    def run(
        self,
        actions: list[PPAction],
        *,
        recv_shape: tuple[int, ...],
        recv_dtype: torch.dtype,
        microbatch_inputs: dict[int, torch.Tensor] | None = None,
        microbatch_labels: dict[int, torch.Tensor] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Execute action stream;return losses(only on last device)。"""
        if self.is_first_dev and microbatch_inputs is None:
            raise ValueError("First device requires microbatch_inputs dict; got None")

        for action in actions:
            self._dispatch(
                action, microbatch_inputs, microbatch_labels,
                recv_shape, recv_dtype,
            )

        # Collect losses(仅在 global last chunk 的 stage 上有)
        losses: dict[int, torch.Tensor] = {}
        if self.is_last_dev:
            losses = self.stages[self.virtual_stages - 1].pop_all_losses()

        # Sanity:所有 stage 状态干净
        for c, st in enumerate(self.stages):
            try:
                st.assert_clean()
            except RuntimeError as exc:
                raise RuntimeError(f"stages[{c}] not clean after run: {exc}") from exc

        return losses

    # ════════════════════════════════════════════════════════════════
    # Compatibility shims for Trainer(跟 PipelineRunner 同接口)
    # ════════════════════════════════════════════════════════════════

    @property
    def is_first_stage(self) -> bool:
        return self.is_first_dev

    @property
    def is_last_stage(self) -> bool:
        return self.is_last_dev

    # ════════════════════════════════════════════════════════════════
    # Private
    # ════════════════════════════════════════════════════════════════

    def _decode(self, eff_mb_id: int) -> tuple[int, int]:
        return decode_mb_id(eff_mb_id, self.num_microbatches)

    def _alloc_buf(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=self._device)

    def _dispatch(
        self,
        action: PPAction,
        inputs: dict[int, torch.Tensor] | None,
        labels: dict[int, torch.Tensor] | None,
        recv_shape: tuple[int, ...],
        recv_dtype: torch.dtype,
    ) -> None:
        # ── Compute actions ──
        if isinstance(action, Forward):
            chunk, mb = self._decode(action.mb_id)
            stage = self.stages[chunk]
            # Global first chunk:从 inputs_dict 拿输入
            if chunk == 0 and self.is_first_dev:
                if inputs is None or mb not in inputs:
                    raise RuntimeError(
                        f"Forward(eff={action.mb_id}, chunk=0, real_mb={mb}): "
                        f"missing input on global first chunk; "
                        f"available={sorted(inputs.keys()) if inputs else 'None'}"
                    )
                stage.stage_input(mb, inputs[mb])
            # Global last chunk:从 labels_dict 拿 labels(可选,推理模式可无)
            if chunk == self.virtual_stages - 1 and self.is_last_dev:
                if labels is not None and mb in labels:
                    stage.stage_labels(mb, labels[mb])
            stage.forward(mb)

        elif isinstance(action, Backward):
            chunk, mb = self._decode(action.mb_id)
            self.stages[chunk].backward(mb)

        # ── Single-direction comm ──
        elif isinstance(action, RecvForward):
            chunk, mb = self._decode(action.mb_id)
            buf = self._alloc_buf(recv_shape, recv_dtype)
            self.comm.recv_forward(out=buf)
            self.stages[chunk].stage_input(mb, buf)

        elif isinstance(action, SendForward):
            chunk, mb = self._decode(action.mb_id)
            out = self.stages[chunk].get_output(mb)
            self.comm.send_forward(out)

        elif isinstance(action, RecvBackward):
            chunk, mb = self._decode(action.mb_id)
            buf = self._alloc_buf(recv_shape, recv_dtype)
            self.comm.recv_backward(out=buf)
            self.stages[chunk].stage_grad(mb, buf)

        elif isinstance(action, SendBackward):
            chunk, mb = self._decode(action.mb_id)
            g = self.stages[chunk].get_input_grad(mb)
            self.comm.send_backward(g)

        # ── Combined comm(fwd 和 bwd 可能跨不同 chunk/stage) ──
        elif isinstance(action, SendForwardRecvBackward):
            fwd_chunk, fwd_mb = self._decode(action.fwd_mb)
            bwd_chunk, bwd_mb = self._decode(action.bwd_mb)
            act = self.stages[fwd_chunk].get_output(fwd_mb)
            buf = self._alloc_buf(recv_shape, recv_dtype)
            self.comm.send_forward_recv_backward(act, out=buf)
            self.stages[bwd_chunk].stage_grad(bwd_mb, buf)

        elif isinstance(action, SendBackwardRecvForward):
            bwd_chunk, bwd_mb = self._decode(action.bwd_mb)
            fwd_chunk, fwd_mb = self._decode(action.fwd_mb)
            grad = self.stages[bwd_chunk].get_input_grad(bwd_mb)
            buf = self._alloc_buf(recv_shape, recv_dtype)
            self.comm.send_backward_recv_forward(grad, out=buf)
            self.stages[fwd_chunk].stage_input(fwd_mb, buf)

        else:
            raise NotImplementedError(
                f"Unsupported action: {type(action).__name__}"
            )