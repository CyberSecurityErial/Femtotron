"""Temporary scheduler-v2 scaffold for one or more local pipeline stages.

PipelineExecutor exists to validate PPActionV2, TaskKey, StageMapping, multiple
local PipelineStage instances, and local handoff behavior without disturbing
the existing PipelineRunner path. It is not a second long-term runner
abstraction and should not become the integration point for production
training.

When the scheduler-v2 flows stabilize, the reusable pieces here should move
behind the existing runner as a dispatch/local-stage-routing backend. Do not
grow schedule-specific runners or a parallel executor stack around this
scaffold.

It owns no activation or gradient state; per-microbatch tensor state still
lives in PipelineStage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol

import torch

from femtotron.parallel.pipeline_parallel.scheduler.ir import (
    BackwardInputGradV2,
    BackwardV2,
    BackwardWeightGradV2,
    ForwardV2,
    OverlapForwardBackwardV2,
    PPActionV2,
    RecvBackwardV2,
    RecvForwardV2,
    SendBackwardRecvForwardV2,
    SendBackwardV2,
    SendForwardRecvBackwardV2,
    SendForwardV2,
    TaskKey,
    WaitCommV2,
)
from femtotron.parallel.pipeline_parallel.scheduler.mapping import StageMapping
from femtotron.parallel.pipeline_parallel.stage import MicrobatchKey, PipelineStage


class PipelineCommLike(Protocol):
    """Blocking pipeline communication interface used by PipelineExecutor."""

    def send_forward(self, act: torch.Tensor) -> None:
        ...

    def recv_forward(self, out: torch.Tensor | None = None) -> torch.Tensor | None:
        ...

    def send_backward(self, grad: torch.Tensor) -> None:
        ...

    def recv_backward(self, out: torch.Tensor | None = None) -> torch.Tensor | None:
        ...

    def send_forward_recv_backward(
        self,
        act: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        ...

    def send_backward_recv_forward(
        self,
        grad: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        ...


MicrobatchPayloads = Mapping[MicrobatchKey, torch.Tensor]


def _format_microbatch_key(mb: MicrobatchKey) -> str:
    if isinstance(mb, TaskKey):
        return f"TaskKey({mb.short()})"
    return str(mb)


def _format_available_mbs(payloads: MicrobatchPayloads) -> list[str]:
    return sorted(_format_microbatch_key(mb) for mb in payloads.keys())


def _infer_device(stages: list[PipelineStage]) -> torch.device:
    for stage in stages:
        for param in stage.model.parameters(recurse=True):
            return param.device

    for stage in stages:
        for buffer in stage.model.buffers(recurse=True):
            return buffer.device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PipelineExecutor:
    """Execute scheduler-v2 actions as a temporary runner-side scaffold."""

    def __init__(
        self,
        stages: list[PipelineStage],
        comm: PipelineCommLike,
        mapping: StageMapping,
        *,
        pp_rank: int,
        num_microbatches: int,
        recv_shape: tuple[int, ...],
        recv_dtype: torch.dtype,
    ) -> None:
        if not stages:
            raise ValueError("PipelineExecutor requires at least one local stage")
        if pp_rank < 0 or pp_rank >= mapping.num_physical_ranks:
            raise ValueError(
                f"pp_rank must be in [0, {mapping.num_physical_ranks}), "
                f"got {pp_rank}"
            )
        if num_microbatches < 1:
            raise ValueError(
                f"num_microbatches must be >= 1, got {num_microbatches}"
            )

        local_logical_stages = mapping.local_stages(pp_rank)
        if len(stages) != len(local_logical_stages):
            raise ValueError(
                f"number of local stages ({len(stages)}) must match "
                f"mapping.local_stages(pp_rank={pp_rank})={local_logical_stages}"
            )

        self.stages = stages
        self.comm = comm
        self.mapping = mapping
        self.pp_rank = pp_rank
        self.num_microbatches = num_microbatches
        self.recv_shape = recv_shape
        self.recv_dtype = recv_dtype
        self._device = _infer_device(stages)

    def stage_for(self, key: TaskKey) -> PipelineStage:
        """Return the local PipelineStage for key after placement validation."""

        self._validate_local_key(key)
        return self.stages[key.local_stage]

    def run(
        self,
        actions: list[PPActionV2],
        *,
        microbatch_inputs: MicrobatchPayloads | None = None,
        microbatch_labels: MicrobatchPayloads | None = None,
    ) -> dict[MicrobatchKey, torch.Tensor]:
        for action in actions:
            self.dispatch(
                action,
                microbatch_inputs=microbatch_inputs,
                microbatch_labels=microbatch_labels,
            )

        losses = self.pop_all_losses()
        self.assert_clean()
        return losses

    def dispatch(
        self,
        action: PPActionV2,
        *,
        microbatch_inputs: MicrobatchPayloads | None = None,
        microbatch_labels: MicrobatchPayloads | None = None,
    ) -> None:
        if isinstance(action, ForwardV2):
            self._do_forward(
                action.key,
                microbatch_inputs=microbatch_inputs,
                microbatch_labels=microbatch_labels,
            )
        elif isinstance(action, BackwardV2):
            self.stage_for(action.key).backward(action.key)
        elif isinstance(action, BackwardInputGradV2):
            self.stage_for(action.key).backward_input_grad(action.key)
        elif isinstance(action, BackwardWeightGradV2):
            self.stage_for(action.key).backward_weight_grad(action.key)
        elif isinstance(action, RecvForwardV2):
            self._recv_forward(action.key)
        elif isinstance(action, SendForwardV2):
            self._send_forward(action.key)
        elif isinstance(action, RecvBackwardV2):
            self._recv_backward(action.key)
        elif isinstance(action, SendBackwardV2):
            self._send_backward(action.key)
        elif isinstance(action, SendForwardRecvBackwardV2):
            self._send_forward_recv_backward(action.fwd_key, action.bwd_key)
        elif isinstance(action, SendBackwardRecvForwardV2):
            self._send_backward_recv_forward(action.bwd_key, action.fwd_key)
        elif isinstance(action, OverlapForwardBackwardV2):
            raise NotImplementedError(
                "OverlapForwardBackwardV2 execution is deferred to a later commit"
            )
        elif isinstance(action, WaitCommV2):
            raise NotImplementedError(
                "WaitCommV2 requires async comm support and is deferred"
            )
        else:
            raise NotImplementedError(
                f"Unsupported scheduler-v2 action type: {type(action).__name__}"
            )

    def pop_all_losses(self) -> dict[MicrobatchKey, torch.Tensor]:
        losses: dict[MicrobatchKey, torch.Tensor] = {}

        for local_stage, stage in enumerate(self.stages):
            for mb, loss in stage.pop_all_losses().items():
                if mb in losses:
                    raise RuntimeError(
                        f"duplicate loss for microbatch {_format_microbatch_key(mb)} "
                        f"while collecting losses from local stage {local_stage}"
                    )
                losses[mb] = loss

        return losses

    def assert_clean(self) -> None:
        for local_stage, stage in enumerate(self.stages):
            try:
                stage.assert_clean()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"local stage {local_stage} is not clean"
                ) from exc

    def _validate_local_key(self, key: TaskKey) -> None:
        owner = self.mapping.physical_rank(key.logical_stage)
        if owner != self.pp_rank:
            raise RuntimeError(
                f"TaskKey {key.short()} belongs to physical rank {owner}, "
                f"but this executor is for pp_rank {self.pp_rank}"
            )

        expected_local_stage = self.mapping.local_index(
            self.pp_rank,
            key.logical_stage,
        )
        if key.local_stage != expected_local_stage:
            raise RuntimeError(
                f"TaskKey {key.short()} has local_stage={key.local_stage}, "
                f"but mapping expects local_stage={expected_local_stage} "
                f"for logical_stage={key.logical_stage} on pp_rank={self.pp_rank}. "
                f"Schedule builder should call mapping.with_local_stage(key)."
            )

    def _do_forward(
        self,
        key: TaskKey,
        *,
        microbatch_inputs: MicrobatchPayloads | None,
        microbatch_labels: MicrobatchPayloads | None,
    ) -> None:
        stage = self.stage_for(key)

        if stage.is_first:
            x = self._lookup_required_payload(
                microbatch_inputs,
                key,
                name="input",
            )
            stage.stage_input(key, x)

        if stage.is_last:
            label = self._lookup_optional_payload(microbatch_labels, key)
            if label is not None:
                stage.stage_labels(key, label)

        stage.forward(key)

    def _lookup_required_payload(
        self,
        payloads: MicrobatchPayloads | None,
        key: TaskKey,
        *,
        name: str,
    ) -> torch.Tensor:
        payload = self._lookup_optional_payload(payloads, key)
        if payload is not None:
            return payload

        if payloads is None:
            raise ValueError(
                f"missing {name} payloads for TaskKey({key.short()})"
            )
        raise RuntimeError(
            f"missing {name} for TaskKey({key.short()}); "
            f"available={_format_available_mbs(payloads)}"
        )

    def _lookup_optional_payload(
        self,
        payloads: MicrobatchPayloads | None,
        key: TaskKey,
    ) -> torch.Tensor | None:
        if payloads is None:
            return None
        if key in payloads:
            return payloads[key]
        if key.mb in payloads:
            return payloads[key.mb]
        return None

    def _consumer_key(self, key: TaskKey) -> TaskKey | None:
        consumer = self.mapping.consumer_logical_stage(key)
        if consumer is None:
            return None
        return self.mapping.with_local_stage(replace(key, logical_stage=consumer))

    def _producer_key(self, key: TaskKey) -> TaskKey | None:
        producer = self.mapping.producer_logical_stage(key)
        if producer is None:
            return None
        return self.mapping.with_local_stage(replace(key, logical_stage=producer))

    def _is_local_key(self, key: TaskKey) -> bool:
        return self.mapping.physical_rank(key.logical_stage) == self.pp_rank

    def _remote_peer_or_none(self, key: TaskKey | None) -> int | None:
        if key is None:
            return None
        peer = self.mapping.physical_rank(key.logical_stage)
        if peer == self.pp_rank:
            return None
        return peer

    def _send_forward(self, key: TaskKey) -> None:
        stage = self.stage_for(key)
        consumer_key = self._consumer_key(key)
        if consumer_key is None:
            return

        out = stage.get_output(key)
        if self._is_local_key(consumer_key):
            consumer_stage = self.stage_for(consumer_key)
            handoff = out if consumer_stage.is_first else out.detach()
            consumer_stage.stage_input(consumer_key, handoff)
            return

        self.comm.send_forward(out)

    def _recv_forward(self, key: TaskKey) -> None:
        stage = self.stage_for(key)
        producer_key = self._producer_key(key)
        if producer_key is None:
            return
        if self._is_local_key(producer_key):
            return

        buf = self._alloc_recv_buffer()
        recv = self.comm.recv_forward(out=buf)
        stage.stage_input(key, buf if recv is None else recv)

    def _send_backward(self, key: TaskKey) -> None:
        stage = self.stage_for(key)
        producer_key = self._producer_key(key)
        if producer_key is None:
            return

        grad = stage.get_input_grad(key)
        if self._is_local_key(producer_key):
            self.stage_for(producer_key).stage_grad(producer_key, grad)
            return

        self.comm.send_backward(grad)

    def _recv_backward(self, key: TaskKey) -> None:
        stage = self.stage_for(key)
        consumer_key = self._consumer_key(key)
        if consumer_key is None:
            return
        if self._is_local_key(consumer_key):
            return

        buf = self._alloc_recv_buffer()
        recv = self.comm.recv_backward(out=buf)
        stage.stage_grad(key, buf if recv is None else recv)

    def _send_forward_recv_backward(
        self,
        fwd_key: TaskKey,
        bwd_key: TaskKey,
    ) -> None:
        self.stage_for(fwd_key)
        self.stage_for(bwd_key)

        fwd_consumer = self._consumer_key(fwd_key)
        bwd_consumer = self._consumer_key(bwd_key)
        fwd_remote_peer = self._remote_peer_or_none(fwd_consumer)
        bwd_remote_peer = self._remote_peer_or_none(bwd_consumer)

        if fwd_remote_peer is not None and bwd_remote_peer is not None:
            if fwd_remote_peer != bwd_remote_peer:
                raise NotImplementedError(
                    "SFRB with two different remote peers requires async comm "
                    "or explicit multi-peer batching; deferred to a later commit."
                )

            act = self.stage_for(fwd_key).get_output(fwd_key)
            buf = self._alloc_recv_buffer()
            grad = self.comm.send_forward_recv_backward(act, out=buf)
            self.stage_for(bwd_key).stage_grad(
                bwd_key,
                buf if grad is None else grad,
            )
            return

        self._send_forward(fwd_key)
        self._recv_backward(bwd_key)

    def _send_backward_recv_forward(
        self,
        bwd_key: TaskKey,
        fwd_key: TaskKey,
    ) -> None:
        self.stage_for(bwd_key)
        self.stage_for(fwd_key)

        bwd_producer = self._producer_key(bwd_key)
        fwd_producer = self._producer_key(fwd_key)
        bwd_remote_peer = self._remote_peer_or_none(bwd_producer)
        fwd_remote_peer = self._remote_peer_or_none(fwd_producer)

        if bwd_remote_peer is not None and fwd_remote_peer is not None:
            if bwd_remote_peer != fwd_remote_peer:
                raise NotImplementedError(
                    "SBRF with two different remote peers requires async comm "
                    "or explicit multi-peer batching; deferred to a later commit."
                )

            grad = self.stage_for(bwd_key).get_input_grad(bwd_key)
            buf = self._alloc_recv_buffer()
            act = self.comm.send_backward_recv_forward(grad, out=buf)
            self.stage_for(fwd_key).stage_input(
                fwd_key,
                buf if act is None else act,
            )
            return

        self._send_backward(bwd_key)
        self._recv_forward(fwd_key)

    def _alloc_recv_buffer(self) -> torch.Tensor:
        return torch.empty(
            self.recv_shape,
            dtype=self.recv_dtype,
            device=self._device,
        )
