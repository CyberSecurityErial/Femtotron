"""Interleaved 1F1B flow with explicit local-stage metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True, slots=True)
class VPPSingleAction:
    """Base value object for VPP actions on one local stage."""

    local_stage: int
    mb_id: int

    def __post_init__(self) -> None:
        _validate_non_negative("local_stage", self.local_stage)
        _validate_non_negative("mb_id", self.mb_id)


@dataclass(frozen=True, slots=True)
class VPPRecvForward(VPPSingleAction):
    def __repr__(self) -> str:
        return f"VRF(l{self.local_stage},mb{self.mb_id})"


@dataclass(frozen=True, slots=True)
class VPPForward(VPPSingleAction):
    def __repr__(self) -> str:
        return f"VF(l{self.local_stage},mb{self.mb_id})"


@dataclass(frozen=True, slots=True)
class VPPSendForward(VPPSingleAction):
    def __repr__(self) -> str:
        return f"VSF(l{self.local_stage},mb{self.mb_id})"


@dataclass(frozen=True, slots=True)
class VPPRecvBackward(VPPSingleAction):
    def __repr__(self) -> str:
        return f"VRB(l{self.local_stage},mb{self.mb_id})"


@dataclass(frozen=True, slots=True)
class VPPBackward(VPPSingleAction):
    def __repr__(self) -> str:
        return f"VB(l{self.local_stage},mb{self.mb_id})"


@dataclass(frozen=True, slots=True)
class VPPSendBackward(VPPSingleAction):
    def __repr__(self) -> str:
        return f"VSB(l{self.local_stage},mb{self.mb_id})"


@dataclass(frozen=True, slots=True)
class VPPSendForwardRecvBackward:
    fwd_local_stage: int
    fwd_mb: int
    bwd_local_stage: int
    bwd_mb: int

    def __post_init__(self) -> None:
        _validate_non_negative("fwd_local_stage", self.fwd_local_stage)
        _validate_non_negative("fwd_mb", self.fwd_mb)
        _validate_non_negative("bwd_local_stage", self.bwd_local_stage)
        _validate_non_negative("bwd_mb", self.bwd_mb)

    def __repr__(self) -> str:
        return (
            "VSFRB("
            f"f=l{self.fwd_local_stage}:mb{self.fwd_mb},"
            f"b=l{self.bwd_local_stage}:mb{self.bwd_mb}"
            ")"
        )


@dataclass(frozen=True, slots=True)
class VPPSendBackwardRecvForward:
    bwd_local_stage: int
    bwd_mb: int
    fwd_local_stage: int
    fwd_mb: int

    def __post_init__(self) -> None:
        _validate_non_negative("bwd_local_stage", self.bwd_local_stage)
        _validate_non_negative("bwd_mb", self.bwd_mb)
        _validate_non_negative("fwd_local_stage", self.fwd_local_stage)
        _validate_non_negative("fwd_mb", self.fwd_mb)

    def __repr__(self) -> str:
        return (
            "VSBRF("
            f"b=l{self.bwd_local_stage}:mb{self.bwd_mb},"
            f"f=l{self.fwd_local_stage}:mb{self.fwd_mb}"
            ")"
        )


VPPAction: TypeAlias = (
    VPPRecvForward
    | VPPForward
    | VPPSendForward
    | VPPRecvBackward
    | VPPBackward
    | VPPSendBackward
    | VPPSendForwardRecvBackward
    | VPPSendBackwardRecvForward
)


def build_interleaved_one_f_one_b_flow(
    *,
    num_microbatches: int,
    pp_size: int,
    pp_rank: int,
    virtual_stages: int,
) -> list[VPPAction]:
    """Return interleaved 1F1B actions with real microbatch ids."""
    if num_microbatches <= 0:
        raise ValueError(
            f"num_microbatches must be positive, got {num_microbatches}"
        )
    if pp_size <= 0:
        raise ValueError(f"pp_size must be positive, got {pp_size}")
    if pp_rank < 0 or pp_rank >= pp_size:
        raise ValueError(f"pp_rank must be in [0, {pp_size}), got {pp_rank}")
    if virtual_stages < 2:
        raise ValueError(
            "virtual_stages must be >= 2; use build_one_f_one_b_flow for V=1"
        )
    if pp_size > 1 and num_microbatches % pp_size != 0:
        raise ValueError(
            f"num_microbatches ({num_microbatches}) must be divisible by "
            f"pp_size ({pp_size}) for interleaved 1F1B"
        )

    P, V, N = pp_size, virtual_stages, num_microbatches
    r = pp_rank
    total_vmb = N * V

    is_first_dev = r == 0
    is_last_dev = r == P - 1

    def local_stage_for_forward(vmb: int) -> int:
        return (vmb % (P * V)) // P

    def local_stage_for_backward(vmb: int) -> int:
        return V - 1 - ((vmb % (P * V)) // P)

    def real_microbatch(vmb: int) -> int:
        return (vmb // (P * V)) * P + (vmb % P)

    def sf_needed(local_stage: int) -> bool:
        return not (local_stage == V - 1 and is_last_dev)

    def rb_needed(local_stage: int) -> bool:
        return not (local_stage == V - 1 and is_last_dev)

    def sb_needed(local_stage: int) -> bool:
        return not (local_stage == 0 and is_first_dev)

    def rf_needed(local_stage: int) -> bool:
        return not (local_stage == 0 and is_first_dev)

    num_warmup = min((P - r - 1) * 2 + (V - 1) * P, total_vmb)
    num_steady = total_vmb - num_warmup
    num_cooldown = num_warmup

    actions: list[VPPAction] = []

    def append_recv_forward(vmb: int) -> None:
        actions.append(
            VPPRecvForward(
                local_stage=local_stage_for_forward(vmb),
                mb_id=real_microbatch(vmb),
            )
        )

    def append_forward(vmb: int) -> None:
        actions.append(
            VPPForward(
                local_stage=local_stage_for_forward(vmb),
                mb_id=real_microbatch(vmb),
            )
        )

    def append_send_forward(vmb: int) -> None:
        actions.append(
            VPPSendForward(
                local_stage=local_stage_for_forward(vmb),
                mb_id=real_microbatch(vmb),
            )
        )

    def append_recv_backward(vmb: int) -> None:
        actions.append(
            VPPRecvBackward(
                local_stage=local_stage_for_backward(vmb),
                mb_id=real_microbatch(vmb),
            )
        )

    def append_backward(vmb: int) -> None:
        actions.append(
            VPPBackward(
                local_stage=local_stage_for_backward(vmb),
                mb_id=real_microbatch(vmb),
            )
        )

    def append_send_backward(vmb: int) -> None:
        actions.append(
            VPPSendBackward(
                local_stage=local_stage_for_backward(vmb),
                mb_id=real_microbatch(vmb),
            )
        )

    def append_send_forward_recv_backward(fwd_vmb: int, bwd_vmb: int) -> None:
        actions.append(
            VPPSendForwardRecvBackward(
                fwd_local_stage=local_stage_for_forward(fwd_vmb),
                fwd_mb=real_microbatch(fwd_vmb),
                bwd_local_stage=local_stage_for_backward(bwd_vmb),
                bwd_mb=real_microbatch(bwd_vmb),
            )
        )

    def append_send_backward_recv_forward(bwd_vmb: int, fwd_vmb: int) -> None:
        actions.append(
            VPPSendBackwardRecvForward(
                bwd_local_stage=local_stage_for_backward(bwd_vmb),
                bwd_mb=real_microbatch(bwd_vmb),
                fwd_local_stage=local_stage_for_forward(fwd_vmb),
                fwd_mb=real_microbatch(fwd_vmb),
            )
        )

    for vmb in range(num_warmup):
        fwd_local_stage = local_stage_for_forward(vmb)
        if rf_needed(fwd_local_stage):
            append_recv_forward(vmb)
        append_forward(vmb)
        if sf_needed(fwd_local_stage):
            append_send_forward(vmb)

    for k in range(num_steady):
        fwd_vmb = num_warmup + k
        bwd_vmb = k

        fwd_local_stage = local_stage_for_forward(fwd_vmb)
        bwd_local_stage = local_stage_for_backward(bwd_vmb)

        is_first_steady = k == 0
        is_last_steady = k == num_steady - 1

        if is_first_steady and rf_needed(fwd_local_stage):
            append_recv_forward(fwd_vmb)

        append_forward(fwd_vmb)

        sf = sf_needed(fwd_local_stage)
        rb = rb_needed(bwd_local_stage)
        if sf and rb:
            append_send_forward_recv_backward(fwd_vmb, bwd_vmb)
        elif sf:
            append_send_forward(fwd_vmb)
        elif rb:
            append_recv_backward(bwd_vmb)

        append_backward(bwd_vmb)

        sb = sb_needed(bwd_local_stage)
        if is_last_steady:
            if sb:
                append_send_backward(bwd_vmb)
        else:
            next_fwd_vmb = fwd_vmb + 1
            next_fwd_local_stage = local_stage_for_forward(next_fwd_vmb)
            rf_next = rf_needed(next_fwd_local_stage)

            if sb and rf_next:
                append_send_backward_recv_forward(bwd_vmb, next_fwd_vmb)
            elif sb:
                append_send_backward(bwd_vmb)
            elif rf_next:
                append_recv_forward(next_fwd_vmb)

    for j in range(num_cooldown):
        bwd_vmb = num_steady + j
        bwd_local_stage = local_stage_for_backward(bwd_vmb)

        if rb_needed(bwd_local_stage):
            append_recv_backward(bwd_vmb)
        append_backward(bwd_vmb)
        if sb_needed(bwd_local_stage):
            append_send_backward(bwd_vmb)

    return actions
