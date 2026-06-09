"""Scheduler v2 intermediate representation.

This module intentionally does not replace the legacy PPAction classes. It
adds explicit task identity for future schedulers that need logical stage,
local stage, phase, and pipeline direction without overloading microbatch ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FlowDirection(str, Enum):
    """Logical flow direction for bidirectional pipeline schedules."""

    FORWARD = "forward"
    REVERSE = "reverse"


@dataclass(frozen=True, slots=True)
class TaskKey:
    """Stable identity for a scheduled unit of pipeline work."""

    mb: int
    logical_stage: int
    local_stage: int = 0
    phase: int = 0
    direction: FlowDirection = FlowDirection.FORWARD

    def short(self) -> str:
        """Return a compact, stable debug representation."""
        return (
            f"mb={self.mb},ls={self.logical_stage},local={self.local_stage},"
            f"phase={self.phase},dir={self.direction.value}"
        )

    def __repr__(self) -> str:
        return f"TaskKey({self.short()})"


@dataclass(frozen=True, slots=True)
class PPActionV2:
    """Base class for scheduler v2 actions."""


@dataclass(frozen=True, slots=True)
class ForwardV2(PPActionV2):
    key: TaskKey

    def __repr__(self) -> str:
        return f"F2({self.key.short()})"


@dataclass(frozen=True, slots=True)
class BackwardV2(PPActionV2):
    key: TaskKey

    def __repr__(self) -> str:
        return f"B2({self.key.short()})"


@dataclass(frozen=True, slots=True)
class BackwardInputGradV2(PPActionV2):
    key: TaskKey

    def __repr__(self) -> str:
        return f"D2({self.key.short()})"


@dataclass(frozen=True, slots=True)
class BackwardWeightGradV2(PPActionV2):
    key: TaskKey

    def __repr__(self) -> str:
        return f"W2({self.key.short()})"


@dataclass(frozen=True, slots=True)
class RecvForwardV2(PPActionV2):
    key: TaskKey

    def __repr__(self) -> str:
        return f"RF2({self.key.short()})"


@dataclass(frozen=True, slots=True)
class SendForwardV2(PPActionV2):
    key: TaskKey

    def __repr__(self) -> str:
        return f"SF2({self.key.short()})"


@dataclass(frozen=True, slots=True)
class RecvBackwardV2(PPActionV2):
    key: TaskKey

    def __repr__(self) -> str:
        return f"RB2({self.key.short()})"


@dataclass(frozen=True, slots=True)
class SendBackwardV2(PPActionV2):
    key: TaskKey

    def __repr__(self) -> str:
        return f"SB2({self.key.short()})"


@dataclass(frozen=True, slots=True)
class SendForwardRecvBackwardV2(PPActionV2):
    fwd_key: TaskKey
    bwd_key: TaskKey

    def __repr__(self) -> str:
        return f"SFRB2(fwd={self.fwd_key.short()},bwd={self.bwd_key.short()})"


@dataclass(frozen=True, slots=True)
class SendBackwardRecvForwardV2(PPActionV2):
    bwd_key: TaskKey
    fwd_key: TaskKey

    def __repr__(self) -> str:
        return f"SBRF2(bwd={self.bwd_key.short()},fwd={self.fwd_key.short()})"


@dataclass(frozen=True, slots=True)
class OverlapForwardBackwardV2(PPActionV2):
    fwd_key: TaskKey
    bwd_key: TaskKey

    def __repr__(self) -> str:
        return f"OFB2(fwd={self.fwd_key.short()},bwd={self.bwd_key.short()})"


@dataclass(frozen=True, slots=True)
class WaitCommV2(PPActionV2):
    """Wait for outstanding asynchronous pipeline communication."""

    def __repr__(self) -> str:
        return "WaitComm2()"
