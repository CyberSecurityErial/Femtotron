"""Backward action split pass for existing microbatch-based schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from femtotron.parallel.pipeline_parallel.action import (
    Backward,
    BackwardInputGrad,
    PPAction,
)


@dataclass(frozen=True, slots=True)
class PendingWGrad:
    """Non-executable marker produced by the backward split pass."""

    mb_id: int

    def __post_init__(self) -> None:
        if self.mb_id < 0:
            raise ValueError(f"mb_id must be non-negative, got {self.mb_id}")

    def __repr__(self) -> str:
        return f"PW({self.mb_id})"


BackwardSplitItem: TypeAlias = PPAction | PendingWGrad


def split_backward_dw(actions: list[PPAction]) -> list[BackwardSplitItem]:
    """Replace each Backward action with D plus a pending W marker.

    This pass only expresses the B -> D/W split. It does not decide where
    executable BackwardWeightGrad actions should be placed.
    """
    out: list[BackwardSplitItem] = []

    for action in actions:
        if isinstance(action, Backward):
            out.append(BackwardInputGrad(mb_id=action.mb_id))
            out.append(PendingWGrad(mb_id=action.mb_id))
        else:
            out.append(action)

    return out
