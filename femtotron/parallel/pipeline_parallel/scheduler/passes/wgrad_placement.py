"""Weight-gradient placement pass for the current ZB-lite semantics."""

from __future__ import annotations

from femtotron.parallel.pipeline_parallel.action import (
    BackwardWeightGrad,
    PPAction,
    SendBackward,
    SendBackwardRecvForward,
)
from femtotron.parallel.pipeline_parallel.scheduler.passes.backward_split import (
    BackwardSplitItem,
    PendingWGrad,
)


def place_wgrad_after_dgrad_send(
    items: list[BackwardSplitItem],
    *,
    is_first: bool,
) -> list[PPAction]:
    """Convert pending W markers into executable W actions.

    First stages have no upstream dgrad send, so W can run immediately after D.
    Other stages delay W until the corresponding dgrad has been sent upstream
    by SendBackward or SendBackwardRecvForward.
    """
    actions: list[PPAction] = []
    pending_w: set[int] = set()

    for item in items:
        if isinstance(item, PendingWGrad):
            if is_first:
                actions.append(BackwardWeightGrad(mb_id=item.mb_id))
            else:
                pending_w.add(item.mb_id)
            continue

        actions.append(item)

        if isinstance(item, SendBackward):
            if item.mb_id in pending_w:
                actions.append(BackwardWeightGrad(mb_id=item.mb_id))
                pending_w.remove(item.mb_id)
        elif isinstance(item, SendBackwardRecvForward):
            if item.bwd_mb in pending_w:
                actions.append(BackwardWeightGrad(mb_id=item.bwd_mb))
                pending_w.remove(item.bwd_mb)

    if pending_w:
        raise RuntimeError(f"unmatched pending W actions: {sorted(pending_w)}")

    return actions
