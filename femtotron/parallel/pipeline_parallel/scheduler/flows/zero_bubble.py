"""ZB-lite flow boundary composed from reusable microbatch passes."""

from __future__ import annotations

from femtotron.parallel.pipeline_parallel.action import PPAction
from femtotron.parallel.pipeline_parallel.scheduler.flows.one_f_one_b import (
    build_one_f_one_b_flow,
)
from femtotron.parallel.pipeline_parallel.scheduler.passes.backward_split import (
    split_backward_dw,
)
from femtotron.parallel.pipeline_parallel.scheduler.passes.wgrad_placement import (
    place_wgrad_after_dgrad_send,
)


def build_zero_bubble_flow(
    *,
    num_microbatches: int,
    pp_size: int,
    pp_rank: int,
) -> list[PPAction]:
    """Return the current ZB-lite flow using explicit scheduler passes."""
    base = build_one_f_one_b_flow(
        num_microbatches=num_microbatches,
        pp_size=pp_size,
        pp_rank=pp_rank,
    )
    split = split_backward_dw(base)
    return place_wgrad_after_dgrad_send(
        split,
        is_first=(pp_rank == 0),
    )
