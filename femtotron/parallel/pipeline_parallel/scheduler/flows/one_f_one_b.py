"""1F1B flow boundary preserving legacy microbatch semantics."""

from __future__ import annotations

from femtotron.parallel.pipeline_parallel.action import PPAction
from femtotron.parallel.pipeline_parallel.schedule import one_f_one_b_schedule


def build_one_f_one_b_flow(
    *,
    num_microbatches: int,
    pp_size: int,
    pp_rank: int,
) -> list[PPAction]:
    """Return the microbatch-preserving 1F1B action flow.

    This is the scheduler/flows entry point for legacy 1F1B semantics. It
    intentionally preserves mb_id / fwd_mb / bwd_mb exactly.
    """
    return one_f_one_b_schedule(
        num_microbatches=num_microbatches,
        pp_size=pp_size,
        pp_rank=pp_rank,
    )
