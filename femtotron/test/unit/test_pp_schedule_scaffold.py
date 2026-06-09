"""Scaffold tests for adding new PP schedule variants.

Run:
    PYTHONPATH=. python femtotron/test/unit/test_pp_schedule_scaffold.py

The zero-bubble section is intentionally optional. Once a
`zero_bubble_schedule` function is added to schedule.py, this file will begin
validating its basic action coverage and ordering without needing a new test
entry point.
"""

from __future__ import annotations

from femtotron.parallel.pipeline_parallel import schedule as schedule_mod
from femtotron.parallel.pipeline_parallel.schedule import (
    gpipe_schedule,
    one_f_one_b_schedule,
)
from femtotron.test.unit.pp_schedule_test_utils import (
    assert_vanilla_pp_invariants,
    assert_zero_bubble_candidate_invariants,
    format_action_trace,
    summarize_actions,
)


def log(msg: str) -> None:
    print(f"  {msg}")


def test_existing_schedules_with_shared_harness() -> None:
    cases = []
    for num_microbatches in (1, 2, 4):
        cases.extend([
            (
                f"gpipe N={num_microbatches} first",
                gpipe_schedule(num_microbatches, is_first=True, is_last=False),
                num_microbatches,
                True,
                False,
            ),
            (
                f"gpipe N={num_microbatches} mid",
                gpipe_schedule(num_microbatches, is_first=False, is_last=False),
                num_microbatches,
                False,
                False,
            ),
        ])

    for pp_size in (1, 2, 3, 4):
        for pp_rank in range(pp_size):
            for num_microbatches in (max(1, pp_size - 1), pp_size, pp_size + 2):
                cases.append((
                    f"1f1b P={pp_size} R={pp_rank} N={num_microbatches}",
                    one_f_one_b_schedule(num_microbatches, pp_size, pp_rank),
                    num_microbatches,
                    pp_rank == 0,
                    pp_rank == pp_size - 1,
                ))

    for tag, actions, num_microbatches, is_first, is_last in cases:
        assert_vanilla_pp_invariants(
            actions,
            num_microbatches,
            is_first=is_first,
            is_last=is_last,
            tag=tag,
        )

    log(f"validated {len(cases)} existing schedule cases with shared harness")


def test_zero_bubble_schedule_placeholder() -> None:
    zero_bubble_schedule = getattr(schedule_mod, "zero_bubble_schedule", None)
    if zero_bubble_schedule is None:
        log("zero_bubble_schedule not implemented yet; scaffold is ready")
        return

    for pp_size in (2, 3, 4):
        for pp_rank in range(pp_size):
            num_microbatches = max(pp_size, 4)
            actions = zero_bubble_schedule(num_microbatches, pp_size, pp_rank)
            assert_zero_bubble_candidate_invariants(
                actions,
                num_microbatches,
                is_first=pp_rank == 0,
                is_last=pp_rank == pp_size - 1,
                tag=f"zero-bubble P={pp_size} R={pp_rank} N={num_microbatches}",
            )

    log("zero_bubble_schedule basic invariants passed")


def test_trace_helpers() -> None:
    actions = one_f_one_b_schedule(4, 2, 0)
    trace = format_action_trace(actions, line_width=40)
    summary = summarize_actions(actions)
    assert "F(0)" in trace and "SFRB(1,0)" in trace
    assert summary["Forward"] == 4
    log(f"trace helper summary: {summary}")


def main() -> None:
    tests = [
        test_existing_schedules_with_shared_harness,
        test_zero_bubble_schedule_placeholder,
        test_trace_helpers,
    ]
    print(f"\nRunning {len(tests)} PP schedule scaffold tests\n")
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"\nAll {len(tests)} PP schedule scaffold tests passed\n")


if __name__ == "__main__":
    main()
