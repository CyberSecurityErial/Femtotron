"""Reusable assertions for pipeline-parallel schedule tests.

This module is intentionally pure Python: no CUDA, no distributed process
group, and no model construction. It is meant to make new schedule variants
cheap to validate before wiring them into Runner/Stage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from femtotron.parallel.pipeline_parallel.action import (
    Backward,
    Forward,
    PPAction,
    RecvBackward,
    RecvForward,
    SendBackward,
    SendBackwardRecvForward,
    SendForward,
    SendForwardRecvBackward,
)


ActionEvent = tuple[str, int]


def action_events(action: PPAction) -> list[ActionEvent]:
    """Return logical events contributed by one action.

    Combined communication actions count as two logical events. This keeps
    invariant checks independent from whether a schedule used separate or
    combined P2P operations.
    """
    if isinstance(action, Forward):
        return [("f", action.mb_id)]
    if isinstance(action, Backward):
        return [("b", action.mb_id)]
    if isinstance(action, RecvForward):
        return [("rf", action.mb_id)]
    if isinstance(action, SendForward):
        return [("sf", action.mb_id)]
    if isinstance(action, RecvBackward):
        return [("rb", action.mb_id)]
    if isinstance(action, SendBackward):
        return [("sb", action.mb_id)]
    if isinstance(action, SendForwardRecvBackward):
        return [("sf", action.fwd_mb), ("rb", action.bwd_mb)]
    if isinstance(action, SendBackwardRecvForward):
        return [("sb", action.bwd_mb), ("rf", action.fwd_mb)]

    # Future zero-bubble actions can be introduced without immediately
    # rewriting the shared harness. These names mirror common dgrad/wgrad
    # terminology while still requiring the action to expose mb_id.
    action_name = type(action).__name__
    if action_name in {"BackwardInput", "BackwardInputGrad", "BackwardDgrad"}:
        return [("bi", action.mb_id)]
    if action_name in {"BackwardWeight", "BackwardWeightGrad", "BackwardWgrad"}:
        return [("bw", action.mb_id)]

    raise TypeError(f"unknown action type: {type(action).__name__}")


def count_events(actions: Iterable[PPAction]) -> Counter[ActionEvent]:
    counts: Counter[ActionEvent] = Counter()
    for action in actions:
        counts.update(action_events(action))
    return counts


def positions_by_event(actions: Iterable[PPAction]) -> dict[str, dict[int, int]]:
    positions: dict[str, dict[int, int]] = defaultdict(dict)
    for idx, action in enumerate(actions):
        for kind, mb_id in action_events(action):
            positions[kind][mb_id] = idx
    return positions


def assert_vanilla_pp_invariants(
    actions: list[PPAction],
    num_microbatches: int,
    *,
    is_first: bool,
    is_last: bool,
    tag: str,
) -> None:
    """Validate GPipe/1F1B style schedules with one full backward per mb."""
    counts = count_events(actions)
    expected = {
        "f": 1,
        "b": 1,
        "rf": 0 if is_first else 1,
        "sf": 0 if is_last else 1,
        "rb": 0 if is_last else 1,
        "sb": 0 if is_first else 1,
    }

    for mb_id in range(num_microbatches):
        for kind, want in expected.items():
            got = counts[(kind, mb_id)]
            assert got == want, (
                f"{tag} mb={mb_id}: {kind}={got}, expected {want}; "
                f"counts={_counts_for_mb(counts, mb_id)}"
            )

    assert_vanilla_pp_ordering(actions, num_microbatches, tag=tag)


def assert_vanilla_pp_ordering(
    actions: list[PPAction],
    num_microbatches: int,
    *,
    tag: str,
) -> None:
    """Validate per-microbatch ordering constraints."""
    pos = positions_by_event(actions)
    for mb_id in range(num_microbatches):
        assert mb_id in pos["f"], f"{tag} mb={mb_id}: missing forward"
        assert mb_id in pos["b"], f"{tag} mb={mb_id}: missing backward"
        assert pos["f"][mb_id] < pos["b"][mb_id], (
            f"{tag} mb={mb_id}: F@{pos['f'][mb_id]} >= B@{pos['b'][mb_id]}"
        )

        for before, after in (("rf", "f"), ("f", "sf"), ("rb", "b"), ("b", "sb")):
            if mb_id in pos[before] and mb_id in pos[after]:
                assert pos[before][mb_id] < pos[after][mb_id], (
                    f"{tag} mb={mb_id}: {before}@{pos[before][mb_id]} "
                    f">= {after}@{pos[after][mb_id]}"
                )


def assert_zero_bubble_candidate_invariants(
    actions: list[PPAction],
    num_microbatches: int,
    *,
    is_first: bool,
    is_last: bool,
    tag: str,
) -> None:
    """Validate basic coverage for a vanilla or split-backward PP schedule.

    A future zero-bubble schedule is expected to replace full Backward with
    separate input-gradient and weight-gradient actions. This assertion accepts
    either shape:
        - vanilla: B once per microbatch
        - split: BI and BW once per microbatch
    """
    counts = count_events(actions)
    pos = positions_by_event(actions)

    for mb_id in range(num_microbatches):
        assert counts[("f", mb_id)] == 1, f"{tag} mb={mb_id}: missing forward"

        has_full_backward = counts[("b", mb_id)] == 1
        has_split_backward = counts[("bi", mb_id)] == 1 and counts[("bw", mb_id)] == 1
        assert has_full_backward or has_split_backward, (
            f"{tag} mb={mb_id}: expected B or BI+BW; "
            f"counts={_counts_for_mb(counts, mb_id)}"
        )

        expected_comm = {
            "rf": 0 if is_first else 1,
            "sf": 0 if is_last else 1,
            "rb": 0 if is_last else 1,
            "sb": 0 if is_first else 1,
        }
        for kind, want in expected_comm.items():
            got = counts[(kind, mb_id)]
            assert got == want, (
                f"{tag} mb={mb_id}: {kind}={got}, expected {want}; "
                f"counts={_counts_for_mb(counts, mb_id)}"
            )

        if has_full_backward:
            assert pos["f"][mb_id] < pos["b"][mb_id], (
                f"{tag} mb={mb_id}: F must precede B"
            )
        else:
            assert pos["f"][mb_id] < pos["bi"][mb_id], (
                f"{tag} mb={mb_id}: F must precede BI"
            )
            assert pos["f"][mb_id] < pos["bw"][mb_id], (
                f"{tag} mb={mb_id}: F must precede BW"
            )
            if mb_id in pos["rb"]:
                assert pos["rb"][mb_id] < pos["bi"][mb_id], (
                    f"{tag} mb={mb_id}: RB must precede BI"
                )
                assert pos["rb"][mb_id] < pos["bw"][mb_id], (
                    f"{tag} mb={mb_id}: RB must precede BW"
                )
            if mb_id in pos["sb"]:
                assert pos["bi"][mb_id] < pos["sb"][mb_id], (
                    f"{tag} mb={mb_id}: BI must precede SB"
                )



def summarize_actions(actions: Iterable[PPAction]) -> dict[str, int]:
    summary: Counter[str] = Counter(type(action).__name__ for action in actions)
    return dict(sorted(summary.items()))


def format_action_trace(actions: Iterable[PPAction], *, line_width: int = 100) -> str:
    """Format a compact schedule trace for debugging."""
    rendered = [repr(action) for action in actions]
    lines: list[str] = []
    current = ""
    for token in rendered:
        piece = token if not current else ", " + token
        if current and len(current) + len(piece) > line_width:
            lines.append(current)
            current = token
        else:
            current += piece
    if current:
        lines.append(current)
    return "\n".join(lines)


def _counts_for_mb(counts: Counter[ActionEvent], mb_id: int) -> dict[str, int]:
    return {
        kind: counts[(kind, mb_id)]
        for kind in ("rf", "f", "sf", "rb", "b", "bi", "bw", "sb")
    }
