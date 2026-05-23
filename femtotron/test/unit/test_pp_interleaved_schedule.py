"""Unit tests for interleaved_one_f_one_b_schedule.

Pure-data tests, no GPU / no distributed / no model. Verifies:
    - encode/decode roundtrip
    - Per-(real_mb, chunk) invariants: F=1, B=1, comm counts based on boundary
    - Per-(real_mb, chunk) ordering: RF<F<SF, RB<B<SB, F<B
    - Total compute count: N*V F's and N*V B's per device
    - Warmup formula correctness (Megatron's (P-r-1)*2 + (V-1)*P)
    - Boundary chunks (global first/last) skip the right comm actions
    - Combined ops (SFRB/SBRF) never have a no-op side
    - Edge cases: P=1, V=1, N=P
"""

from collections import defaultdict

from femtotron.parallel.pipeline_parallel.interleaved_schedule import (
    interleaved_one_f_one_b_schedule,
    encode_mb_id,
    decode_mb_id,
)
from femtotron.parallel.pipeline_parallel.action import (
    PPAction,
    Forward, Backward,
    RecvForward, SendForward,
    RecvBackward, SendBackward,
    SendForwardRecvBackward, SendBackwardRecvForward,
)


# ════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════

def log(msg):
    print(f"  {msg}")


def _eff(chunk, mb, N):
    return chunk * N + mb


def _counts_for(actions, target_eff):
    """Count (rf, f, sf, rb, b, sb) for a specific eff_mb_id (combined ops contribute)."""
    rf = f = sf = rb = b = sb = 0
    for a in actions:
        if   isinstance(a, Forward)        and a.mb_id == target_eff: f  += 1
        elif isinstance(a, Backward)       and a.mb_id == target_eff: b  += 1
        elif isinstance(a, RecvForward)    and a.mb_id == target_eff: rf += 1
        elif isinstance(a, SendForward)    and a.mb_id == target_eff: sf += 1
        elif isinstance(a, RecvBackward)   and a.mb_id == target_eff: rb += 1
        elif isinstance(a, SendBackward)   and a.mb_id == target_eff: sb += 1
        elif isinstance(a, SendForwardRecvBackward):
            if a.fwd_mb == target_eff: sf += 1
            if a.bwd_mb == target_eff: rb += 1
        elif isinstance(a, SendBackwardRecvForward):
            if a.bwd_mb == target_eff: sb += 1
            if a.fwd_mb == target_eff: rf += 1
    return rf, f, sf, rb, b, sb


def _check_per_chunk_mb_invariants(actions, N, V, P, r, tag):
    """For each (real_mb, chunk): F=1, B=1, comm counts based on boundary status."""
    is_first_dev = (r == 0)
    is_last_dev = (r == P - 1)

    for mb in range(N):
        for chunk in range(V):
            eff = _eff(chunk, mb, N)
            rf, f, sf, rb, b, sb = _counts_for(actions, eff)

            assert f == 1, f"{tag} mb={mb} chunk={chunk}: F={f} (expected 1)"
            assert b == 1, f"{tag} mb={mb} chunk={chunk}: B={b} (expected 1)"

            expected_sf = 0 if (chunk == V - 1 and is_last_dev) else 1
            assert sf == expected_sf, (
                f"{tag} mb={mb} chunk={chunk}: SF={sf} (expected {expected_sf})"
            )
            expected_rb = 0 if (chunk == V - 1 and is_last_dev) else 1
            assert rb == expected_rb, (
                f"{tag} mb={mb} chunk={chunk}: RB={rb} (expected {expected_rb})"
            )
            expected_sb = 0 if (chunk == 0 and is_first_dev) else 1
            assert sb == expected_sb, (
                f"{tag} mb={mb} chunk={chunk}: SB={sb} (expected {expected_sb})"
            )
            expected_rf = 0 if (chunk == 0 and is_first_dev) else 1
            assert rf == expected_rf, (
                f"{tag} mb={mb} chunk={chunk}: RF={rf} (expected {expected_rf})"
            )


def _check_per_chunk_ordering(actions, N, V, tag):
    """Per (real_mb, chunk): RF<F<SF, RB<B<SB, F<B."""
    pos = defaultdict(dict)
    for i, a in enumerate(actions):
        if   isinstance(a, Forward):       pos['f'][a.mb_id]  = i
        elif isinstance(a, Backward):      pos['b'][a.mb_id]  = i
        elif isinstance(a, RecvForward):   pos['rf'][a.mb_id] = i
        elif isinstance(a, SendForward):   pos['sf'][a.mb_id] = i
        elif isinstance(a, RecvBackward):  pos['rb'][a.mb_id] = i
        elif isinstance(a, SendBackward):  pos['sb'][a.mb_id] = i
        elif isinstance(a, SendForwardRecvBackward):
            pos['sf'][a.fwd_mb] = i
            pos['rb'][a.bwd_mb] = i
        elif isinstance(a, SendBackwardRecvForward):
            pos['sb'][a.bwd_mb] = i
            pos['rf'][a.fwd_mb] = i

    for mb in range(N):
        for chunk in range(V):
            eff = _eff(chunk, mb, N)
            assert eff in pos['f'] and eff in pos['b'], (
                f"{tag} (mb={mb} chunk={chunk}): missing F or B"
            )
            assert pos['f'][eff] < pos['b'][eff], (
                f"{tag} (mb={mb} chunk={chunk}): "
                f"F@{pos['f'][eff]} >= B@{pos['b'][eff]}"
            )
            for before, after in [('rf', 'f'), ('f', 'sf'), ('rb', 'b'), ('b', 'sb')]:
                if eff in pos[before] and eff in pos[after]:
                    assert pos[before][eff] < pos[after][eff], (
                        f"{tag} (mb={mb} chunk={chunk}): "
                        f"{before}@{pos[before][eff]} >= {after}@{pos[after][eff]}"
                    )


def _check_total_compute(actions, N, V, tag):
    """Total F = total B = N * V."""
    f_count = sum(1 for a in actions if isinstance(a, Forward))
    b_count = sum(1 for a in actions if isinstance(a, Backward))
    assert f_count == N * V, f"{tag}: total F={f_count}, expected {N*V}"
    assert b_count == N * V, f"{tag}: total B={b_count}, expected {N*V}"


def _full_validation(actions, N, V, P, r, tag):
    _check_total_compute(actions, N, V, tag)
    _check_per_chunk_mb_invariants(actions, N, V, P, r, tag)
    _check_per_chunk_ordering(actions, N, V, tag)


# ════════════════════════════════════════════════════════════════
# tests
# ════════════════════════════════════════════════════════════════

def test_encode_decode_roundtrip():
    """encode_mb_id / decode_mb_id are exact inverses."""
    for N in [1, 4, 8, 16]:
        for chunk in range(5):
            for mb in range(N):
                eff = encode_mb_id(chunk, mb, N)
                c, m = decode_mb_id(eff, N)
                assert (c, m) == (chunk, mb), \
                    f"roundtrip failed: ({chunk},{mb},N={N}) → eff={eff} → ({c},{m})"
    log("✓ encode/decode roundtrip exact for various (chunk, mb, N)")


def test_v2_p4_n8():
    """Canonical case: P=4, V=2, N=8."""
    P, V, N = 4, 2, 8
    for r in range(P):
        actions = interleaved_one_f_one_b_schedule(N, P, r, V)
        _full_validation(actions, N, V, P, r, f"P{P}V{V}N{N}R{r}")
    log(f"✓ P=4 V=2 N=8: invariants pass for all 4 ranks")


def test_v2_p2_n4():
    """Minimum interesting case: P=2, V=2, N=4."""
    P, V, N = 2, 2, 4
    for r in range(P):
        actions = interleaved_one_f_one_b_schedule(N, P, r, V)
        _full_validation(actions, N, V, P, r, f"P{P}V{V}N{N}R{r}")
    log(f"✓ P=2 V=2 N=4: invariants hold for both ranks")


def test_v3_p2_n4():
    """Larger V: P=2, V=3, N=4。"""
    P, V, N = 2, 3, 4
    for r in range(P):
        actions = interleaved_one_f_one_b_schedule(N, P, r, V)
        _full_validation(actions, N, V, P, r, f"P{P}V{V}N{N}R{r}")
    log(f"✓ P=2 V=3 N=4: invariants hold (3 chunks per device)")


def test_v4_p4_n8():
    """Many chunks: P=4, V=4, N=8 (V=4 stress test)."""
    P, V, N = 4, 4, 8
    for r in range(P):
        actions = interleaved_one_f_one_b_schedule(N, P, r, V)
        _full_validation(actions, N, V, P, r, f"P{P}V{V}N{N}R{r}")
    log(f"✓ P=4 V=4 N=8: invariants hold (V=4)")


def test_pp1_degenerate():
    """P=1 单设备:所有 chunks 都在 rank 0;chunk 0 = first, chunk V-1 = last。"""
    P, V, N = 1, 3, 4
    actions = interleaved_one_f_one_b_schedule(N, P, 0, V)
    _full_validation(actions, N, V, P, 0, "P1V3N4")
    log(f"✓ P=1 V=3 N=4: degenerate single-device case works")


def test_warmup_formula():
    """Verify warmup count matches Megatron formula: (P-r-1)*2 + (V-1)*P, capped at N*V.

    注意:数"first B 之前的 F 个数"会**多出 1**,因为 steady iter 0 的 F 也在第一个
    B 之前。所以 f_before_first_b == num_warmup + 1(假设 steady 非空)。
    """
    cases = [
        # (P, V, N, r, expected_warmup) — expected_warmup 是纯 warmup F 数,不含 steady 第一个 F
        (4, 2, 8, 0, 10),  # (3)*2 + 4 = 10
        (4, 2, 8, 1, 8),   # (2)*2 + 4 = 8
        (4, 2, 8, 2, 6),
        (4, 2, 8, 3, 4),   # 0 + 4 = 4
        (2, 2, 4, 0, 4),   # 1*2 + 2 = 4
        (2, 2, 4, 1, 2),   # 0 + 2 = 2
        (4, 4, 8, 0, 18),  # 6 + 12 = 18
        (4, 4, 8, 3, 12),  # 0 + 12 = 12
    ]
    for P, V, N, r, expected_warmup in cases:
        actions = interleaved_one_f_one_b_schedule(N, P, r, V)
        # 数所有 Backward 之前的 Forward(包括 warmup + steady 第一个 F)
        f_before_first_b = 0
        for a in actions:
            if isinstance(a, Backward):
                break
            if isinstance(a, Forward):
                f_before_first_b += 1
        # warmup 段贡献 expected_warmup 个 F;steady iter 0 又贡献 1 个 F 在 B 之前
        expected_total = expected_warmup + 1
        assert f_before_first_b == expected_total, (
            f"P={P} V={V} N={N} r={r}: F before first B = {f_before_first_b}, "
            f"expected {expected_total} (warmup={expected_warmup} + 1 steady-iter-0 F)"
        )
    log(f"✓ warmup formula matches Megatron's for {len(cases)} configurations")


def test_n_equals_p_with_v():
    """N=P edge: warmup may saturate at total_vmb=N*V."""
    P, V, N = 4, 2, 4
    for r in range(P):
        actions = interleaved_one_f_one_b_schedule(N, P, r, V)
        _full_validation(actions, N, V, P, r, f"P{P}V{V}N={P}R{r}")
    log(f"✓ N=P edge: P=4 V=2 N=4 all ranks valid")


def test_invalid_args():
    """ValueError for bad inputs."""
    cases = [
        dict(num_microbatches=0, pp_size=2, pp_rank=0, virtual_stages=2),
        dict(num_microbatches=4, pp_size=0, pp_rank=0, virtual_stages=2),
        dict(num_microbatches=4, pp_size=2, pp_rank=-1, virtual_stages=2),
        dict(num_microbatches=4, pp_size=2, pp_rank=2, virtual_stages=2),  # rank>=size
        dict(num_microbatches=4, pp_size=2, pp_rank=0, virtual_stages=0),
        # N not divisible by P (interleaved-specific)
        dict(num_microbatches=5, pp_size=2, pp_rank=0, virtual_stages=2),
        dict(num_microbatches=6, pp_size=4, pp_rank=0, virtual_stages=2),
    ]
    for kwargs in cases:
        try:
            interleaved_one_f_one_b_schedule(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {kwargs}")
    log(f"✓ {len(cases)} invalid arg combinations all rejected with ValueError")


def test_v1_produces_valid_schedule():
    """V=1 不等价于 vanilla 1F1B(warmup 是 2x),但仍然产生 valid schedule。

    用户被警告 V=1 用 schedule.one_f_one_b_schedule,但本函数 V=1 仍然要正确。
    """
    P, V, N = 2, 1, 4
    for r in range(P):
        actions = interleaved_one_f_one_b_schedule(N, P, r, V)
        _full_validation(actions, N, V, P, r, f"P{P}V{V}N{N}R{r}")
    log(f"✓ V=1 edge: produces valid (though non-vanilla-equivalent) schedule")


def test_combined_op_never_no_op_side():
    """SFRB / SBRF 都不应有"一边是 no-op"的情况(应被 schedule 拆成单向)。"""
    # 试一组覆盖广的 (P, V, N) 组合
    configs = [
        (2, 2, 4), (2, 3, 4),
        (4, 2, 8), (4, 2, 4), (4, 4, 8),
    ]
    for P, V, N in configs:
        for r in range(P):
            actions = interleaved_one_f_one_b_schedule(N, P, r, V)
            is_first_dev = (r == 0)
            is_last_dev = (r == P - 1)

            for a in actions:
                if isinstance(a, SendForwardRecvBackward):
                    fwd_c, _ = decode_mb_id(a.fwd_mb, N)
                    bwd_c, _ = decode_mb_id(a.bwd_mb, N)
                    assert not (fwd_c == V - 1 and is_last_dev), (
                        f"P={P} V={V} N={N} r={r}: SFRB has SF on global last "
                        f"(fwd_chunk={fwd_c}); should have been split."
                    )
                    assert not (bwd_c == V - 1 and is_last_dev), (
                        f"P={P} V={V} N={N} r={r}: SFRB has RB on global last "
                        f"(bwd_chunk={bwd_c}); should have been split."
                    )
                elif isinstance(a, SendBackwardRecvForward):
                    bwd_c, _ = decode_mb_id(a.bwd_mb, N)
                    fwd_c, _ = decode_mb_id(a.fwd_mb, N)
                    assert not (bwd_c == 0 and is_first_dev), (
                        f"P={P} V={V} N={N} r={r}: SBRF has SB on global first "
                        f"(bwd_chunk={bwd_c}); should have been split."
                    )
                    assert not (fwd_c == 0 and is_first_dev), (
                        f"P={P} V={V} N={N} r={r}: SBRF has RF on global first "
                        f"(fwd_chunk={fwd_c}); should have been split."
                    )
    log(f"✓ combined ops have no no-op side in any of {len(configs)} configurations")


def test_p2_v2_n4_specific_trace():
    """P=2 V=2 N=4 rank 0 的具体调度结构 sanity check。

    rank 0, warmup = (2-0-1)*2 + 1*2 = 4 个 F:
        vid=0: F(mb=0,c=0)  → c=0 是 global first(rank 0),no RF,有 SF
        vid=1: F(mb=1,c=0)  → 同上
        vid=2: F(mb=0,c=1)  → c=1 不是 global first,有 RF,有 SF
        vid=3: F(mb=1,c=1)  → 同上
    所以 warmup 段总 actions = 2*(F,SF) + 2*(RF,F,SF) = 4 + 6 = 10 actions。
    """
    P, V, N = 2, 2, 4
    actions = interleaved_one_f_one_b_schedule(N, P, 0, V)

    warmup_phase = actions[:10]  # 前 10 个 action 是 warmup 段

    f_in_warmup = sum(1 for a in warmup_phase if isinstance(a, Forward))
    sf_in_warmup = sum(1 for a in warmup_phase if isinstance(a, SendForward))
    rf_in_warmup = sum(1 for a in warmup_phase if isinstance(a, RecvForward))

    assert f_in_warmup == 4, f"warmup F count: {f_in_warmup} (expected 4)"
    assert sf_in_warmup == 4, f"warmup SF count: {sf_in_warmup} (expected 4)"
    assert rf_in_warmup == 2, f"warmup RF count: {rf_in_warmup} (expected 2)"

    # 第 11 个 action(index 10)应该是 steady iter 0 的 Forward
    # (rank 0 chunk 0 不需 first-steady RF,所以直接 F)
    assert isinstance(actions[10], Forward), \
        f"action[10] should be steady iter 0's F, got {type(actions[10]).__name__}"
    log(f"✓ P=2 V=2 N=4 r=0: warmup 段 (10 actions: 4 F, 4 SF, 2 RF) 结构正确")


def main():
    tests = [
        test_encode_decode_roundtrip,
        test_v2_p4_n8,
        test_v2_p2_n4,
        test_v3_p2_n4,
        test_v4_p4_n8,
        test_pp1_degenerate,
        test_warmup_formula,
        test_n_equals_p_with_v,
        test_invalid_args,
        test_v1_produces_valid_schedule,
        test_combined_op_never_no_op_side,
        test_p2_v2_n4_specific_trace,
    ]
    print(f"\nRunning {len(tests)} tests for interleaved_one_f_one_b_schedule\n")
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    print(f"\n✅ All {len(tests)} tests passed\n")


if __name__ == "__main__":
    main()