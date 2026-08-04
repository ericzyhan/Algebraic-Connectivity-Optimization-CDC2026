"""Verification for the deterministic r-1 deflation macro-action.

Covers the env-level contract the macro-action is supposed to provide:

  1. multiplicity is detectable at all between exact anchors (regression)
  2. a degenerate initialization is deflated before the policy observes it
  3. the macro-action can be switched off
  4. the reset batch's reward reaches the episode return
  5. every state the policy acts on has a simple lambda_2
  6. a batch is truncated by the remaining edge budget

Run:  python -m rl_train_new_model.tests.test_deflation_macro
"""

from __future__ import annotations

import numpy as np

from ..config import CurriculumConfig, CurriculumPhaseConfig
from ..curriculum import CurriculumScheduler
from ..env import GraphEnv
from ..graph_math import edge_count, laplacian
from ..spectral import SpectralTracker


def k24():
    """K_{2,4}: lambda_2 has multiplicity 3."""
    A = np.zeros((6, 6), dtype=np.uint8)
    for a in (0, 1):
        for c in (2, 3, 4, 5):
            A[a, c] = A[c, a] = 1
    return A


def dense_mult(adj, tol=1e-8):
    ev = np.linalg.eigvalsh(laplacian(adj).astype(np.float64))
    return int(np.sum(np.abs(ev[1:] - ev[1]) <= tol))


def make_env(deflation=True, seed=0, probe_rel=1e-5, reset_every=50):
    return GraphEnv(
        env_id=0,
        scheduler=CurriculumScheduler(
            CurriculumConfig(phases=[CurriculumPhaseConfig("p", 0, 10**9, 6, 12)])
        ),
        top_k=128,
        dist_cap=4,
        terminal_bonus_coef=0.2,
        seed=seed,
        rl_variant="full",
        spectral_degeneracy_probe_rel=probe_rel,
        spectral_exact_reset_every=reset_every,
        deflation_macro_action=deflation,
    )


# --------------------------------------------------------------------------- #
#  1. Multiplicity must be detectable between exact anchors
# --------------------------------------------------------------------------- #

def random_connected(n, pr, rng):
    while True:
        A = (rng.random((n, n)) < pr).astype(np.uint8)
        A = np.triu(A, 1)
        A = A + A.T
        if np.linalg.eigvalsh(laplacian(A).astype(np.float64))[1] > 1e-8:
            return A


def test_multiplicity_detected_between_anchors():
    """An r=1 -> plateau transition must register as r > 1 without an anchor.

    Regression: the block power iteration resolves the tracked eigenvalues to
    only ~1e-6, but hard multiplicity is decided at 1e-10 relative. Without the
    escalation to an exact spectrum, a genuine plateau reads as r=1, and the
    macro-action would fire only in the rare step right after a periodic anchor.
    """
    rng = np.random.default_rng(0)
    checked = 0
    missed_without_probe = 0
    for n in (8, 9, 10):
        for _ in range(25):
            A = random_connected(n, 0.3, rng)
            for (i, j) in zip(*np.triu_indices(n, k=1)):
                i, j = int(i), int(j)
                if A[i, j]:
                    continue
                A2 = A.copy()
                A2[i, j] = A2[j, i] = 1
                truth = dense_mult(A2)
                if truth == 1:
                    continue
                base = SpectralTracker(A, exact_reset_every=10**9)
                if base.multiplicity != 1:
                    continue  # want specifically the simple -> degenerate step
                base.add_edge(i, j)
                assert base.multiplicity == truth, (
                    f"n={n} edge ({i},{j}): dense r={truth}, tracker r={base.multiplicity}"
                )
                checked += 1

                without = SpectralTracker(
                    A, exact_reset_every=10**9, degeneracy_probe_rel=0.0
                )
                without.add_edge(i, j)
                if without.multiplicity != truth:
                    missed_without_probe += 1
    assert checked > 0, "no simple -> plateau transition found; test is not exercising the path"
    assert missed_without_probe > 0, (
        "disabling the probe did not reproduce the miss -- the regression this "
        "test pins may no longer be reachable"
    )
    return (
        f"{checked} simple->plateau transitions, all detected exactly; "
        f"with the probe disabled {missed_without_probe} of them read as r=1"
    )


# --------------------------------------------------------------------------- #
#  2-4. Reset-time deflation
# --------------------------------------------------------------------------- #

def test_reset_deflates_degenerate_init():
    """K_2,4 has r=3; the policy's first observation must already be simple."""
    env = make_env()
    obs = env.reset_with_target(n=6, rho_target=0.9, initial_adj=k24())
    assert dense_mult(env.adj) == 1, f"still degenerate after reset: r={dense_mult(env.adj)}"
    assert env.current_multiplicity == 1, f"tracker says r={env.current_multiplicity}"
    added = edge_count(env.adj) - 8
    assert added == 2, f"expected r-1 = 2 deflation edges, got {added}"
    assert env._pending_reward != 0.0, "deflation reward was not carried over"
    assert obs.candidate_pairs.shape[0] > 0, "no candidates offered"
    return (
        f"K_2,4 r=3 -> r=1 at reset via {added} edges, "
        f"pending reward {env._pending_reward:+.5f}"
    )


def test_disabled_flag_leaves_plateau():
    """With the macro-action off the env must stay on the plateau (control)."""
    env = make_env(deflation=False)
    env.reset_with_target(n=6, rho_target=0.9, initial_adj=k24())
    assert edge_count(env.adj) == 8, "edges were added despite the flag being off"
    assert env.current_multiplicity == 3, f"expected r=3 preserved, got {env.current_multiplicity}"
    return "flag off: r=3 preserved, no edges added"


def test_pending_reward_lands_in_return():
    env = make_env()
    env.reset_with_target(n=6, rho_target=0.9, initial_adj=k24())
    pending = env._pending_reward
    obs = env._build_observation()
    i, j = (int(v) for v in obs.candidate_pairs[0])
    _, reward, _, _ = env.step((i, j))
    assert env._pending_reward == 0.0, "pending reward not consumed"
    assert abs(env.episode_return - reward) < 1e-12, "episode return lost the carry-over"
    return f"pending {pending:+.5f} folded into first step reward {reward:+.5f}"


# --------------------------------------------------------------------------- #
#  5-6. Episode-level contract
# --------------------------------------------------------------------------- #

def test_full_episodes_stay_simple():
    """Random-policy episodes: every state the policy acts on has r == 1."""
    rng = np.random.RandomState(0)
    checked = 0
    deflations = 0
    for seed in range(12):
        env = make_env(seed=seed)
        obs = env.reset_with_target(n=9, rho_target=0.55)
        m_before = edge_count(env.adj)
        done = edge_count(env.adj) >= env.m_target
        while not done:
            assert env.current_multiplicity == 1, (
                f"policy asked to act at r={env.current_multiplicity}"
            )
            assert dense_mult(env.adj) == 1, "tracker and dense spectrum disagree on r"
            checked += 1
            k = obs.candidate_pairs.shape[0]
            assert k > 0, "no candidates but episode not done"
            i, j = (int(v) for v in obs.candidate_pairs[rng.randint(k)])
            m_pre = edge_count(env.adj)
            obs, reward, done, info = env.step((i, j))
            if edge_count(env.adj) - m_pre > 1:
                deflations += 1
            assert np.isfinite(reward), f"non-finite reward {reward}"
        assert edge_count(env.adj) == env.m_target, (
            f"finished at {edge_count(env.adj)} edges, target {env.m_target}"
        )
        assert env.episode_len == edge_count(env.adj) - m_before, (
            f"episode_len {env.episode_len} != edges added {edge_count(env.adj) - m_before}"
        )
    return (
        f"12 episodes n=9: {checked} policy decisions, all at r=1; "
        f"{deflations} mid-episode deflation batches; budget exact"
    )


def test_budget_never_overshoots():
    """A deflation batch must be truncated by the remaining edge budget.

    K_2,4 is r=3 and wants 2 deflation edges, but is given only 1 of headroom.
    """
    env = make_env()
    env.n = 6
    env.rho_target = 0.0
    env.m_target = 9  # 8 base edges + exactly 1 spare
    env.adj = k24()
    env.episode_return = 0.0
    env.episode_len = 0
    env._initialize_incremental_state()
    env._reset_finalize(env._init_spectral_tracker())

    assert edge_count(env.adj) == 9, f"budget overshot: {edge_count(env.adj)} edges > 9"
    assert env.episode_len == 1, f"took {env.episode_len} edges, only 1 was affordable"
    assert not env._spectral_tracker.batch_active, "truncated batch left open"
    assert not env._spectral_tracker._deferred_exact, "deferred anchor never taken"
    return f"batch truncated at the budget: 8 -> {edge_count(env.adj)} edges, batch closed"


def test_batch_stops_when_basis_is_redrawn():
    """The env must not commit plan edges after the frozen basis is gone.

    A batch aborts when a step does not land the predicted r; if an anchor was
    deferred, closing the batch takes it immediately and re-draws the basis at
    the new r. The plan's remaining edges were chosen for a cascade property
    they only have in the basis that just went away, and their z columns are no
    longer coordinates in it -- committing them anyway is what made the
    accumulator ragged and crashed sigma_r_accum one edge later (n=64 eval).

    A natural mid-plan abort needs a plateau accumulated across an episode
    (every symmetric fixture deflates one clean dimension per edge), so the
    abort is forced here through the tracker's own contract: close the batch
    after its first edge, exactly as a failed post-condition does.
    """
    env = make_env(deflation=False, reset_every=1)  # anchor due every step
    env.reset_with_target(n=6, rho_target=0.9, initial_adj=k24())
    tracker = env._spectral_tracker
    assert tracker.multiplicity == 3, f"fixture should start at r=3, got {tracker.multiplicity}"

    env.deflation_macro_action = True
    real_add = tracker.add_edge
    committed = []
    stale = []

    def aborting_add(i, j, *, cached=None, z=None):
        if z is not None and int(np.asarray(z).reshape(-1).shape[0]) != tracker._z_accum_r:
            stale.append((i, j))
        real_add(i, j, cached=cached, z=z)
        committed.append((i, j))
        if len(committed) == 1 and tracker.batch_active:
            tracker.end_deflation_batch()  # simulate the failed post-condition

    tracker.add_edge = aborting_add
    env._run_deflation_batch()

    assert not stale, f"committed plan edges in a re-drawn basis: {stale}"
    widths = {int(c.shape[0]) for c in tracker._z_accum_cols}
    assert len(widths) <= 1, f"accumulator went ragged: widths {sorted(widths)}"
    assert np.isfinite(tracker.sigma_r_accum()), "sigma_r_accum did not survive the abort"
    assert not tracker.batch_active, "batch left open"
    assert dense_mult(env.adj) == 1, f"still degenerate: r={dense_mult(env.adj)}"
    return (
        f"batch aborted after 1 of 2 planned edges; {len(committed)} edges committed "
        f"across re-plans, none in a dead basis, r=3 -> 1"
    )


def test_deflation_plan_is_independent():
    """The r-1 selected z columns must be linearly independent by construction."""
    from ..candidacy import plan_deflation_batch

    env = make_env(deflation=False)
    env.reset_with_target(n=6, rho_target=0.9, initial_adj=k24())
    tracker = env._spectral_tracker
    pair_i, pair_j, pair_is_nonedge, _ = env._get_pair_arrays()
    plan = plan_deflation_batch(
        tracker=tracker,
        pair_i=pair_i,
        pair_j=pair_j,
        pair_is_nonedge=pair_is_nonedge,
        n=6,
    )
    assert plan is not None, "no plan produced on a known r=3 graph"
    assert plan.edges.shape[0] == plan.r - 1 == 2
    svals = np.linalg.svd(plan.z_columns, compute_uv=False)
    assert svals.min() > 1e-8, f"selected z columns are near-dependent: {svals}"
    assert np.isfinite(plan.cond_R), f"cond(R) = {plan.cond_R}"
    return (
        f"r={plan.r}, {plan.edges.shape[0]} edges, cond(R)={plan.cond_R:.3f}, "
        f"sigma_min(Z_sel)={svals.min():.4f}, pool={plan.pool_size}"
    )


TESTS = [
    test_multiplicity_detected_between_anchors,
    test_reset_deflates_degenerate_init,
    test_disabled_flag_leaves_plateau,
    test_pending_reward_lands_in_return,
    test_full_episodes_stay_simple,
    test_budget_never_overshoots,
    test_batch_stops_when_basis_is_redrawn,
    test_deflation_plan_is_independent,
]


def main() -> int:
    passed = 0
    for fn in TESTS:
        try:
            detail = fn()
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            continue
        passed += 1
        print(f"PASS  {fn.__name__}" + (f"  --  {detail}" if detail else ""))
    print(f"\n{passed}/{len(TESTS)} passed")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
