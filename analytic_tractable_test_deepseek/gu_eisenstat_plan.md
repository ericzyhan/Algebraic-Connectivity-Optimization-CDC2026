# Gu–Eisenstat Incremental Spectral State — Implementation Plan

Status: **planned, not implemented.** Companion to `algorithm.md`,
`algorithm_implementation.md`, and `algorithm_implement_update.md`. Paths in this
document are relative to `analytic_tractable_test_deepseek/`.

---

## Context

`algorithm_implement_update.md` §1.6 / §3.6 names the gap precisely: `add_edge`
Case A writes only `evals[1]` and `Q[:,1]`, so `evals[2:]` / `Q[:,2:]` go stale
after exactly one edge. Two consequences follow, both measured:

- `_refresh_mu` ([analytic_state.py:741-769](analytic_state.py#L741-L769)) runs a
  dense `eigvalsh` on **every** `add_edge`, purely to find μ.
- The drift check re-anchors with a full `eigh` on **half** of all steps — 20
  resets per 40 edges at every n ∈ {12, 16, 32, 64, 128}.

Closing it means propagating the *whole* spectrum through the rank-1 update —
the classical **Cuppen / Gu–Eisenstat symmetric rank-1 eigen-update**. This plan
adds that as a **second model** alongside the current one: a new
`AnalyticStateGE` subclass whose only difference is how `(evals, Q, mu)` are
maintained. Everything else — screening, ranking, reward, Case B MaxVol,
persistence, env integration — is *inherited verbatim*, so "identical otherwise"
is enforced by the language rather than by discipline.

### Expectation setting (measured — see the Appendix)

1. **In pure NumPy this will be slower than the current model at n ≤ 64.** The
   deliverable is a drift-free basis, μ without a dense solve, and real
   certificates — not speed. Speed needs the Numba path (§6), planned here but
   not built.
2. **Rankings will differ from the current model.** With a fresh basis, ψ = Qᵀa
   is correct on every step instead of only after an anchor, so
   `select_candidates` scores differ wherever the old basis was stale — i.e.
   most steps. "All other operations identical" means identical *code and
   formulas*, not identical *outputs*. Existing checkpoints need retraining
   against the new model before their numbers are comparable.

---

## 1. New module: `rank1_eigen_update.py`

Pure functions, no coupling to `AnalyticState`, so they can be tested directly
against `np.linalg.eigh` before anything is wired in.

**Entry point** — `rank1_eigh_update(evals, Q, i, j) -> (evals_new, Q_new, info)`.
Given `L = Q diag(evals) Qᵀ` and `a = e_i − e_j`, returns the eigendecomposition
of `L + aaᵀ`.

### Step 0 — project

`psi = Q[i,:] − Q[j,:]` (= `Qᵀa`). Two exact facts to exploit: `‖ψ‖² = 2` since
`Q` is orthogonal and `‖a‖² = 2`; and `ψ₀ = 0` since `q₀ ∝ 1` and `a ⟂ 1`.

### Step 1 — `deflate(d, psi, tol)`

Non-optional: path and cycle starts have exact multiplicities, and a repeated `d`
makes the corresponding secular interval empty. `tol = 8·eps·max(|d[-1]|, ‖ψ‖²)`
(LAPACK `dlaed2`'s rule).

- `|ψ_k| ≤ tol` → deflate; `(λ_k, q_k)` carries over unchanged. Index 0 always
  hits this branch exactly.
- consecutive survivors with `|d_k − d_l| ≤ tol` → Givens rotation in the `(k,l)`
  plane, `ρ = hypot(ψ_k, ψ_l)`, `c = ψ_l/ρ`, `s = −ψ_k/ρ`, giving `ψ_k ← ρ`,
  `ψ_l ← 0`; rotate columns `k,l` of `Q`, deflate `l` at `d_l`.

Returns `nd` (non-deflated indices, `d` ascending), `defl`, the rotated `Q`, and
`w = ψ[nd]`.

### Step 2 — `solve_secular_all_roots(d, w) -> (roots, Delta)`

Roots of `f(λ) = 1 + Σ w_k²/(d_k − λ)`, one per interval: `λ'_m ∈ (d_m, d_{m+1})`
for `m < K−1`, and `λ'_{K−1} ∈ (d_{K−1}, d_{K−1} + Σw²)`.

This is the **existing batched solver generalized from one shared bracket to
per-root brackets** — reuse the structure at
[analytic_state.py:408-454](analytic_state.py#L408-L454) directly:

- pick the shift pole `s(m) ∈ {m, m+1}` from the sign of `f` at the interval
  midpoint. The root then lies in the half nearest `s`, so the *other* pole sits
  at distance ≥ gap/2 and stays smooth over the search region;
- work in `δ[k,m] = d_k − d_{s(m)}` and solve for `η_m = λ'_m − d_{s(m)}`, so
  `d_k − λ'_m = δ[k,m] − η_m`. This shifted arithmetic is the numerical crux —
  forming `d_k − λ'` directly cancels catastrophically near the pole — and it is
  why `Delta` is *returned* rather than recomputed by the caller;
- iterate the same pole-free `h(η) = η(1+g) − w_s²` with `g = Σ_{k≠s} w_k²/(δ_k − η)`,
  safeguarded Newton with bisection fallback, terminating on **step size** not
  bracket width, compacting the active set each pass;
- sign bookkeeping is the one genuine change: `sign(f) = sign(h·η)`, and `η < 0`
  when shifted to the upper pole. In the existing solver `η > 0` always, so this
  is invisible there.

### Step 3 — `lowner_weights(d, roots, Delta)`

The Gu–Eisenstat correction, and the whole reason this is stable:

$$\hat w_k^2 \;=\; \frac{\prod_m (\lambda'_m - d_k)}{\prod_{m \ne k}(d_m - d_k)}
\;=\; (\lambda'_{K-1}-d_k)\prod_{m<k}\frac{\lambda'_m-d_k}{d_m-d_k}
\prod_{m=k}^{K-2}\frac{\lambda'_m-d_k}{d_{m+1}-d_k}$$

with `sign(ŵ_k) = sign(w_k)`. The paired form on the right keeps every factor
O(1). The identity itself comes from evaluating the characteristic polynomial at
`λ = d_k`.

The point: the *computed* roots are exact for the perturbed weights `ŵ`, so
eigenvectors built from `ŵ` come out orthogonal to machine precision. Building
them from the original `w` loses orthogonality catastrophically when roots
cluster — §5 test 4 asserts exactly that, so the correction cannot be quietly
removed later.

### Step 4 — eigenvectors

`V[k,m] = ŵ_k / Delta[k,m]`, normalize columns; then `Q_new[:, nd] = Q[:, nd] @ V`
— one GEMM, `2nK²` flops. Deflated columns carry over untouched.

### Step 5 — assemble

Merge deflated and new pairs, stable `argsort` ascending, permute `Q` columns to
match. Then pin the analytically-known null pair: `evals[0] = 0`,
`Q[:,0] = 1/√n`. Free, and removes the one drift mode no certificate localizes.

---

## 2. New state class: `AnalyticStateGE(AnalyticState)`

In `analytic_state_ge.py`. Overrides **four** methods; everything else inherited.

- **`add_edge`** — the preamble is copied verbatim from
  [analytic_state.py:1167-1210](analytic_state.py#L1167-L1210) (ψ, the `β` guard,
  Sherman–Morrison ΔR_G, adjacency write, `_rank1_update`, `compute_p`). Only the
  Case A / Case B block is replaced by a single `rank1_eigh_update` call, then
  `r`, `U`, `inv_lambda`, `inv_lambda_sq`, `mu` are refreshed from the fresh
  spectrum, then `_drift_check`.
  - `mu` = first eval `> λ₂ + eps_mult`, read straight off `evals` — **the
    unconditional dense `eigvalsh` is gone.**
  - Case A/B vanishes as a *state-update* distinction; deflation handles
    multiplicity uniformly, so `_rayleigh_ritz_update` is never called. `r` is
    still computed and still drives the Case A/B branch inside
    `select_candidates`, which is unchanged.
  - Keep `_check_secular_bracket` as a diagnostic. On a fresh basis its ~10
    violations per 40 edges should fall to ~0; if they don't, that is a bug
    signal worth having.
- **`_refresh_mu`** — read from `self.evals` instead of `eigvalsh(self.L)`. Kept
  as a method because `_recompute_exact` calls it in the `r+1 >= n` fallback.
- **`_cheap_drift`** — see §3.
- **`from_state_dict`** — `super().from_state_dict(state)` (it already uses
  `cls.__new__(cls)`, so it returns a GE instance), then zero the new diagnostic
  fields. `state_dict` needs **no** change: the GE state maintains exactly the
  same attribute set, so checkpoints stay structurally compatible.

Keep Sherman–Morrison for `L⁺` ([`_rank1_update`](analytic_state.py#L1013-L1039))
rather than rebuilding it from the spectrum — `O(n²)` against `O(n³)`, and
keeping the two objects on *independent* update paths is precisely what makes the
new `cert_spec_trace` certificate in §3 meaningful.

---

## 3. Certificates

`max_steps_without_reset` default 256 for this class (from 32); `tau_drift`
unchanged. If the update is correct these should essentially never fire, and a
firing is the bug signal. Added to `_cheap_drift`, all `O(n)` except the last:

| name | formula | catches |
|---|---|---|
| `cert_trace_L` | `\|Σλᵢ − 2·\|E\|\|` | eigenvalue drift; exact, integer target |
| `cert_spec_trace` | `\|Σ_{i≥1} 1/λᵢ − tr(L⁺)\| / \|tr(L⁺)\|` | ties the spectrum to the independently maintained `L⁺` |
| `cert_orth` | `‖Q[:,k]ᵀQ − e_k‖∞`, `k = steps_since_exact mod n` | loss of orthogonality; `O(n²)`, deterministic, cycles all columns |

Keep Foster, `cert_trace`, and the Fiedler residual unchanged. Determinism
matters (`selfcheck.py` compares split-and-resumed runs), hence a column-cycling
probe rather than a random one.

---

## 4. Config and wiring

Follow the `rl_variant` pattern exactly.

- `config.py`: add `VariantConfig.spectral_backend: str = "analytic"`
  ([config.py:114](config.py#L114)), the matching key in the defaults dict
  (~line 271), and validation against `{"analytic", "gu_eisenstat"}` beside the
  existing `variant.name` check (~line 363).
- `env.py`: `GraphEnv.__init__` and `VectorGraphEnv.__init__` take
  `spectral_backend: str = "analytic"` and pass it down, mirroring `rl_variant`
  at [env.py:832](env.py#L832); `_init_analytic_state`
  ([env.py:182-190](env.py#L182-L190)) selects the class. Add it to the env
  `state_dict` / `load_state` round-trip next to `compute_spectral_each_step`.
- Three construction sites read `cfg.variant.*` and need the extra kwarg:
  [train.py:278](train.py#L278), [evaluate.py:570](evaluate.py#L570),
  [rollout.py:305](rollout.py#L305).
- New `../configs/rl_cpu_ge.yaml`: copy of `rl_cpu_standard.yaml` plus
  `variant.spectral_backend: gu_eisenstat`, with its own `output_dir` /
  `metrics_path` / checkpoint dir so runs don't collide.

---

## 5. Tests — `test_rank1_eigen_update.py`

Plain `main()` + `if __name__ == "__main__"` script with `sys.path` insertion,
matching every other test here (**there is no pytest in this repo**). Run as
`python -m analytic_tractable_test_deepseek.test_rank1_eigen_update`.

1. **Single update vs `eigh`.** Fixtures: path, **cycle** (exact multiplicity →
   exercises Givens deflation), star, complete-minus-k, ER at n ∈ {8, 16, 32, 64},
   and a hand-built graph with a deliberately repeated eigenvalue. Assert
   `max|λ' − λ'_exact| ≤ 1e-12·‖L‖`, `‖Q'ᵀQ' − I‖ ≤ 1e-13`,
   `‖L'Q' − Q'Λ'‖ ≤ 1e-12·‖L‖`.
2. **Deflation actually fires.** On cycle fixtures assert `K < n` — otherwise
   test 1 passes while silently never entering the Givens branch.
3. **100 sequential updates, no re-anchor** (`tau_drift=inf`,
   `max_steps_without_reset=10**9`). Assert the same three bounds still hold at
   the end. This is the actual claim being made.
4. **Löwner necessity.** Re-run test 3 with `ŵ` replaced by the raw `w`; assert
   orthogonality *degrades* past a loose threshold.
5. **Everything-else-identical.** Build `AnalyticState` and `AnalyticStateGE` on
   the same adjacency; both sit at an exact anchor after `__init__`, so their
   bases agree — assert `select_candidates` returns identical `pairs` and
   identical `info` arrays.
6. **Existing suites over both classes.** `test_analytic_state.py`,
   `test_select_candidates_fast.py`, `test_env_analytic.py` take a class/backend
   parameter and run twice. Note `_states()`
   ([test_select_candidates_fast.py:60-77](test_select_candidates_fast.py#L60-L77))
   includes a deliberately *stale* mid-rollout fixture; under GE that fixture is
   no longer stale, which is fine — those assertions are self-consistent
   (reference vs shipped on the same state), not absolute.

---

## 6. Numba plan

### 6.1 Where `add_edge` actually spends time (measured, 1 thread, mid-rollout path graph)

| component | n=32 | n=64 | scales |
|---|---|---|---|
| `psi = Q.T @ a` | 1.7 µs | 2.0 µs | 1.2× |
| `L_plus @ a` (Sherman–Morrison) | 1.6 µs | 2.0 µs | 1.2× |
| `_rank1_update` (2 outer products) | 7.3 µs | 13.1 µs | 1.8× |
| `compute_p` | 18.1 µs | 27.8 µs | 1.5× |
| **`solve_secular_equation` (brentq)** | **105.3 µs** | **105.3 µs** | **1.00×** |
| `update_fiedler_vector` | 18.2 µs | 18.5 µs | 1.0× |
| `_drift_residual` | 5.2 µs | 5.5 µs | 1.1× |
| `_cheap_drift` | 4.3 µs | 4.4 µs | 1.0× |
| `_refresh_mu` (dense `eigvalsh`) | 43.9 µs | 142.6 µs | 3.3× |
| `_recompute_exact` (dense `eigh` + L⁺ + p) | 175.2 µs | 499.2 µs | 2.9× |

LAPACK share of `add_edge`: **45 % at n=32, 69 % at n=64.**

Two conclusions, and the first one is the more actionable:

1. **`solve_secular_equation` costs 105 µs at both n=32 and n=64** — exactly
   size-independent, so it is 100 % `brentq`'s Python-callback overhead
   (~15 callbacks/edge). At n=32 that is more than *twice* the `eigvalsh` beside
   it and the largest single non-LAPACK cost in the function. **It does not need
   numba**: `solve_secular_equation_batch`, already in this module, does the same
   job without callbacks. Fix this first, independently of everything below.
2. **Numba's ceiling is bounded.** Making the entire spectral update free takes
   n=32 from ~396 µs to ~180 µs — 2.2×, not 10×. `compute_p`, the outer products,
   the certificates and the Python glue are untouched by it.

### 6.1a The overhead is NumPy's, not Python's — and the difference matters

It is tempting to read §6.1 as "too many Python function calls" and reach for
inlining. Measured, that is not what is happening:

| | cost |
|---|---|
| empty Python function call | **0.047 µs** |
| Python method call | 0.052 µs |
| `a + b`, shape `(32,)` | 0.595 µs |
| `np.add(a, b, out=…)` (no allocation) | 0.531 µs |
| `a.sum()` | 0.905 µs |
| `np.sum(a)` | **1.969 µs** |

A NumPy op costs 12–40× a Python call. Directly on `_secular_f`:

```
_secular_f(...) as shipped        4.459 us
same body inlined, no call        4.498 us   <- removing the call changes nothing
+ psi*psi hoisted out             3.821 us
PURE PYTHON loop over 32 floats   2.364 us   <- 1.9x faster than the numpy form
```

The cost is NumPy's per-operation machinery — argument parsing, the
`__array_ufunc__` protocol, dtype promotion, broadcast resolution, output
allocation. Of the ~105 µs `solve_secular_equation` spends per edge, the Python
calls are 15 × 0.047 ≈ 0.7 µs, under 1 %. Two consequences:

- **`np.sum(x)` → `x.sum()` is free money.** The free function detours through
  `fromnumeric._wrapreduction` (936 calls per 40 edges in the profile) for ~1 µs
  of pure Python argument-shuffling. This module uses `np.sum` throughout.
- **Neither NumPy nor Python is priced correctly for n ≈ 32 arrays**, which is the
  real argument for compiling. But note the limit: the Python-loop win applies to
  *single short vector ops* like `_secular_f`. The Gu–Eisenstat kernel needs
  `K² × iters ≈ 8000` inner steps per edge; in pure Python at ~40 ns each that is
  ~330 µs — worse than either alternative. Python loops are a micro-fix, not the
  strategy.

### 6.2 Estimated cost of the jitted kernel

The inner loop of `solve_secular_all_roots` is, per `(k, m, iteration)`:

```
t   = 1.0 / (delta[k,m] - eta[m])     # 1 sub, 1 divide
p   = w2[k] * t                       # 1 mul
g  += p                               # 1 add
gp += p * t                           # 1 mul, 1 add
```

Scalar `divsd` throughput is ~4 cycles on x86 and bounds the loop; the mul/adds
hide underneath, and iterations are independent so they pipeline. With `K ≈ n`
and ~5 iterations after active-set compaction, cost ≈ `4 · K² · iters` cycles:

| n | scalar est. | AVX2-vectorized est. | + `Q[:,nd] @ V` GEMM |
|---|---|---|---|
| 32 | ~11 µs | ~5.5 µs | +3.7 µs |
| 64 | ~44 µs | ~22 µs | +14 µs |
| 128 | ~175 µs | ~87 µs | +117 µs |

Note the shape: **O(n²) with a ~4-cycle constant, against `eigh`'s O(n³)**.
Projected `add_edge` totals, and what they replace:

| n | GE + numba (est.) | today's `add_edge` | honest baseline (`_recompute_exact` every step) |
|---|---|---|---|
| 32 | ~140 µs | 396 µs | 175 µs |
| 64 | ~230 µs | 755 µs | 499 µs |
| 128 | ~450 µs | 1856 µs | ~1600 µs |

Break-even against full recomputation lands near n≈32, ~2× at n=64, ~3.5× at
n=128. In pure NumPy the same algorithm is a 4–8× *loss* at those sizes, so numba
is what flips the verdict — but it is a 2–3× win, not a landslide.

### 6.3 Kernel boundaries

`rank1_kernels.py` holds each kernel **twice** — a NumPy reference `*_ref` and a
jitted `*_jit` — with a module-level `_USE_NUMBA` switch and a `set_backend()`
entry point so §6.6 can test both in one process. `rank1_eigen_update.py` stays
pure orchestration and never imports numba.

| kernel | why it's the right cut |
|---|---|
| `deflate` scan + Givens | already a scalar loop with data-dependent branching; NumPy buys nothing |
| `solve_secular_all_roots` | the dominant cost — ~42 array ops × ~8 iterations collapses to one `K × iters` scalar double loop |
| `lowner_weights` | `K²` products with data-dependent pairing; awkward vectorized, trivial compiled |

`Q[:, nd] @ V` stays NumPy — it is already one GEMM and numba would only make it
worse. Signatures must be numba-friendly: positional args only, no keyword or
optional parameters, C-contiguous `float64` arrays in and out, tuple returns, no
Python objects crossing the boundary.

### 6.4 Compilation settings that actually matter

- **`fastmath=False` — mandatory, not a default to accept passively.** `fastmath`
  licenses reassociation of float ops. That breaks `selfcheck.py`'s
  split-and-resume hash equality, and it breaks the §3 certificates, which are
  *exact* invariants whose whole value is that any nonzero reading is a real bug.
- **`cache=True`** — first-call compilation is ~1–3 s per kernel. Training runs
  in-process (`VectorGraphEnv` holds a `List[GraphEnv]`, not subprocesses) so it
  amortizes over a run, but the test scripts are short-lived and would otherwise
  pay it every invocation. Caching writes `__pycache__/*.nbi|.nbc` — add to
  `.gitignore`.
- **No `parallel=True`.** Thread scheduling is nondeterministic, which
  `selfcheck.py` would catch, and the Appendix measurement shows threading
  actively *hurts* at these matrix sizes.
- **`error_model='numpy'`** so division by zero yields `inf` rather than raising —
  matching the `np.errstate(divide="ignore")` semantics the existing solver
  relies on for its `accept`/bisect fallback.

### 6.5 Dependency risk — check before committing to numba

Numba's supported NumPy range **lags NumPy releases by several months**, and it
pins an upper bound. The interpreter used for these measurements has NumPy 2.5.0;
verify a numba release supports it before adopting, and pin both in
`requirements-rl.txt` as an optional extra (`pip install -r requirements-rl.txt`
must still succeed without it). The import guard

```python
try:
    from numba import njit
except ImportError:
    def njit(*a, **k): return (lambda f: f) if not a else a[0]
```

keeps absence free but does not remove the risk of numba *pinning NumPy
backwards* for everyone. If that bites, a Cython or plain C extension is the
fallback — the §6.3 boundaries are the same either way, which is the point of
cutting them there.

### 6.6 Validating the two paths agree

- Every §5 test parameterized over `set_backend("ref"|"jit")`, asserting results
  agree to `1e-14` — not just that both pass their own thresholds.
- One full run under `NUMBA_DISABLE_JIT=1` to exercise the interpreted path.
- `selfcheck.py` under the jitted backend must still report identical hashes;
  this is the real guard on `fastmath` and threading regressions.

---

## 7. Verification

```bash
# unit — the new kernel against LAPACK, in isolation
python -m analytic_tractable_test_deepseek.test_rank1_eigen_update

# regression — existing suites, both backends
python -m analytic_tractable_test_deepseek.test_analytic_state
python -m analytic_tractable_test_deepseek.test_select_candidates_fast
python -m analytic_tractable_test_deepseek.test_env_analytic
python -m analytic_tractable_test_deepseek.test_policy_integration

# determinism — split-and-resume must still match under the new backend
python -m analytic_tractable_test_deepseek.selfcheck --config configs/rl_cpu_ge.yaml \
    --total-env-steps 4000 --split-env-steps 2000 --device cpu
```

Plus a new `bench_ge.py` reporting, for both backends over 40 edges at
n ∈ {12, 32, 64, 128}: µs/edge, re-anchors per 40 edges (expect 20 → ~0), final
λ₂ error vs `eigvalsh`, and each certificate's peak value. Nothing currently
reads `steps_since_exact` / `cert_*` / `bracket_violations`, so this script is
their first consumer and is what makes the "certificates never fire" claim
checkable.

**Acceptance:** unit bounds in §5 hold; existing suites pass on both backends;
`selfcheck` reports identical hashes; re-anchors drop to ~0 with λ₂ accuracy no
worse than today's ~1e-15.

---

## Appendix — why the NumPy version will not be faster (measured)

Same machine, numpy 2.5.0, `OMP/OPENBLAS/MKL_NUM_THREADS=1`. Retained because it
sets the expectation above and motivates §6.

**The dispatch floor is size-independent.** `d+v` on shape `(n,)`: 0.61 µs at
n=12, 0.61 at n=32, 0.61 at n=64, 0.73 at n=256. Every NumPy call costs ~0.6 µs
of interpreter + `__array_ufunc__` + dtype/broadcast + allocation regardless of
size. At n=32 a `(32,32)` elementwise op does 1024 flops in 0.95 µs (~1 GFLOP/s);
`eigh` at n=32 does ~3·10⁵ flops in 48.4 µs (~6 GFLOP/s) in *one* dispatch.

**Flops aren't the problem.** A Cuppen update at n=32 is ~120k flops vs `eigh`'s
~300k — 2.5× *fewer* — but delivered in ~400 chunks of ~300 flops. One secular
iteration measured 26.7 / 29.5 / 36.8 / 61.1 µs at n = 12 / 32 / 64 / 128;
×8 iterations = 236 µs at n=32 against `eigh`'s 48 µs.

**The existing code already demonstrates it.** 40 random edges onto a path graph,
real `add_edge` vs a naive `eigh` per edge:

| n | `add_edge` | naive `eigh` | ratio | resets/40 | λ₂ err |
|---|---|---|---|---|---|
| 12 | 373 µs | 33 µs | 11.3× slower | 20 | 3.6e-15 |
| 32 | 396 µs | 106 µs | 3.8× slower | 20 | 1.1e-16 |
| 64 | 755 µs | 390 µs | 1.9× slower | 20 | 5.8e-16 |
| 128 | 1856 µs | 1359 µs | 1.4× slower | 20 | 5.3e-16 |

`cProfile` at n=32: `eigh` 2.0 ms + `eigvalsh` 2.0 ms out of 22 ms cumulative —
82 % of the time is NumPy dispatch, 18 % is the LAPACK it exists to avoid. The
ratio trend 11.3 → 3.8 → 1.9 → 1.4 puts break-even for the current code at
n ≈ 200–300; a correct Cuppen update moves that to roughly n ≈ 100, not to n=32.

**Two orthogonal wins, not part of this plan.** Pinning BLAS threads to 1 is
worth 2.5–3.5× on `eigh` at these sizes (n=32 381→106 µs, n=64 1053→390 µs,
n=128 3640→1359 µs) — note `configs/rl_cpu_standard.yaml` sets
`determinism.num_threads: 6`. And `add_edge` still drives `brentq` with a Python
lambda (~15 callbacks/edge) instead of the batched solver already in this module.

*Measurements used
`C:/Users/avalon/Documents/AFRL/Algebraic-Connectivity-Optimization-CDC2026/.venv/Scripts/python.exe`
(numpy 2.5.0, scipy 1.18.0) — the repo's own interpreter has no numpy installed.*
