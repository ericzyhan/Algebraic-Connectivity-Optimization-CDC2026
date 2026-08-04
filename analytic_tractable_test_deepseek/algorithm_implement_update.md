# Analytic Edge-Selection Algorithm — Implementation Update

This document specifies the algorithm **as implemented** in `analytic_state.py`
after the Stage-2 vectorisation work. It supersedes the Case A portions of
`algorithm.md` / `algorithm_implementation.md` where they differ; the
differences are called out explicitly in §2 and §3.

Three corrections are folded in relative to the previous implementation:

| # | Change | Nature |
|---|---|---|
| 1 | Batched secular solve (`solve_secular_equation_batch`) | Performance; algebraically equivalent |
| 2 | Pole-free secular bracket (`secular_upper_endpoint`) | **Correctness** — bracket could contain a pole |
| 3 | ΔR_G screening formula in the eigenbasis | **Correctness** — ψ was used as the incidence vector |
| 4 | Collision endpoint tested against the clamped bracket, then re-anchored (§2.10) | **Correctness** — divide-by-zero produced a NaN Fiedler vector that the drift check failed open on |

---

## 1. Overview — what is computed, when, and why

### 1.1 The problem

Given a connected graph on `n` nodes, repeatedly add one edge at a time to
raise the algebraic connectivity λ₂ (the Fiedler value). At each step an RL
policy must choose an edge, but the non-edge set has size
`k = O(n²)` — far too many actions. The algorithm's job is to narrow those
`k` non-edges down to ≤ 32 high-quality candidates **without** recomputing a
dense eigendecomposition per candidate, and then to update the spectral state
incrementally once the policy commits to one edge.

### 1.2 State carried between steps

`AnalyticState` maintains, for the current adjacency `adj`:

| Symbol | Code | Meaning |
|---|---|---|
| `L` | `self.L` | Laplacian |
| `Λ` | `self.evals` | eigenvalues, ascending; `evals[0] = 0` |
| `Q` | `self.Q` | eigenvectors, columns aligned with `evals` |
| `r` | `self.r` | multiplicity of λ₂ |
| `U` | `self.U` | `Q[:, 1 : 1+r]`, the λ₂ eigenspace |
| `μ` | `self.mu` | first eigenvalue strictly greater than λ₂ |
| `L⁺` | `self.L_plus` | Moore–Penrose pseudoinverse |
| `1/λ`, `1/λ²` | `self.inv_lambda`, `self.inv_lambda_sq` | with index 0 set to 0 |
| `R_G` | `self.R_G` | total effective resistance, `n·tr(L⁺)` |
| `p` | `self.p` | resistance curvature vector (below) |
| `p_min` | `self.p_min` | `min_i p_i` |

The resistance curvature is

$$p_i = 1 - \tfrac12 \sum_{b \sim i} R_{\text{eff}}(i,b), \qquad R_{\text{eff}}(u,v) = L^+_{uu} + L^+_{vv} - 2L^+_{uv}$$

`p` is maintained as a full vector (not just its minimum) because the drift
certificate needs `Σᵢ pᵢ`, which Foster's theorem pins to exactly 1.

### 1.3 Flow of one environment step

```
        ┌─────────────────────────────────────────────┐
        │ select_candidates()  — narrow k → ≤32       │
        │                                             │
        │  Stage 1: score ALL k non-edges (cheap)     │
        │    R_e  from L⁺        [O(1) per non-edge]  │
        │    F    from U         [O(r) per non-edge]  │
        │    keep top-32 by R_e ∪ top-32 by F, cap 64 │
        │                                             │
        │  Stage 2: score the ≤64 survivors (costly)  │
        │    form ψ = Qᵀa only for survivors          │
        │    r = 1 → Case A: Δλ₂, ΔR_G, Δp_min → rank │
        │    r > 1 → Case B: MaxVol on the eigenspace │
        └─────────────────────────────────────────────┘
                              │  ≤32 pairs
                              ▼
                     policy picks one edge
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │ add_edge(i, j)  — commit and update state   │
        │   1. ψ = Qᵀa       (current basis)          │
        │   2. ΔR_G via Sherman–Morrison, update R_G  │
        │   3. rank-1 update of L and L⁺              │
        │   4. refresh p, p_min                       │
        │   5. r=1 → secular λ₂', Fiedler vector      │
        │      r>1 → Rayleigh–Ritz on U               │
        │   6. drift check → maybe full eigh          │
        │   7. refresh μ  (dense eigvalsh)            │
        └─────────────────────────────────────────────┘
```

### 1.4 Why the two-stage split

Stage 1 must touch every non-edge, so it may only use quantities readable in
`O(1)` from already-maintained matrices:

- **`R_e`** — effective resistance, read directly off `L⁺` by indexing. High
  `R_e` means the two endpoints are electrically distant, which is where a new
  edge does the most good.
- **`F`** — squared distance in the λ₂ eigenspace, `‖U[i] − U[j]‖²`. This is
  the first-order Δλ₂ proxy: for a simple λ₂, `Δλ₂ ≈ ψ₂²`, and `F` **is** `ψ₂²`
  when `r = 1`.

Both avoid forming the full `n × k` projection matrix, keeping Stage 1 at
`O(n²)` rather than `O(n·k) = O(n³)`.

The union of two top-32 lists (cap 64) hedges the two criteria against each
other: `R_e` is spectrum-wide, `F` is λ₂-specific, and an edge that scores well
on either is worth the exact evaluation.

Stage 2 then spends real effort — a secular root-find per candidate — on ≤64
edges instead of `O(n²)` of them.

### 1.5 Case A vs Case B

The split is on `r`, the multiplicity of λ₂:

- **Case A (`r = 1`, simple λ₂).** λ₂ is differentiable in the edge weight,
  the rank-1 update has a closed-form secular equation, and candidates can be
  *ranked* by a scalar reward. Returns the top-32.
- **Case B (`r > 1`, repeated λ₂).** λ₂ is not differentiable; adding one edge
  generally cannot raise a multiplicity-`r` eigenvalue, because the new edge
  can only "see" a one-dimensional slice of an `r`-dimensional eigenspace. The
  algorithm instead selects `r−1` edges that jointly break the degeneracy,
  chosen by MaxVol to maximise the volume spanned in `U`. Returns those as the
  candidate set.

### 1.6 Why an incremental state at all — and its current limit

The point of maintaining `Q`, `Λ`, `L⁺` is to avoid an `O(n³)` dense
eigendecomposition per step. **As implemented this goal is not met**, and the
document would be misleading without saying so:

- `_refresh_mu` runs a dense `eigvalsh(L)` **unconditionally on every**
  `add_edge` (step 7).
- The drift check triggers a full `eigh` on **half** of all steps — measured at
  exactly 20 resets per 40 edges at every `n ∈ {12,16,32,64,128}`, all from the
  drift threshold, none from the 32-step periodic cadence.

The reason for the second is structural: Case A writes only `evals[1]` and
`Q[:,1]`, leaving `evals[2:]`/`Q[:,2:]` stale. The drift residual is ~1e-13 on
the step right after an exact anchor and jumps to **O(1)** on the very next
step, so the incremental basis has an effective horizon of exactly **one edge**.

Net dense work per edge is therefore ~1 `eigvalsh` + 0.5 `eigh`, versus 1
`eigh` for a naive recompute-everything baseline. See §3.5.

---

## 2. Exact numerical methods

### 2.0 Constants

| Constant | Value | Role |
|---|---|---|
| `EPS_MULT_DEFAULT` | `1e-12` | λ₂ multiplicity tolerance (absolute); also the "strictly greater" tolerance for μ |
| `SECULAR_EPS_DEFAULT` | `1e-12` | secular bracket margin floor |
| `ZERO_PROJ_GUARD` | `1e-14` | ψ₂ below this ⇒ Fiedler vector does not move |
| `TAU_DRIFT_DEFAULT` | `1e-6` | drift threshold |
| `MAX_STEPS_WITHOUT_RESET_DEFAULT` | `32` | forced periodic full recomputation |
| `MAXVOL_EPS_DEFAULT` | `1e-6` | MaxVol stopping tolerance |
| `MAXVOL_ITER_DEFAULT` | `200` | MaxVol iteration cap |
| `ALPHA1/2/3_DEFAULT` | `0.10 / 0.20 / 0.70` | reward weights (convex) |

### 2.1 Multiplicity and μ

$$r = \#\{\lambda_i : |\lambda_i - \lambda_2| < \varepsilon_{\text{mult}},\; i \ge 1\}$$

μ is the first eigenvalue strictly above λ₂. After a full recompute it is read
off directly as `evals[r+1]` when `r+1 < n`; otherwise (and on every
`add_edge`) it comes from `_refresh_mu`, a dense `eigvalsh` of the **current**
`L`, returning `+∞` when no eigenvalue lies above λ₂.

Returning `+∞` rather than `λ_max` matters: `λ_max` is not a conservative
bracket but one spanning the whole spectrum, poles included, so a root-finder
could converge inside the wrong interval. `+∞` makes callers skip the solve.

### 2.2 Stage 1 screening

For all `k` non-edges simultaneously:

$$R_e[c] = L^+_{i_ci_c} + L^+_{j_cj_c} - 2L^+_{i_cj_c}, \qquad F[c] = \|U[i_c,:] - U[j_c,:]\|_2^2$$

$$\mathcal{I}_{\text{cand}} = \operatorname{unique}\big(\text{top-32}(R_e) \cup \text{top-32}(F)\big), \quad \text{truncated to } 64$$

Ties are broken by `mergesort` (stable) for determinism.

### 2.3 The secular equation

For the rank-1 update `L' = L + aaᵀ` with `a = e_i − e_j`, writing `ψ = Qᵀa`:

$$\det(L' - \lambda I) = \det(\Lambda - \lambda I)\cdot\Big(1 + \sum_i \frac{\psi_i^2}{\lambda_i - \lambda}\Big)$$

so the updated eigenvalues are the roots of

$$f(\lambda) = 1 + \sum_i \frac{\psi_i^2}{\lambda_i - \lambda} = 0$$

Index 0 drops out identically: `ψ₀ = q₀ᵀa = n^{-1/2}·1ᵀa = 0` because
`a = e_i − e_j` sums to zero.

**Monotonicity.** `f'(λ) = Σ ψᵢ²/(λᵢ−λ)² > 0`, so `f` is strictly increasing
between consecutive poles. On a pole-free `(λ₂, μ)` it runs `−∞ → +∞`: exactly
one root, and a guaranteed bracket. The *sign of f alone* locates λ relative to
the root, which is what makes the bracket update vectorisable.

**Non-convexity.** `f''(λ) = Σ 2ψᵢ²/(λᵢ−λ)³` is **not** single-signed on
`(λ₂,μ)` — the pole below contributes negatively, poles above positively. Plain
Newton is therefore not unconditionally convergent, which is why the literature
(and `algorithm.md`) reach for BNS rational interpolation.

### 2.4 The bracket must be pole-free — **correction #2**

The poles of `f` sit at the **stored** `lambdas`, but μ comes from
`_refresh_mu`, computed on the **current** `L`. After the first incremental
step past an exact anchor these disagree, and a stale `lambdas[2]` can lie
strictly inside `(λ₂, μ)`. `f` is then non-monotonic on the bracket with two
roots in it, and the root-finder converges to whichever it stumbles into.

Measured at `n = 32`, 7 edges in:

```
lambda2 = 2.790368304843   mu = 2.911837945535
evals[1:5] = [2.7903683, 2.79056354, 2.97070744, 2.99817181]
                          ^^^^^^^^^^ stale pole INSIDE the bracket

exact λ₂' (eigh)  = 2.790509286737
scalar brentq     = 2.790661708292   err 1.5e-4
safeguarded Newton= 2.790497786376   err 1.2e-5
```

**Fix.** Interlacing for a rank-1 PSD update gives `λ_k ≤ λ'_k ≤ λ_{k+1}`, so
λ₂′ lies below the next *stored* eigenvalue. The upper endpoint becomes

$$\texttt{secular\_upper\_endpoint} = \min\big(\mu,\; \min\{\lambda_i : \lambda_i > \lambda_2 + \varepsilon_{\text{mult}}\}\big)$$

applied in **both** the scalar and batched solvers. When the basis is exact
(always immediately after `_recompute_exact`, where `r = 1` gives `μ = evals[2]`)
this returns μ unchanged, so the common path is bit-identical to before.

### 2.5 Bracket endpoints and the three-way outcome

$$\text{gap} = \mu - \lambda_2,\quad \text{margin} = \max(\varepsilon_{\text{sec}},\, 10^{-8}\,\text{gap}),\quad a = \lambda_2 + \text{margin},\quad b = \mu - \text{margin}$$

A candidate is *bracketed* iff `f(a)`, `f(b)` are both finite and
`¬(f(a)·f(b) > 0)` — the non-strict form, so an exact endpoint root still
counts. Outcomes:

| Condition | Result |
|---|---|
| `μ = ∞` | λ₂ unchanged (no bracket exists) |
| bracketed | the root in `(a, b)` |
| not bracketed, `\|ψ₂\| ≥ 1e-14`, `f(b) < 0` | **μ** — the μ-endpoint collision |
| otherwise | λ₂ unchanged |

The μ-collision arises when `ψ₃ ≈ 0`: `f` stays negative all the way up and the
root sits on the endpoint. The previous analytic implementation silently
reported `Δλ₂ = 0` for such edges.

### 2.10 The collision endpoint is a pole — **correction #4**

The endpoint returned on a collision is `SecularUpperEndpoint(Λ, λ₂, μ)`, and
whenever the `min(above)` branch of §2.4 wins, that value **is a stored `λ_k`**
— a pole of `f`. Two consequences, both live before this fix:

1. `add_edge` tested the collision as `|λ₂′ − μ| < 1e-10` against the
   *unclamped* `μ`. Every collision resolved by the stored bound therefore
   missed the guard and fell through to the eigenvector sum, which divides by
   `λ_k − λ₂′ = 0` **exactly**. The resulting `±inf` coefficients produce a NaN
   `Q[:,1]`, and NaN propagates through the observation into the policy —
   observed as `Categorical(logits=[nan × 32])` killing a training run.
   Reproduced on path/cycle starts from `n = 6` up; 277 corrupted states in
   ~83k edges.

2. `_drift_check` did not catch it: `max()` propagates NaN and `nan > τ_drift`
   is `False`, so the certificate **failed open** and left the corrupt state in
   place. It now treats any non-finite drift as drift.

Substituting `q_k` for the divergent sum is *not* a fix. With `ψ_k ≈ 0`,
`L'q_k = Lq_k + a(aᵀq_k) = λ_k q_k + ψ_k a ≈ λ_k q_k`, so `(q_k, λ_k)` is a
genuine eigenpair of the updated `L` — just not the **second** one when `Λ[2:]`
is stale. The residual then reads ~0, the drift check passes, and λ₂ is retained
wrong by as much as **2.0** (measured). A wrong-index eigenpair is invisible to
every certificate carried here, so the collision branch re-anchors with
`RecomputeExact()` instead.

Cost is nil in practice: re-anchors stay at **0.50/edge** at `n ∈ {12,16,32,64}`
on both path and random-tree starts — unchanged from the rate §1.6 already
records — because collisions land on steps the drift check was going to reset
anyway. Maintained λ₂ accuracy over 85k edges is `1.5e-10` worst case.

### 2.6 Batched solve on the pole-free reformulation — **correction #1**

The scalar path drives `scipy.optimize.brentq` with a Python callable, once per
candidate: ~58 candidates × ~12.6 iterations ≈ **730 interpreter round-trips per
step**, each performing 64 flops behind ~3.3 µs of NumPy dispatch (≈150:1
overhead). Every candidate is an *independent scalar root-find over the same
`lambdas`* — precisely the shape that vectorises.

Direct Newton on `f` does not work in batch: near the λ₂ pole `f` is nearly
vertical, every Newton step leaves the bracket, and the safeguard degrades to
bisection needing ~40 halvings. Measured at `n = 64`: **median 5 iterations,
maximum 40**; at `n = 16`, maximum 67.

**Reformulation.** Substitute `η = λ − λ₂` and split the λ₂ pole out:

$$f = 1 - \frac{\psi_2^2}{\eta} + g(\lambda), \qquad g(\lambda) = \sum_{i \ne 1} \frac{\psi_i^2}{\lambda_i - \lambda}$$

Multiplying by `η > 0` removes the pole entirely:

$$\boxed{\;h(\eta) = \eta\,(1 + g) - \psi_2^2 = 0, \qquad h'(\eta) = 1 + g + \eta\,g'\;}$$

- `h = η·f` with `η > 0`, so **h and f have identical signs** — the bracketing
  decided from `f` transfers unchanged.
- On a pole-free interval every surviving term of `g` has `λᵢ ≥ μ > λ` (index 0
  contributes nothing since `ψ₀ = 0`), hence `g > 0` and `g' > 0`, so
  `h' > 1 > 0`. `h` is strictly increasing and well scaled exactly where `f` is
  not.

This drops the worst case from 40–67 iterations to **5–12**.

**Iteration** (all `m` candidates at once, `tol = 1e-12`, `max_iter = 100`):

```
D  = lambdas[:,None] - (lambda2 + eta)[None,:]     # (n, |act|)
T  = P_rest / D
g  = T.sum(0);   g' = (T/D).sum(0)
h  = eta*(1+g) - psi2_sq
h' = 1 + g + eta*g'
lo = where(h < 0, eta, lo);   hi = where(h > 0, eta, hi)      # h increasing
step   = eta - h/h'
accept = isfinite(step) & (step > lo) & (step < hi)
eta_new = where(accept, step, 0.5*(lo+hi))                    # bisection fallback
```

Two details that are load-bearing:

1. **Terminate on step size, not bracket width.** Newton closes on this root
   from one side, so only one of `lo`/`hi` ever moves and `hi − lo` stalls at
   the distance to the far endpoint. A width test never fires and burns all 100
   iterations on an answer converged since iteration ~6.
2. **Compact the active set.** Convergence is uneven; under an
   all-must-converge test a single straggler drags every other column with it
   (measured: 43 columns, median 5, max 40 — an ~8× waste). Indexing by the
   still-active columns makes cost track the *mean* iteration count, and shrinks
   the `(n, |act|)` work per iteration.

A non-finite `h` or `h'` fails the `accept` test and bisects, so a pathological
column degrades rather than diverging.

### 2.7 Δλ₂, ΔR_G, Δp_min for Case A ranking

**Δλ₂** — from the batched solve above, `0` for guarded columns (`|ψ₂| < 1e-14`).

**ΔR_G — correction #3.** In the eigenbasis, with `P2 = ψ⊙ψ`:

$$\beta = 1 + \sum_i \frac{\psi_i^2}{\lambda_i} \;=\; 1 + a^\top L^+ a \;=\; 1 + R_{\text{eff}}(i,j)$$

$$\Delta R_G^{\text{red}} = n\,\frac{\sum_i \psi_i^2/\lambda_i^2}{\beta} \;=\; n\,\frac{a^\top (L^+)^2 a}{\beta}$$

reported as a **positive reduction** `R_G^old − R_G^new`. Implemented as
`beta = 1 + inv_lambda @ P2` and `n * (inv_lambda_sq @ P2) / max(beta, 1e-14)`,
which matches `algorithm.md` line 73 verbatim and needs no matmul.

> **What was wrong.** The previous code computed `V = L_plus @ Bp;
> beta = 1 + Σ Bp·V`, using `Bp = ψ = Qᵀa` as though it were the incidence
> vector `a`. That evaluates `ψᵀL⁺ψ = aᵀ(Q L⁺ Qᵀ)a`, **not** `aᵀL⁺a`. At
> `n = 24` it gave `β = 1.477` where all three independent correct forms give
> `1.753`, leaving only **3 of the top-8** candidates by ΔR_G in common with the
> true ordering.
>
> The corresponding computation inside `add_edge` (§2.9 step 2) was **always
> correct** — there `a` genuinely is the incidence vector. So the maintained
> state was never wrong; only the candidate ranking was. This is why no existing
> test caught it.

**Δp_min** — closed form. Since `p` is already maintained and the first-order
rule holds `L⁺` fixed while appending exactly one resistance term:

$$p_i^{\text{new}} = p_i - \tfrac12 R_{\text{eff}}(i,j), \qquad p_j^{\text{new}} = p_j - \tfrac12 R_{\text{eff}}(i,j)$$

$$\Delta p_{\min}[c] = \min\Big(\min(p_{i_c}, p_{j_c}) - \tfrac12 R_e[c],\; p_{\min}\Big) - p_{\min}$$

with `R_eff(i,j)` already computed for every non-edge as `r_e` in Stage 1. The
prior loop form re-derived the neighbour sums at ~1575 interpreted calls per
step. Algebraically exact; differs only in float summation order (~1e-16).

Note `Δp_min ≤ 0` identically, since `R_eff ≥ 0` (`L⁺` is PSD). See §3.6.

**Reward and ranking:**

$$\mathcal{R}_c = \alpha_1 \frac{\Delta\lambda_2}{n} + \alpha_2\,\eta_R \frac{\Delta R_G^{\text{red}}}{n^2} + \alpha_3\,\eta_p\,\Delta p_{\min}$$

sorted descending with stable `mergesort`; top-`min(32, m)` returned.

### 2.8 Case B — MaxVol

1. `Z_mult = Uᵀ B'` — project candidates onto the λ₂ eigenspace, shape `(r, m)`.
2. Keep columns with `‖Z[:,j]‖ ≥ τ`, τ = 75th percentile of the column norms;
   if fewer than `r` survive, relax to the 50th then 25th percentile.
3. Thin QR: `Z_keepᵀ = Q_R R`, `Q_R ∈ ℝ^{m'×r}` orthonormal.
4. MaxVol on `Q_R`: initial active set from LU row pivoting, then swap the row
   maximising `|C| = |Q_R Q_R[I,:]^{-1}|` until `max|C| ≤ 1 + 1e-6` or 200
   iterations.
5. Return `r` selected edges, **discarding the last** when more than one
   survives, yielding `r−1`. If the percentile filter left fewer than `r`
   columns, those columns are returned as-is without the MaxVol step.

### 2.9 `add_edge` — the state update

1. `ψ = Qᵀa` on the **current** basis (before any mutation).
2. Sherman–Morrison ΔR_G: `Lpa = L⁺a`, `β = 1 + aᵀLpa`. If `β` is non-finite
   or `≤ 1e-14`, full recompute and return. Else
   `ΔR_G^red = n·(LpaᵀLpa)/β`, `R_G ← R_G − ΔR_G^red`.
3. Rank-1 update:
   $$L \leftarrow L + aa^\top, \qquad L^+ \leftarrow L^+ - \frac{v v^\top}{\beta},\quad v = L^+a - \operatorname{mean}(L^+a)$$
   Centring the `O(n)` vector `v` replaces double-centring the `n×n` matrix:
   `L⁺1 = 0` implies `1ᵀ(L⁺a) = 0` exactly, so the outer product is already
   centred. The spec's `(I−J)L⁺(I−J)` is roundoff hygiene costing ~2n³ flops
   for what one mean subtraction achieves.
4. Refresh `p`, `p_min` — **before** the drift check, since the Foster
   certificate reads `p` and a stale `p` would register as drift that isn't
   there.
5. Spectral update:
   - **`r = 1`:** solve the secular equation for λ₂′.
     - **Endpoint collision** — `λ₂′ > λ₂` and λ₂′ equals the bracket's upper
       endpoint to within `1e-10`, where the endpoint is the one the solver
       actually used, `SecularUpperEndpoint(Λ, λ₂, μ) = min(μ, next stored λ)`,
       **not** `μ` (see §2.10): `RecomputeExact()`.
     - Else if `λ₂′ > λ₂`:
       $$u_2' \propto \sum_i \frac{\psi_i}{\lambda_i - \lambda_2'} q_i$$
       normalised. Write `Q[:,1] = u₂`, `evals[1] = λ₂′`, then drift-check.
     - Else (edge does not raise λ₂): Fiedler pair unchanged, but still
       drift-check so the periodic reset fires during no-raise streaks.
   - **`r > 1`:** Rayleigh–Ritz — `H = UᵀLU`, `Θ,V = eigh(H)`, `U ← UV`.
6. Drift check (§3.1) — may trigger a full `eigh`.
7. `μ ← _refresh_mu(evals[1])` — dense `eigvalsh`, unconditional.

---

## 3. Sanity checks

### 3.1 Runtime drift certificates

Three complementary checks run inside `_drift_check` on every `add_edge`; the
maximum is compared against `τ_drift = 1e-6`, and a full `_recompute_exact()`
fires on exceedance **or** after 32 incremental steps.

| Check | Formula | Covers | Cost |
|---|---|---|---|
| Eigenpair residual | `‖L u₂ − λ₂ u₂‖` | the Fiedler pair | `O(n²)`, one matvec (0.1–1.5% of a dense eigh) |
| Foster | `\|Σᵢ pᵢ − 1\|` | `L⁺` via `p` | `O(n)`, free — `p` already built |
| Trace | `\|R_G − n·tr(L⁺)\| / \|R_G\|` | the `R_G` accumulator | `O(n)` |

Both Foster and trace are *exact invariants* of a correct state, so any nonzero
value is drift by definition rather than approximation error. Foster's theorem
gives `Σ_{(i,j)∈E} R_eff(i,j) = n − 1`, hence `Σᵢ pᵢ = n − (n−1) = 1`
identically.

They are complementary, not substitutes: an earlier revision replaced the
residual with the two cheap certificates and λ₂ silently drifted to 8e-2 within
three edges, because neither can observe the Fiedler pair at all.

### 3.2 Secular bracket certificate

`_check_secular_bracket` certifies the root against a closed-form two-sided
bound. With `δ' = μ − λ₂` and `ψ₂ = q₂ᵀa`:

$$\frac{\delta'}{\delta' + 2}\,\psi_2^2 \;\le\; \Delta\lambda_2 \;\le\; \psi_2^2$$

Both bounds are `O(1)` (ψ is already formed), verified against exact
eigendecompositions on 90 random `(graph, edge)` pairs with zero violations.

**Diagnostic only — it does not force a recomputation.** Measurement showed
re-anchoring on violation buys nothing: over 40-edge runs at `n = 12/32/64/128`
the final λ₂ error is bit-identical (3.55e-15, 2.22e-16, 4.83e-15, 5.69e-16)
whether the guard re-anchors or merely counts. It fires ~10 times per 40 edges,
and those firings are benign — Case A leaves `Q[:,2:]`/`evals[2:]` stale between
anchors, so the bracket is evaluated against a partly out-of-date basis, while
the eigenpair residual already catches every case that actually moves λ₂.
Violations are recorded in `bracket_violations`.

### 3.3 Test suite — `test_select_candidates_fast.py`

| Test | Asserts | Achieved |
|---|---|---|
| `test_batch_matches_scalar_solver` | batched == scalar over 7 fixtures, 296 roots | worst 3.0e-11 rel-to-gap |
| `test_batch_semantics_edge_cases` | `μ=∞`, empty block, `ψ₂=0` collision all match scalar | exact |
| `test_secular_root_vs_exact_eigh` | batch root vs true λ₂ from `eigh` | 5.6e-13 |
| `test_delta_pmin_closed_form` | closed form == `_cheap_delta_pmin` loop, 286 candidates | 8.9e-16 |
| `test_delta_pmin_sign` | `Δp_min ≤ 0` on every candidate | holds |
| `test_end_to_end_ranking_unchanged` | selected **set** identical; reordering only on ties | max \|Δreward\| 3.6e-15 |
| `test_rollout_trajectory_identical` | 25-edge greedy rollout vs exact `eigh` | 2.7e-15 |
| `test_drg_formula` | shipped ΔR_G vs 3 independent references, 128 candidates | 1.2e-15 rel |

The ΔR_G test cross-checks three derivations that must agree:
`1 + R_eff(i,j)` (from Stage 1's own `r_e`), the eigenbasis form, and direct
`aᵀL⁺a`. They agree with each other to **8.9e-16** and with the shipped value to
**1.2e-15** relative.

`_cheap_delta_pmin` is deliberately retained in the module as the reference
implementation the closed form is tested against.

> **On the equivalence assertion.** `test_end_to_end_ranking_unchanged` checks
> set equality plus tie-bounded reordering, not bitwise order equality. On
> symmetric fixtures (path graphs) equal-reward candidates are common, and the
> vectorised Δλ₂/Δp_min differ from the scalar ones by ~1 ULP — enough to swap
> two exactly-tied entries under `argsort`. Asserting bitwise order would be
> testing float tie-breaking, not equivalence. Any position that reorders must
> have rewards agreeing to `1e-13`.

Pre-existing suites `test_analytic_state`, `test_env_analytic`,
`test_policy_integration` all pass unchanged.

### 3.4 Behaviour change from the two correctness fixes

Both change which edges get selected, so checkpoints trained against the old
rankings need retraining. Paired greedy rollouts, 45 graphs (`n ∈ {16,32,64}`),
25 edges each:

| Fix | Decisions changed | Δ final λ₂ | t | Verdict |
|---|---|---|---|---|
| Pole-free bracket (#2) | 69% | +0.0042 ± 0.016 | +0.25 | no measurable effect |
| ΔR_G eigenbasis (#3) | 90% | **+0.166 ± 0.054** | **+3.08** | better on 37/45 |

Maintained-λ₂ accuracy is unaffected by the bracket fix (~5e-15 either way),
because `add_edge` re-anchors regardless.

### 3.5 Performance

Case A Stage 2 scoring, reference loops vs vectorised:

| n | reference | vectorised | speedup |
|---|---|---|---|
| 12 | 1.82 ms | 0.13 ms | 13.6× |
| 32 | 2.61 ms | 0.15 ms | **17.6×** |
| 64 | 2.86 ms | 0.33 ms | 8.7× |
| 128 | 3.07 ms | 0.20 ms | 15.4× |

Full `AnalyticState` step loop, 20 edges:

| n | before | after | speedup | vs naive dense-`eigh` baseline |
|---|---|---|---|---|
| 12 | 32.2 ms | 7.4 ms | 4.4× | 86× → 20× slower |
| 32 | 49.8 ms | 15.2 ms | 3.3× | 16× → 5.4× slower |
| 64 | 72.0 ms | 23.5 ms | 3.1× | 7.6× → **2.8× slower** |

### 3.6 Known issues not addressed here

1. **The dense-solve floor.** ~1 `eigvalsh` + 0.5 `eigh` per edge (§1.6). Until
   Case A propagates the full spectrum through the rank-1 update — the
   Cuppen / Gu–Eisenstat problem — there is no `n` at which this beats a naive
   recompute. It is now ~22% of kernel runtime and is the binding constraint.

2. **Stage 1 is now the bottleneck.** At `n = 128` it is **~69%** of
   `select_candidates`, dominated by `non_edges` materialising a Python list of
   ~n²/2 tuples and converting back to an array, plus two argsorts over all
   non-edges.

3. **Δp_min stops discriminating as n grows.** It is `≤ 0` identically, and
   measured spread across candidates:

   | n | candidates with `Δp_min = 0` | discriminating spread (max−min) |
   |---|---|---|
   | 16 | 9/38 | λ₂ 2.7e-3, R_G 2.2e-2, p_min 9.2e-1 |
   | 32 | 30/34 | λ₂ 2.9e-3, R_G 3.0e-3, p_min 1.4e-1 |
   | 64 | **42/42** | λ₂ 4.2e-4, R_G 2.1e-5, p_min **0.0** |

   At `n = 64` the α₃ term — weight **0.70**, the dominant reward component —
   has zero spread and cannot distinguish between candidates at all. Ranking is
   decided entirely by Δλ₂ and ΔR_G. This is why correction #3 moved 90% of
   decisions despite ΔR_G carrying a nominal weight of `α₂/n² ≈ 5e-5`. The
   reward weighting deserves a separate review.

---

## 4. Complete pseudocode

```
CONSTANTS
    eps_mult   = 1e-12      zero_proj  = 1e-14      tau_drift  = 1e-6
    eps_sec    = 1e-12      maxvol_eps = 1e-6       max_no_reset = 32
    tol_batch  = 1e-12      maxit_batch = 100       maxvol_iter  = 200
    alpha1, alpha2, alpha3 = 0.10, 0.20, 0.70


# ══════════════════════════════════════════════════════════════════════
PROCEDURE RecomputeExact(state)                       # O(n³), the anchor
# ══════════════════════════════════════════════════════════════════════
    L          ← Laplacian(adj)
    Λ, Q       ← eigh(L)                              # ascending
    r          ← #{ i ≥ 1 : |Λ_i − Λ_1| < eps_mult }
    U          ← Q[:, 1 : 1+r]
    μ          ← Λ[r+1]                if r+1 < n  else RefreshMu(Λ_1)
    invλ       ← [0, 1/Λ_1, …, 1/Λ_{n−1}]             # clamped at 1e-14
    invλ²      ← invλ ⊙ invλ
    L⁺         ← Q[:,1:] · diag(invλ[1:]) · Q[:,1:]ᵀ
    R_G        ← n · Σ invλ
    p          ← 1 − ½ · rowsum( adj ⊙ R_eff_matrix(L⁺) )
    p_min      ← min(p)
    steps_since_exact ← 0 ;  cert_foster ← cert_trace ← 0


# ══════════════════════════════════════════════════════════════════════
FUNCTION RefreshMu(λ₂) → float                        # O(n³), every step
# ══════════════════════════════════════════════════════════════════════
    vals ← sort(eigvalsh(L))                          # CURRENT L
    above ← { v ∈ vals : v > λ₂ + eps_mult }
    RETURN min(above) if above ≠ ∅ else +∞


# ══════════════════════════════════════════════════════════════════════
FUNCTION SecularUpperEndpoint(Λ, λ₂, μ) → float        # correction #2
# ══════════════════════════════════════════════════════════════════════
    # Poles of f sit at the STORED Λ; the bracket may not contain one.
    above ← { Λ_i : Λ_i > λ₂ + eps_mult }
    RETURN μ                     if above = ∅
    RETURN min(μ, min(above))    otherwise


# ══════════════════════════════════════════════════════════════════════
FUNCTION SolveSecularBatch(Ψ, Λ, λ₂, μ) → roots[m]     # correction #1
#   Ψ is (n, m); column c is ψ = Qᵀa_c
# ══════════════════════════════════════════════════════════════════════
    roots ← [λ₂] * m
    IF m = 0 OR μ = ∞:  RETURN roots

    μ    ← SecularUpperEndpoint(Λ, λ₂, μ)
    gap  ← μ − λ₂ ;  IF gap ≤ 0: RETURN roots
    marg ← max(eps_sec, 1e-8 · gap)
    a, b ← λ₂ + marg, μ − marg ;  IF b ≤ a: RETURN roots

    Ψ²   ← Ψ ⊙ Ψ                                      # loop invariant
    f(x) ≜ 1 + colsum( Ψ² / (Λ[:,None] − x) )
    fa, fb ← f(a), f(b)

    bracketed ← finite(fa) ∧ finite(fb) ∧ ¬(fa·fb > 0)
    collided  ← ¬bracketed ∧ (|Ψ[1,:]| ≥ zero_proj) ∧ finite(fb) ∧ (fb < 0)
    roots[collided] ← μ                               # μ-endpoint collision
    # unbracketed, uncollided columns keep λ₂

    idx ← where(bracketed) ;  IF idx = ∅: RETURN roots

    # ---- iterate the POLE-FREE reformulation h(η) = η(1+g) − ψ₂² ----
    # h = η·f, η > 0 ⇒ same signs, same bracketing; h' > 1 > 0 always.
    P      ← Ψ²[:, idx]
    ψ₂²    ← P[1, :]
    P_rest ← P with row 1 zeroed                      # g omits the λ₂ pole
    lo, hi ← a − λ₂, b − λ₂
    η      ← (lo + hi)/2
    act    ← all columns

    REPEAT up to maxit_batch:
        D   ← Λ[:,None] − (λ₂ + η[act])
        T   ← P_rest[:,act] / D
        g   ← colsum(T) ;  g′ ← colsum(T / D)
        h   ← η[act]·(1 + g) − ψ₂²[act]
        h′  ← 1 + g + η[act]·g′

        lo[act] ← where(h < 0, η[act], lo[act])        # h strictly increasing
        hi[act] ← where(h > 0, η[act], hi[act])

        step   ← η[act] − h/h′
        accept ← finite(step) ∧ (lo[act] < step < hi[act])
        η_new  ← where(accept, step, (lo[act] + hi[act])/2)   # bisect fallback

        δ ← |η_new − η[act]| ;  η[act] ← η_new
        act ← act[ δ > tol_batch · max(1, |λ₂ + η_new|) ]     # COMPACT
        IF act = ∅: BREAK
        # NB: terminate on STEP SIZE. Newton closes one-sidedly, so hi−lo
        #     stalls and a bracket-width test would never fire.

    roots[idx] ← λ₂ + η
    RETURN roots


# ══════════════════════════════════════════════════════════════════════
FUNCTION SelectCandidates(state, top = 32) → (pairs, info)
# ══════════════════════════════════════════════════════════════════════
    (i_idx, j_idx) ← all unused non-edges              # k = O(n²)
    IF k = 0: RETURN empty

    # ---------- STAGE 1: screen all k non-edges, O(n²) ----------
    R_e ← L⁺[i,i] + L⁺[j,j] − 2·L⁺[i,j]                # O(1) per non-edge
    F   ← ‖U[i,:] − U[j,:]‖²                           # O(r) per non-edge
    I   ← unique( top32(R_e) ∪ top32(F) )[:64]         # stable mergesort
    m   ← |I| ;  IF m = 0: RETURN empty

    Ψ   ← ( Q[i_idx[I], :] − Q[j_idx[I], :] )ᵀ         # (n, m) — survivors only

    # ---------- STAGE 2 ----------
    IF r = 1:                                          # ── CASE A ──
        P2   ← Ψ ⊙ Ψ

        # ΔR_G in the eigenbasis (correction #3).
        # β = 1 + Σψᵢ²/λᵢ = 1 + aᵀL⁺a.  Using L⁺Ψ here would compute
        # ψᵀL⁺ψ — treating the projection as the incidence vector.
        β        ← 1 + invλ · P2
        ΔR_G_red ← n · (invλ² · P2) / max(β, 1e-14)

        # Δλ₂ — one batched solve; guarded columns stay at 0.
        active ← where( |Ψ[1,:]| ≥ zero_proj )
        Δλ₂    ← zeros(m)
        Δλ₂[active] ← SolveSecularBatch(Ψ[:,active], Λ, λ₂, μ) − λ₂

        # Δp_min in closed form: p is maintained, L⁺ held fixed, so
        # pᵢ^new = pᵢ − ½R_eff(i,j) and R_eff is already R_e.
        pe    ← min( p[i_idx[I]], p[j_idx[I]] ) − ½·R_e[I]
        Δp    ← min(pe, p_min) − p_min                 # ≤ 0 identically

        reward ← alpha1·Δλ₂/n + alpha2·η_R·ΔR_G_red/n² + alpha3·η_p·Δp
        order  ← argsort(−reward, stable)
        RETURN pairs[ order[:min(top, m)] ], diagnostics

    ELSE:                                              # ── CASE B ──
        Z      ← Uᵀ Ψ                                  # (r, m)
        J      ← columns with ‖Z[:,j]‖ ≥ τ,  τ = pctl(75 → 50 → 25)
        IF |J| < r: sel ← J
        ELSE:
            Q_R, _ ← qr(Z[:,J]ᵀ)                       # (|J|, r) orthonormal
            sel    ← J[ MaxVol(Q_R, maxvol_eps, maxvol_iter) ]
            IF |sel| > 1: sel ← sel[:−1]               # discard one edge → r−1
        RETURN pairs[ sel[:min(top, |sel|)] ], diagnostics


# ══════════════════════════════════════════════════════════════════════
PROCEDURE AddEdge(state, i, j)
# ══════════════════════════════════════════════════════════════════════
    ASSERT i ≠ j  AND  adj[i,j] = 0
    λ₂_old, p_min_old ← Λ[1], p_min

    a ← e_i − e_j
    ψ ← Qᵀa                                            # on the CURRENT basis

    # ---- exact ΔR_G via Sherman–Morrison (a IS the incidence vector) ----
    v ← L⁺a ;  β ← 1 + aᵀv
    IF ¬finite(β) OR β ≤ 1e-14:  RecomputeExact() ; RETURN
    R_G ← R_G − n·(vᵀv)/β

    adj[i,j] ← adj[j,i] ← 1

    # ---- rank-1 update.  v is mean-centred: L⁺1 = 0 ⇒ 1ᵀ(L⁺a) = 0, so
    #      the outer product is already centred and (I−J)L⁺(I−J) is
    #      roundoff hygiene costing ~2n³ flops for nothing.
    v ← v − mean(v)
    L⁺ ← L⁺ − v vᵀ / β
    L  ← L + a aᵀ

    # ---- refresh p BEFORE the drift check (Foster reads p) ----
    p ← 1 − ½·rowsum( adj ⊙ R_eff_matrix(L⁺) ) ;  p_min ← min(p)

    IF r = 1:                                          # ── CASE A ──
        λ₂′ ← SolveSecular(ψ, Λ, λ₂_old, μ)            # scalar, full precision
        CheckSecularBracket(λ₂′, λ₂_old, ψ[1])         # diagnostic only

        # Compare against the endpoint the SOLVER used, not μ (§2.10).
        endpt ← SecularUpperEndpoint(Λ, λ₂_old, μ)
        collided ← λ₂′ > λ₂_old AND
                   ( |λ₂′ − endpt| < 1e-10 OR |λ₂′ − μ| < 1e-10 )

        IF collided:
            RecomputeExact()                           # endpoint is a POLE of f
        ELIF λ₂′ > λ₂_old:
            u₂ ← normalise( Σᵢ ψᵢ/(Λᵢ − λ₂′) · qᵢ )
            Q[:,1] ← u₂ ;  Λ[1] ← λ₂′
            DriftCheck(u₂, λ₂′)
        ELSE:
            DriftCheck(Q[:,1], Λ[1])                   # still tick the counter
    ELSE:                                              # ── CASE B ──
        H ← UᵀLU ;  Θ, V ← eigh(H) ;  U ← UV
        DriftCheck(U[:,0], Λ[1])

    μ ← RefreshMu(Λ[1])                                # dense eigvalsh, ALWAYS
    record Δλ₂, ΔR_G_red, Δp_min for the caller


# ══════════════════════════════════════════════════════════════════════
PROCEDURE DriftCheck(u₂, λ₂)
# ══════════════════════════════════════════════════════════════════════
    steps_since_exact ← steps_since_exact + 1
    residual    ← ‖L u₂ − λ₂ u₂‖                       # only Fiedler-aware check
    cert_foster ← |Σᵢ pᵢ − 1|                          # exact invariant (Foster)
    cert_trace  ← |R_G − n·tr(L⁺)| / |R_G|             # exact invariant
    drift ← max(residual, cert_foster, cert_trace)
    IF drift > tau_drift OR steps_since_exact ≥ max_no_reset:
        RecomputeExact()
```

### Complexity per environment step

| Phase | Cost | Notes |
|---|---|---|
| Stage 1 screening | `O(n²)` | now ~69% of `select_candidates` at n=128 |
| Stage 2, Case A | `O(n·m·it)`, `m ≤ 64`, `it ≈ 5–12` | was `O(n·m·12.6)` interpreted |
| `add_edge` rank-1 | `O(n²)` | Sherman–Morrison |
| `RefreshMu` | `O(n³)` | **every step** |
| `RecomputeExact` | `O(n³)` | **~every other step** (§1.6) |

The two `O(n³)` rows are the ceiling; everything above them is now cheap
relative to them.
