# Analytic Edge-Addition Algorithm — Implementation Notes

This document describes how the **Complete Edge Addition Algorithm** in
[`algorithm.md`](analytic_tractable_test_deepseek/algorithm.md) is implemented in this
package, and how it replaces the edge-selection step inside the RL loop.

The specification is implemented from scratch in
[`analytic_state.py`](analytic_tractable_test_deepseek/analytic_state.py) and wired into
the environment in [`env.py`](analytic_tractable_test_deepseek/env.py). All numerical
choices were fixed by the task's Q&A; those decisions are tabulated at the end.

- **Part 1** is a high-level, readable overview — what the algorithm does and why.
- **Part 2** is an explicit, step-by-step iteration through every stage, with the exact
  formulas, mapped to the code that implements them.

---

## Part 1 — Step-by-step overview

### The problem

We grow a graph one edge at a time. At every step we have a graph $G=(V,E)$ with
$|V|=n$ and its Laplacian $L=D-A$. We care about three quantities that improve as edges
are added:

1. **Algebraic connectivity** $\lambda_2$ — the smallest non-zero eigenvalue of $L$
   (how "well-connected" the graph is);
2. **Total effective resistance** $R_G = n\,\mathrm{tr}(L^+)$ — the sum of resistance
   across the network (lower is better);
3. **Minimum resistance curvature** $p_{\min}$ — a per-node robustness measure.

The algorithm's job is to **pick the best edge to add next** in an analytically principled
way, and to **keep an exact spectral state** (eigenvalues, eigenvectors, pseudoinverse)
updated cheaply as edges are added.

### The big picture

```mermaid
flowchart LR
    A[Initial graph] --> B[Full eigendecomposition once]
    B --> C[Maintain exact spectral state: Q, evals, L⁺, R_G, p_min, r, μ]
    C --> D[Stage 1: score ALL non-edges]
    D --> E[Keep ≤64 candidates]
    E --> F[Stage 2: narrow to ≤32 by reward / MaxVol]
    F --> G[Policy picks one of the 32]
    G --> H[Add that edge: exact rank-1 update]
    H --> I[Drift check: eigenpair residual + Foster + trace<br/>→ full recompute if any trips]
    I --> D
```

### The per-step loop (in the RL environment)

For every environment step (one edge added per step):

1. **Build the candidate set** — the analytic algorithm screens every unused non-edge and
   produces at most 32 candidates for the policy.
2. **Let the policy choose** — the retained RL policy picks one of those 32 edges.
3. **Apply the edge** — the chosen edge is added through the analytic state, which updates
   $\lambda_2$, $R_G$, $p_{\min}$ and the eigen-basis **exactly** (closed-form rank-1
   formulas), with a full recomputation only when numerical drift accumulates.

### What each step actually costs

The whole design rests on one trade. Updating the state after an edge is added is
$\mathcal O(n^2)$ work — a couple of matrix-vector products and an outer product.
Recomputing it from scratch is a dense eigendecomposition, $\mathcal O(n^3)$. So the
per-step cost is dominated not by the update but by **how often the fallback fires**.
At the current settings that is about half of all edge additions, which is what keeps
$\lambda_2$ pinned to machine precision.

Three things used to make that trade worse than it needed to be, and all three have
been fixed. None of them changed a single number the algorithm produces:

- **Hunting for $\mu$ with the wrong tool.** $\mu$ is the first eigenvalue above
  $\lambda_2$, and it has to be refreshed after every edge because it sets the bracket
  the secular equation is solved inside. This used to call ARPACK — an *iterative*
  eigensolver built for large sparse matrices, where the win comes from never touching
  most of the matrix. At these sizes there is nothing to save: it ran ~87 iterations per
  edge on a 32×32 matrix, cost about six times the dense solve it was avoiding, and
  accounted for roughly 78% of all time spent adding an edge. It now just does the dense
  solve, which is both faster and exact.

- **Re-centering a whole matrix to fix a rounding error.** After the rank-1 update the
  pseudoinverse has to stay orthogonal to the all-ones vector. The spec writes that as
  sandwiching $L^+$ between two projection matrices, which is two dense $n \times n$
  multiplications — about $2n^3$ arithmetic operations every step. But the vector being
  used for the update is already orthogonal to all-ones (see §4), so the sandwich only
  ever removes accumulated rounding noise. Subtracting the mean off that one vector,
  $\mathcal O(n)$ work, does the same job.

- **Computing the curvature vector one neighbour at a time.** $p_{\min}$ needs the
  resistance curvature at every node, which was a Python loop over nodes and then over
  each node's neighbours — interpreted, one scalar at a time. The same quantity is one
  vectorised expression over the whole matrix.

Together these took the cost per edge from roughly 22× a plain eigendecomposition at
$n=12$ down to about 7×, and from 3.5× down to **below 1×** at $n=128$ — the analytic
path is now cheaper than recomputing from scratch, which is what it was supposed to be
all along.

### How we know the state hasn't gone stale

Incremental updates accumulate error, so something has to notice when the tracked state
has drifted away from the true spectrum. Three signals run every step, and any one of
them triggers a full recomputation:

1. **The eigenpair residual**, $\lVert L u_2 - \lambda_2 u_2 \rVert$. If the tracked
   Fiedler pair really is an eigenpair this is zero, so its size is a direct measure of
   how far $\lambda_2$ and its eigenvector have wandered. It is one matrix-vector
   product — measured at 0.1–1.5% of a dense eigendecomposition — so it is essentially
   free, and it is the **only** one of the three that can see $\lambda_2$ drift at all.
2. **Foster's identity.** The resistance curvatures always sum to exactly 1, for every
   connected graph, as a theorem. So $\lvert \sum_i p_i - 1 \rvert$ is a drift measure
   that costs nothing to check — $p$ is already computed each step for the reward.
3. **The trace identity**, $R_G = n\,\mathrm{tr}(L^+)$. $R_G$ is carried as a running
   total while $L^+$ is carried as a matrix; if the two stop agreeing, the running total
   has drifted.

These are complementary, not interchangeable. The residual watches the eigenpair; the
other two watch the pseudoinverse and the $R_G$ accumulator, which the residual cannot
see. An earlier revision of this work replaced the residual with the other two on the
theory that they were cheaper — $\lambda_2$ silently drifted to $8\times10^{-2}$ within
three edges, because neither identity involves the Fiedler vector at all. In practice,
on clean runs, the residual is the signal that ever fires; Foster and the trace identity
have not tripped in testing and are carried as insurance.

Separately, every secular root is now **certified** against a closed-form interval it
must lie in (§4). That check is diagnostic — it records violations rather than acting on
them — because measurement showed re-anchoring on it changes no digit of the answer.

### Stage 1 — screening (score all non-edges)

For every unused non-edge $(i,j)$, with incidence vector $a = e_i - e_j$ and its projection
$\psi = Q^{\mathsf T}a$ onto the current eigenbasis $Q$:

- **Effective resistance** $R_e = a^{\mathsf T} L^+ a = \sum_{i=2}^{n} \frac{\psi_i^2}{\lambda_i}$
  — how "hard" it is to connect the two nodes.
- **Multispectral Fiedler score** $F = \sum_{\ell=1}^{r} \psi_\ell^2$ — how much the edge
  moves the $\lambda_2$ eigenspace (of multiplicity $r$).

Take the **top 32 by $R_e$** and the **top 32 by $F$**, union them, and truncate to at most
**64 candidates**.

> **Cost note:** $R_e$ is read directly from the maintained pseudoinverse
> ($R_e = L^+_{ii} + L^+_{jj} - 2L^+_{ij}$, $\mathcal{O}(1)$ per edge) and $F$ only needs
> the $\lambda_2$ eigenspace $U$ ($\mathcal{O}(r)$ per edge). The full $n$-dimensional
> projection $\psi$ is computed **only for the ≤64 survivors**. This makes the whole
> screening $\mathcal{O}(n^2)$ instead of $\mathcal{O}(n^3)$ (see the scale check below).

### Stage 2 — second-stage selection

**Case A — $\lambda_2$ is simple ($r = 1$).** For each of the ≤64 candidates, compute a
closed-form estimate of what happens if we add it:

- $\Delta \lambda_2$: solve the **secular equation**
  $f(\lambda) = 1 + \sum_{i=2}^n \frac{\psi_i^2}{\lambda_i - \lambda} = 0$ on
  $(\lambda_2, \mu)$ (Brent's method). The root is the new $\lambda_2'$. If
  $|\psi_2| < 10^{-14}$ the Fiedler vector doesn't move, so $\Delta\lambda_2 = 0$.
- $\Delta R_G$: exact positive reduction $R_G^{old} - R_G^{new} = n\,\frac{v^{\mathsf T}v}{\beta}$
  with $v = L^+a$, $\beta = 1 + a^{\mathsf T}L^+ a$ (Sherman–Morrison).
- $\Delta p_{\min}$: a **cheap first-order** estimate (see Part 2).

Combine them into a reward
$$R_c = \alpha_1 \frac{\Delta\lambda_2}{n} + \alpha_2\, \eta_R \frac{\Delta R_G}{n^2} +
\alpha_3\, \eta_p\, \Delta p_{\min}$$
and keep the **top 32** candidates by $R_c$ for the policy.

**Case B — $\lambda_2$ has multiplicity $r > 1$.** The Fiedler eigenvalue is degenerate, so
a single edge won't "unlock" $\lambda_2$ by itself. The algorithm:

1. projects the candidates onto the $\lambda_2$ eigenspace $Z_{mult} = U^{\mathsf T}B'$;
2. filters to the highest-norm columns (75th/50th/25th percentile);
3. runs **MaxVol** — a deterministic matrix selection that picks $r$ edges whose
   projections nearly span the eigenspace;
4. discards one edge, leaving $r-1$ edges that resolve the multiplicity;
5. offers those $r-1$ edges as the candidate set; the policy picks one, and the edge is
   applied with a **Rayleigh–Ritz** eigenspace refinement.

### Applying an edge — the exact update

Adding edge $(i,j)$ is a **rank-1** perturbation $L \leftarrow L + aa^{\mathsf T}$:

- **Pseudoinverse:** Sherman–Morrison
  $L^+ \leftarrow L^+ - \frac{(L^+a)(L^+a)^{\mathsf T}}{1 + a^{\mathsf T}L^+ a}$, followed
  by a centering projection to keep it on the space orthogonal to $\mathbf 1$.
- **$R_G$:** exact trace update $R_G \leftarrow R_G - n\frac{(L^+a)^{\mathsf T}(L^+a)}{\beta}$.
- **$\lambda_2$ and its vector:**
  - *Case A:* exact secular eigenvector formula
    $u_2' = \sum_{i=2}^{n} \frac{\psi_i}{\lambda_i - \lambda_2'} q_i$, normalised;
  - *Case B:* Rayleigh–Ritz refinement of the eigenspace $U$.
- **$p_{\min}$:** exact recomputation from the updated $L^+$.
- **$\mu$** (the first eigenvalue above $\lambda_2$): refreshed with a few Lanczos
  (ARPACK `eigsh`) iterations so the secular bracket stays correct.
- **Drift:** if $\|L u_2 - \lambda_2 u_2\| > \tau_{drift}$ or 32 steps have passed, do a full
  eigendecomposition to reset the basis.

### Termination

An episode stops when the edge budget (target edge count $m_{target}$) is reached.
The final graph, $\lambda_2$, $R_G$ and $p_{\min}$ are returned.

### Calibration

The scalars $\eta_R$ and $\eta_p$ are calibrated empirically so the three reward terms have
comparable magnitude on average over the construction regime (see
[`calibrate_eta.py`](analytic_tractable_test_deepseek/calibrate_eta.py) →
[`calibration.py`](analytic_tractable_test_deepseek/calibration.py)):

$$\eta_R = \frac{\operatorname{mean}\left|\Delta\lambda_2/n\right|}
{\operatorname{mean}\left|\Delta R_G/n^2\right|},
\qquad
\eta_p = \frac{\operatorname{mean}\left|\Delta\lambda_2/n\right|}
{\operatorname{mean}\left|\Delta p_{\min}\right|}.$$

They are interpolated by graph size at runtime and can be overridden by config.

---

## Part 2 — Explicit step-by-step iteration

This part walks through every stage of the algorithm exactly as implemented, with formulas
and code references. Numbers in brackets like `(A:3)` refer to `algorithm.md` sections.

### 0. Preliminaries and notation

| Symbol | Meaning | Implementation |
|---|---|---|
| $G=(V,E)$, $n$ | graph, node count | `self.n`, `self.adj` |
| $L = D - A$ | Laplacian | [`graph_math.laplacian()`](analytic_tractable_test_deepseek/graph_math.py:50) |
| $0=\lambda_1<\lambda_2\le\dots\le\lambda_n$ | eigenvalues | `self.evals` |
| $q_i$ (columns of $Q$) | eigenvectors | `self.Q` |
| $a = e_i - e_j$ | incidence vector of a non-edge | built in `_rank1_update` / `select_candidates` |
| $\psi = Q^{\mathsf T}a$ | projection of $a$ | `(Q[i] - Q[j]).T` |
| $R_G = n\,\mathrm{tr}(L^+)$ | total effective resistance | `self.R_G` |
| $r$ | multiplicity of $\lambda_2$ | `self.r` |
| $U \in \mathbb R^{n\times r}$ | $\lambda_2$ eigenspace basis | `self.U` |
| $\mu$ | first eigenvalue $> \lambda_2$ | `self.mu` |
| $\mathcal E_{non}$ | the set of unused non-edges | `non_edges(adj)` |

**Reward heuristic** (`algorithm.md` Preliminaries):
$$R = \alpha_1 \frac{\Delta\lambda_2}{n} + \alpha_2\, \eta_R \frac{\Delta R_G}{n^2}
+ \alpha_3\, \eta_p\, \Delta p_{\min},$$
with the confirmed weights $\alpha_1 = 0.10$, $\alpha_2 = 0.20$, $\alpha_3 = 0.70$
(convex sum; `alpha3 = 1 - alpha1 - alpha2`).

---

### 1. Initialisation (`algorithm.md` §2)

Implemented in [`AnalyticState.__init__()`](analytic_tractable_test_deepseek/analytic_state.py:405)
→ [`_recompute_exact()`](analytic_tractable_test_deepseek/analytic_state.py:503).

1. **Build $L$** from the initial adjacency.
2. **Full eigendecomposition** (once): $L = Q\Lambda Q^{\mathsf T}$ via
   `np.linalg.eigh`, storing all eigenvalues and eigenvectors.
3. **Multiplicity** of $\lambda_2$:
   $$r = \#\left\{\lambda_i : |\lambda_i - \lambda_2| < \varepsilon_{mult}\right\},
   \qquad \varepsilon_{mult} = 10^{-12},$$
   excluding the trivial $\lambda_1 = 0$. `U` = the columns of `Q` in this cluster.
   Set $\mu = \lambda_{r+1}$ (first eigenvalue strictly above $\lambda_2$).
4. **Pseudoinverse / auxiliary vectors:**
   $L^+ = \sum_{i=2}^n \frac{1}{\lambda_i}q_iq_i^{\mathsf T}$,
   $\mathbf{inv} = [0, 1/\lambda_2, \dots, 1/\lambda_n]$,
   $\mathbf{inv2} = [0, 1/\lambda_2^2, \dots, 1/\lambda_n^2]$.
5. **Non-edge matrix** $B$ (columns are incidence vectors) and projection
   $Z_{full} = Q^{\mathsf T}B$. *(In this implementation the projection is recomputed only
   for the surviving candidates rather than stored for all non-edges — see the storage
   note at the end; the mathematics is identical.)*
6. **Initial metrics:** $R_G = n\sum_{i=2}^n 1/\lambda_i$, and $p_{\min}$ from the exact
   curvature formula.
7. **Drift control:** tolerance $\tau_{drift} = 10^{-6}$, maximum 32 incremental steps
   between forced full recomputations.

---

### 2. Stage 1 — candidate screening (`algorithm.md` §3.1)

Implemented in [`select_candidates()`](analytic_tractable_test_deepseek/analytic_state.py:579).

For every unused non-edge $j$ with endpoints $(i,j)$ and projection $\psi^{(j)} = Q^{\mathsf T}a^{(j)}$:

- **Effective resistance**
  $$R_e[j] = \sum_{i=2}^{n}\frac{\left(\psi_i^{(j)}\right)^2}{\lambda_i}
  = \mathbf{inv}\cdot\left(\psi^{(j)}\odot\psi^{(j)}\right)
  = L^+_{ii} + L^+_{jj} - 2L^+_{ij}.$$
  *(Implemented as the last expression — an $\mathcal{O}(1)$ lookup in the maintained
  `L_plus`, mathematically identical to the projection formula.)*
- **Multispectral Fiedler score**
  $$F[j] = \sum_{\ell=1}^{r}\left(\psi_\ell^{(j)}\right)^2
  = \sum_{\ell=1}^{r}\left(U_\ell[i] - U_\ell[j]\right)^2.$$

Then:

- $\mathcal I_R$ = indices of the **top 32** by $R_e$ (descending);
- $\mathcal I_F$ = indices of the **top 32** by $F$;
- $\mathcal I_{cand} = \mathrm{unique}(\mathcal I_R \cup \mathcal I_F)$, truncated to at
  most 64; if fewer than 64 non-edges remain, keep all.
  (Because each top-32 set has ≤32 elements, the union has ≤64 automatically, so the cap is
  a safety bound.)

**Complexity note (the optimisation):** Stage 1 never forms the full $n\times k$
projection. $R_e$ is read from $L^+$ in $\mathcal O(1)$ per edge and $F$ from $U$ in
$\mathcal O(r)$ per edge, making the screening $\mathcal O(k\,(1+r)) = \mathcal O(n^2)$.
The full $\psi$ is formed **only for the ≤64 candidates** as
`psi = (Q[i_idx[i_cand]] - Q[j_idx[i_cand]]).T`. The scale check in
[`benchmark_screening.py`](analytic_tractable_test_deepseek/benchmark_screening.py) shows
this is ~59× faster than the $\mathcal O(n\cdot k)$ projection pass at $n=128$, with an
empirical log-log slope of 1.66 (vs 3.12 for the old pass) — i.e. $\mathcal O(n^2)$ vs
$\mathcal O(n^3)$.

---

### 3. Stage 2 — second-stage selection (`algorithm.md` §3.2)

#### Case A — $r = 1$ (simple $\lambda_2$)

For each candidate $c$ in $\mathcal I_{cand}$:

1. **Extract** $a = B'[:,c]$ and $\psi = Z'_{full}[:,c]$.
2. **Zero-projection guard:** if $|\psi_2| < 10^{-14}$, set $\Delta\lambda_2 = 0$ and skip
   the secular solve. Otherwise:
   - $R_e = \sum_{i=2}^n \psi_i^2/\lambda_i$;
   - $\Delta R_G = -n\,\dfrac{\sum_{i=2}^n \psi_i^2/\lambda_i^2}{1 + \sum_{i=2}^n \psi_i^2/\lambda_i}$
     (implemented via $v = L^+a$, $\beta = 1 + a^{\mathsf T}L^+a$, as the **positive
     reduction** $n\,v^{\mathsf T}v/\beta$);
   - solve the **secular equation**
     $$f(\lambda) = 1 + \sum_{i=2}^n \frac{\psi_i^2}{\lambda_i - \lambda} = 0$$
     on $(\lambda_2 + \varepsilon, \mu)$ with $\varepsilon = 10^{-12}$, using **Brent's
     method** (bisection fallback); the root is $\lambda_2'$, and
     $\Delta\lambda_2 = \lambda_2' - \lambda_2$. The bracket margin is
     $\max(10^{-12},\,10^{-8}\cdot(\mu-\lambda_2))$.
3. **Compute $\Delta p_{\min}$** with a **cheap first-order** approximation: only the new
   resistance $R_e(i,j)$ is added to $p_i$ and $p_j$ (using the current $L^+$), then
   $\Delta p_{\min} \approx \min(p_i^{new}, p_j^{new}, p_{\min}) - p_{\min}$.
   *(This is used only for candidate scoring; after the edge is actually added, $p_{\min}$
   is recomputed exactly.)*
4. **Reward:**
   $$R_c = \alpha_1\frac{\Delta\lambda_2}{n} + \alpha_2\,\eta_R\frac{\Delta R_G}{n^2}
   + \alpha_3\,\eta_p\,\Delta p_{\min}.$$

Then rank the candidates by $R_c$ and keep the **top 32** as the candidate set returned to
the policy. The policy picks one; that edge is then applied (Section 5).

> **Boundary case (spec gap, handled):** if the new $\lambda_2$ collides *exactly* with the
> old $\mu$ (typically when the chord is orthogonal to the third eigenvector, $\psi_3\approx0$),
> the open bracket $(\lambda_2,\mu)$ contains no root. In that case the true root is
> $\lambda_2' = \mu$ exactly, and the new Fiedler vector is the old $q_3$. This is detected
> (f stays negative up to $\mu$ with $\psi_2\ne0$) and handled — otherwise $\Delta\lambda_2$
> would be silently under-reported (see Section 6).

#### Case B — $r > 1$ (multiple $\lambda_2$)

1. **Projected matrix:** $Z_{mult} = U^{\mathsf T}B' \in \mathbb R^{r\times|\mathcal I_{cand}|}$.
2. **Filter high-potential columns:** column norms $n_j = \|Z_{mult}[:,j]\|_2$; $\tau$ =
   the 75th percentile; keep columns with $n_j\ge\tau$. If fewer than $r$ survive, lower
   $\tau$ to the 50th then 25th percentile until at least $r$ columns remain. Keep
   $Z_{keep}$.
3. **MaxVol selection** (deterministic, implemented in
   [`maxvol_selection()`](analytic_tractable_test_deepseek/analytic_state.py:322)):
   - thin QR $Z_{keep}^{\mathsf T} = Q_R R$, $Q_R\in\mathbb R^{m\times r}$ orthonormal;
   - LU with row pivoting $P Q_R = LU$ gives an initial active set
     $I_{init} = \{p[0],\dots,p[r-1]\}$;
   - iterate: $C = Q_R(Q_R[I,:])^{-1}$; while $\max|C| > 1 + 10^{-6}$, swap the
     off-set row with the largest $|C|$ into $I$ (max 200 iterations);
   - the result is a set of $r$ edges $\mathcal E_r$.
4. **Choose $r-1$ edges:** discard one edge from $\mathcal E_r$ (the last), giving the
   addition list $\mathcal E_{add}$.
5. **Sequential addition:** each of the $r-1$ edges is added one per environment step
   (per the confirmed "distribute across steps" decision). Before each addition, compute
   the exact $\Delta R_G$ for that single edge (via $v = L^+a$), update $R_G$, apply the
   rank-1 update, and run the **Rayleigh–Ritz** refinement (Section 5). If the refinement
   reveals a simple $\theta_{\min}$, set $r=1$, $\lambda_2 = \theta_{\min}$,
   $u_2$ = that Ritz vector, and switch to the Case-A path.
6. **Post-multiplicity adjustments:** refresh $\mu$ with a few Lanczos steps, and refresh
   the projection rows for the (possibly changed) eigenspace.

---

### 4. The single-edge update (`algorithm.md` §3.3)

Implemented in [`AnalyticState.add_edge()`](analytic_tractable_test_deepseek/analytic_state.py:871),
with the rank-1 core in [`_rank1_update()`](analytic_tractable_test_deepseek/analytic_state.py:725).

Given the chosen edge with incidence vector $a$ and projection $\psi$ (computed from the
basis *before* the update):

1. $\beta = 1 + a^{\mathsf T}L^+ a$.
2. **Exact $\Delta R_G$ (positive reduction):**
   $\Delta R_G = n\,\dfrac{a^{\mathsf T}(L^+)^2 a}{\beta} = n\,\dfrac{v^{\mathsf T}v}{\beta}$,
   $v = L^+a$; update $R_G \leftarrow R_G - \Delta R_G$.
3. **Rank-1 Laplacian / pseudoinverse update:**
   $$L \leftarrow L + aa^{\mathsf T},
   \qquad v \leftarrow v - \bar v\mathbf 1,
   \qquad L^+ \leftarrow L^+ - \frac{vv^{\mathsf T}}{\beta}.$$
   The spec writes the centering as $L^+ \leftarrow (I - \tfrac1n J)L^+(I - \tfrac1n J)$
   with $J = \mathbf 1\mathbf 1^{\mathsf T}$, i.e. two dense $n\times n$ products
   ($\approx 2n^3$ flops) after every edge. That is unnecessary: $L^+\mathbf 1 = 0$, so
   $$\mathbf 1^{\mathsf T}v = \mathbf 1^{\mathsf T}L^+a = (L^+\mathbf 1)^{\mathsf T}a = 0,$$
   i.e. $v$ has zero sum *exactly*, hence so does $vv^{\mathsf T}$ and so does the
   downdated $L^+$. The sandwich can only ever remove accumulated roundoff, and
   subtracting the mean off $v$ — $\mathcal O(n)$ — removes the same roundoff.
   [`_rank1_update()`](analytic_tractable_test_deepseek/analytic_state.py:725).
4. **Fiedler vector update (exact secular formula, Case A):**
   $$u_2' = \sum_{i=2}^{n}\frac{\psi_i}{\lambda_i - \lambda_2'}\,q_i, \qquad
   u_2' \leftarrow u_2'\,/\,\|u_2'\|_2.$$
   If $\lambda_2' = \mu$ (boundary collision), set $u_2' = q_3$ instead (the secular formula
   divides by zero there). Replace $Q[:,1] = u_2'$ and $\lambda_2 \leftarrow \lambda_2'$.
   *(Case B uses the Rayleigh–Ritz path instead of this formula.)*
5. **Certify the root (diagnostic).** With $\delta' = \mu - \lambda_2$ the gap above
   $\lambda_2$ before the edge, the exact root must satisfy
   $$\frac{\delta'}{\delta' + 2}\,\psi_2^2 \;\le\; \Delta\lambda_2 \;\le\; \psi_2^2 .$$
   Both bounds are $\mathcal O(1)$ — $\psi$ is already formed for the secular solve —
   and the inequality was checked against exact eigendecompositions on 90 random
   (graph, edge) pairs with zero violations. This does **not** replace the root; it
   certifies it, which is what would catch a bad $\mu$, a stale basis, or a Brent
   convergence failure. It records violations rather than acting on them — see
   deviation 10. [`_check_secular_bracket()`](analytic_tractable_test_deepseek/analytic_state.py:806).
6. **Refresh $\mu$:** dense `eigvalsh`, taking the first eigenvalue $> \lambda_2' +
   \varepsilon_{mult}$, or $\infty$ if there is none.
   [`_refresh_mu()`](analytic_tractable_test_deepseek/analytic_state.py:488).
   The spec says "a few Lanczos steps" and the implementation used ARPACK `eigsh`
   accordingly. That is the wrong tool at these sizes — an iterative solver's advantage
   is not touching most of a large sparse matrix, and here there is nothing to skip. It
   was measured at ~87 Lanczos iterations per edge on a $32\times32$ matrix, roughly
   $6\times$ the dense solve, and ~78% of all time inside `add_edge`. The dense path is
   faster *and* exact, which matters because $\mu$ is not purely a bracket endpoint in
   this implementation (see deviations 1 and 8).
7. **Drift monitoring** — three signals, any of which triggers a recomputation:
   $$res = \|L u_2' - \lambda_2' u_2'\|_2, \qquad
   c_{\mathrm{Foster}} = \Bigl|\sum_i p_i - 1\Bigr|, \qquad
   c_{\mathrm{trace}} = \frac{\bigl|R_G - n\,\mathrm{tr}(L^+)\bigr|}{|R_G|}.$$
   If $\max(res, c_{\mathrm{Foster}}, c_{\mathrm{trace}}) > \tau_{drift}$ **or** 32 steps
   have passed since the last exact reset, run a **full recomputation** (fresh `eigh` →
   new $Q$, $\{\lambda_i\}$, $L^+$, $r$, $\mu$, $U$, $p$, $p_{\min}$) and reset the step
   counter. [`_drift_check()`](analytic_tractable_test_deepseek/analytic_state.py:806),
   [`_cheap_drift()`](analytic_tractable_test_deepseek/analytic_state.py:780).

   The residual is $\mathcal O(n^2)$ — one matvec, measured at 0.1–1.5% of a dense `eigh`
   for $n = 32\ldots256$ — and it is the only one of the three that observes the Fiedler
   pair. Foster's identity ($\sum_{(i,j)\in E} R_{\text{eff}}(i,j) = n-1$ gives
   $\sum_i p_i = 1$ identically) and the trace identity are exact invariants covering
   $L^+$ and the $R_G$ accumulator, which the residual cannot see. They are
   complementary, not substitutes — see deviation 9.

   *(The no-raise case also routes through drift monitoring so the periodic reset fires even
   during long streaks of edges that don't move $\lambda_2$.)*
8. **Refresh $p$ and $p_{\min}$:** $p_i = 1 - \tfrac12\sum_{j\sim i} R_{\text{eff}}(i,j)$
   with $R_{\text{eff}}(i,j) = L^+_{ii} + L^+_{jj} - 2L^+_{ij}$, computed as one masked
   outer-sum over the whole matrix rather than a Python loop over nodes × neighbours.
   The full vector is kept (not just its minimum) because Foster's certificate in step 7
   needs $\sum_i p_i$ — it was already being built and discarded.
   [`compute_p()`](analytic_tractable_test_deepseek/analytic_state.py:102). This happens
   *before* step 7, so the certificate reads a $p$ that matches the post-edge graph.

---

### 5. Case B refinement — Rayleigh–Ritz

Implemented in [`_rayleigh_ritz_update()`](analytic_tractable_test_deepseek/analytic_state.py:753).
After the rank-1 update in the $r>1$ regime:

1. Form the $r\times r$ projected matrix $H = U^{\mathsf T} L U$.
2. Solve $H V = V\Theta$; the new eigenspace is $U_{new} = U V$.
3. Let $\theta_{\min}$ be the smallest eigenvalue in $\Theta$.
   - If $\theta_{\min}$ is **simple**: set $r = 1$, $\lambda_2 = \theta_{\min}$,
     $u_2$ = the corresponding column of $U_{new}$ (no secular formula — the Ritz vector is
     used directly), and break out of the multiplicity path.
   - Otherwise: set $r = r_{new}$, $U = U_{new}$ (all $r$ vectors), and continue.

---

### 6. Deviations, edge cases, and design choices (flagged, not hidden)

1. **μ-boundary collision (spec gap).** When a chord's new $\lambda_2$ equals the old $\mu$
   exactly (e.g. edge $(1,8)$ on a 10-path, where $\psi_3 \approx 0$), the open bracket
   $(\lambda_2, \mu)$ has no root. The literal spec returns $\Delta\lambda_2 = 0$; this
   implementation detects the collision and sets $\lambda_2' = \mu$ with $u_2' = q_3$, so
   $\lambda_2$ is tracked exactly. *(The pre-existing debug file was investigating exactly
   this edge.)*
2. **Zero-projection guard.** If $|\psi_2| < 10^{-14}$, the Fiedler vector does not move:
   $\Delta\lambda_2 = 0$ and the secular solve is skipped, but $\Delta R_G$ is still
   computed normally (it uses the full $\psi$, not just $\psi_2$).
3. **Sign of $\Delta R_G$ in the reward.** The spec's $\Delta R_G$ formula is negative
   (resistance *reduction*); per the Q&A the reward uses the **positive reduction**
   $R_G^{old} - R_G^{new}$.
4. **$Z_{full}$ storage.** The spec stores $Z_{full} = Q^{\mathsf T}B$ and updates rows
   incrementally. Here $\psi = Q^{\mathsf T}a$ is recomputed on demand for the ≤64
   candidates; the mathematics is identical and it avoids maintaining a stale matrix.
5. **Screening complexity.** Stage 1 is $\mathcal O(n^2)$ (see Section 2) — $R_e$ from
   $L^+$, $F$ from $U$, full $\psi$ only for the 64 survivors. Confirmed by
   [`benchmark_screening.py`](analytic_tractable_test_deepseek/benchmark_screening.py).
6. **No-raise drift.** Edges that don't raise $\lambda_2$ still increment the drift counter
   (via `_drift_check`), so the periodic recompute fires even during long no-raise streaks.
   *(Corrected: an earlier version of this note claimed the 32-step reset "always fires".
   It does not fire on every path — the two $\beta \le 10^{-14}$ guards in `add_edge` and
   `_rank1_update` return before reaching `_drift_check`. The invariant holds anyway,
   because those guards force a full recomputation themselves, but it holds for a
   different reason than stated.)*
7. **Packaging gap (pre-existing, not part of this work).** This package has no
   `features/lite.py`, yet `env.py` imported it eagerly — `env.py` could not be imported at
   all. The lite import is now guarded so the `full` variant runs; `lite_v2` raises a clear
   error.
8. **$\mu$ tolerance and the missing-$\mu$ fallback.** `_refresh_mu` used
   $\max(10^{-9}, \varepsilon_{mult})$ as its "strictly above $\lambda_2$" test while
   multiplicity uses $\varepsilon_{mult} = 10^{-12}$ — a thousand-fold gap, leaving a band
   in which an eigenvalue counted as *outside* the $\lambda_2$ cluster yet *not above*
   $\lambda_2$. $\mu$ then skipped it and overshot the pole at the true $\lambda_3$, so the
   sign test in `_brent_bracket` could succeed for the wrong reason and Brent converge in
   the wrong interval. Both now use $\varepsilon_{mult}$. Separately, when nothing lay
   above $\lambda_2$ the function returned $\lambda_{\max}$ — a bracket spanning the entire
   spectrum, every intervening pole included. It now returns $\infty$ and
   `solve_secular_equation` leaves $\lambda_2$ unchanged.
9. **Drift certificates are additive, not a replacement.** Foster's and the trace identity
   were added alongside the eigenpair residual, not instead of it. This is worth recording
   because the substitution *was* attempted: replacing the residual with the two identities
   let $\lambda_2$ drift to $8\times10^{-2}$ within three edges, since neither identity
   involves the Fiedler vector. On clean runs the residual is the signal that fires
   (measured 20 of 40 edges at $n=32$); Foster and trace have not tripped in testing.
10. **The secular bracket guard is diagnostic.** It counts violations and stores the
    interval, but does not force a recomputation. It was written to re-anchor on
    violation; measurement showed that changes nothing — over 40-edge runs at
    $n = 12/32/64/128$ the final $\lambda_2$ error is bit-identical
    ($3.55\text{e-}15$, $2.22\text{e-}16$, $4.83\text{e-}15$, $5.69\text{e-}16$) whether it
    re-anchors or merely counts. It fires ~10 times per 40 edges, and those firings are
    benign: Case A leaves $Q[:,2:]$ and $\lambda_{3:}$ stale between anchors, so the
    bracket is evaluated against a partly out-of-date basis, while the residual already
    catches every case that actually moves $\lambda_2$. Re-anchoring would have spent ~10
    extra dense eigendecompositions per 40 edges to change no digit.

---

### 7. Numerical constants

| Constant | Value | Used for |
|---|---|---|
| $\varepsilon_{mult}$ | $10^{-12}$ | λ₂ multiplicity (absolute); **also** the "strictly above λ₂" test in `_refresh_mu` |
| zero-projection guard | $10^{-14}$ | $|\psi_2|$ below which Δλ₂ = 0 |
| secular bracket margin | $\max(10^{-12},\,10^{-8}\cdot gap)$ | root bracketing |
| bracket-guard slack | $10^{-8}\cdot\max(1, \psi_2^2, |\Delta\lambda_2|)$ | certifying the secular root (§4 step 5) |
| MaxVol tolerance | $10^{-6}$ | stopping criterion |
| MaxVol max iterations | 200 | stopping criterion |
| $\tau_{drift}$ | $10^{-6}$ | threshold on $\max(res, c_{\mathrm{Foster}}, c_{\mathrm{trace}})$ |
| max steps without reset | 32 | periodic full recomputation |
| $\alpha_1,\alpha_2,\alpha_3$ | 0.10, 0.20, 0.70 | reward weights (convex) |
| $\eta_R,\eta_p$ | per-`n` tables | reward scaling (see [`calibration.py`](analytic_tractable_test_deepseek/calibration.py)) |
| $\mu$ method | dense `eigvalsh` | first eigenvalue above λ₂ (was ARPACK `eigsh` — deviation 8) |

**Not all of these are "confirmed" in the same sense**, and the table used to be
labelled as though they were. $\varepsilon_{mult}$, the zero-projection guard, the
secular margin and the MaxVol constants come from the spec or the task Q&A.
$\tau_{drift}$ and "max steps without reset" do not: `algorithm.md` §2 step 7 says only
*"choose tolerance $\tau_{drift}$ (e.g. $10^{-6}$) and a maximum number of incremental
steps"* — an example value and no step count at all. Both are free parameters.

$\tau_{drift}$ is worth a second look if the recompute rate ever needs tuning: it is an
**absolute, unnormalised** threshold, but $\|L\|_2$ grows as the graph densifies, so it
becomes a progressively tighter relative criterion on denser graphs. That direction is
safe (more recomputation, not less) but it is not free — the recompute rate is currently
~0.5 per edge, and it is the single largest cost in the whole per-step budget.

### 8. File map

| File | Role |
|---|---|
| [`analytic_state.py`](analytic_tractable_test_deepseek/analytic_state.py) | the full algorithm (state, screening, selection, updates) |
| [`env.py`](analytic_tractable_test_deepseek/env.py) | RL environment wiring (analytic candidates + exact updates) |
| [`features/full.py`](analytic_tractable_test_deepseek/features/full.py) | pair-feature builder (accepts analytic candidate set) |
| [`config.py`](analytic_tractable_test_deepseek/config.py) | reward weights / η defaults |
| [`calibrate_eta.py`](analytic_tractable_test_deepseek/calibrate_eta.py) | empirical η_R / η_p calibration script |
| [`calibration.py`](analytic_tractable_test_deepseek/calibration.py) | generated per-`n` η tables |
| [`benchmark_screening.py`](analytic_tractable_test_deepseek/benchmark_screening.py) | O(n²) screening scale check |
| [`test_analytic_state.py`](analytic_tractable_test_deepseek/test_analytic_state.py) | core-math validation vs exact `eigh` |
| [`test_env_analytic.py`](analytic_tractable_test_deepseek/test_env_analytic.py) | end-to-end episode validation |
| [`test_policy_integration.py`](analytic_tractable_test_deepseek/test_policy_integration.py) | policy + analytic candidates integration |

### 9. Validation summary

- Case-A single-edge updates track exact $\lambda_2$, $R_G$, $p_{\min}$ to ~$10^{-16}$.
- Secular roots match exact $\lambda_2'$ to ~$10^{-16}$.
- Case B ($C_6$, $r=2$) resolves to $r=1$ after a symmetry-breaking chord.
- Full episodes (n=12/16/32, up to 186 steps) end exactly at $m_{target}$, candidate sets
  are ≤32 non-edges, and the state matches a fresh dense eigendecomposition to machine
  precision.
- Screening scale check at n=128: **59× faster**, empirical slope 1.66 (vs 3.12 for the
  old $\mathcal O(n^3)$ pass), with identical scores.

#### Per-edge cost, before and after the §4 changes

Wall time per edge added, against a plain dense `eigh` of the whole Laplacian every step
(the cost the analytic path exists to beat). Same edge stream, same machine, 40 edges
from a path graph (25 at n=256):

| n | plain `eigh` | analytic, before | analytic, after | after ÷ `eigh` |
|---|---|---|---|---|
| 12 | 0.019 ms | 0.399 ms (22.2×) | 0.139 ms | **7.3×** |
| 16 | 0.019 ms | 0.397 ms (21.0×) | 0.104 ms | **5.4×** |
| 32 | 0.151 ms | 1.442 ms (9.3×) | 0.213 ms | **1.4×** |
| 64 | 0.438 ms | 2.615 ms (6.1×) | 0.442 ms | **1.01×** |
| 128 | 1.715 ms | 6.196 ms (3.5×) | 1.522 ms | **0.89×** |
| 256 | 7.996 ms | 15.368 ms (1.9×) | 8.866 ms | **1.11×** |

The analytic path now breaks even around $n=64$ and is cheaper than recomputing from
scratch at $n=128$. It remains more expensive at small $n$, where a $12\times12$ `eigh`
is ~19 µs and no amount of bookkeeping competes with simply redoing it.

**Accuracy is unchanged.** Every change in §4 is either algebraically identical (the
$\mathcal O(n)$ centering), the same quantity computed differently (dense $\mu$,
vectorised $p$), or diagnostic-only (the bracket guard). Measured against a fresh
`np.linalg.eigh` at the end of the same runs:

- worst $\lambda_2$ error: $6.66\times10^{-15}$ (test gate: $10^{-6}$)
- worst $p_{\min}$ error: $6.66\times10^{-16}$ (test gate: $10^{-6}$)
- recompute rate: 0.50 per edge, unchanged from before
- all three test modules pass: `test_analytic_state`, `test_env_analytic`,
  `test_policy_integration`

One latent bug was fixed in passing: `from_state_dict` builds the object with
`cls.__new__` and never set the new `p` vector, so `_cheap_drift` would raise
`AttributeError` on the first `add_edge` after any checkpoint resume. `p` is a pure
function of $(A, L^+)$ and is now rebuilt on restore.
