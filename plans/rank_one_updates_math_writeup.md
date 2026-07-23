# Rank-One Updates to the Laplacian and Its Pseudoinverse for Efficient Spectral Property Tracking

## 1. Motivation

In the current codebase, every edge addition triggers a full $O(n^3)$ eigendecomposition of the $n \times n$ Laplacian matrix $L$ to compute the algebraic connectivity $\lambda_2$, Fiedler vector $\phi_2$, higher eigenpairs $(\lambda_3, \phi_3)$, $(\lambda_4, \phi_4)$, and the total effective resistance $R_G = n \sum_{k=2}^n 1/\lambda_k$. These spectral quantities drive:

- The **RL reward**: $r_t = \alpha \cdot \Delta\lambda_2/n + (1-\alpha) \cdot \eta \cdot \Delta R_G / n^2$
- **Node features**: $\phi_2$, $\phi_3$ values per node
- **Candidate selection**: $\lambda_2$-eigenspace gap scores $H(i,j) = \sum_k (\phi_k[i] - \phi_k[j])^2$
- **Global features**: $\lambda_2/n$, $\lambda_3/n$, $\lambda_4/n$

The grounded Laplacian inverse $M = L_g^{-1} \in \mathbb{R}^{(n-1)\times(n-1)}$ is already updated via the Sherman-Morrison formula in [`incremental_linear_algebra.py`](../baseline/src/algebraic_connectivity/incremental_linear_algebra.py), giving $O(1)$ effective resistance queries. But eigenvalues and eigenvectors are still recomputed from scratch.

---

## 2. The Laplacian Update is Exactly Rank-1

Adding an undirected edge between nodes $i$ and $j$ modifies the Laplacian by:

$$
\boxed{L_{\text{new}} = L + \underbrace{(e_i - e_j)(e_i - e_j)^T}_{\Delta L}}
$$

where $e_i \in \mathbb{R}^n$ is the $i$-th standard basis vector. The perturbation $\Delta L = v v^T$ with $v = e_i - e_j$ is a symmetric positive semidefinite rank-1 matrix. Crucially, $v \perp \mathbf{1}$ (since $\sum_k v_k = 0$), so $v$ lies in $\text{range}(L)$ for any connected graph.

---

## 3. What Already Exists: Grounded Inverse via Sherman-Morrison

The code in [`_sherman_morrison_update`](../baseline/src/algebraic_connectivity/incremental_linear_algebra.py:77-L84) maintains the grounded Laplacian inverse $M = L_g^{-1}$ (the inverse of $L$ with the $g$-th row and column removed). The update uses:

$$
M_{\text{new}} = M - \frac{(M g)(M g)^T}{1 + g^T M g}
$$

where $g \in \mathbb{R}^{n-1}$ is the grounded incidence vector for edge $(i,j)$, defined by:

$$
g_k = \begin{cases}
+1 & \text{if node } k \text{ (adjusted for ground) corresponds to } i \\
-1 & \text{if node } k \text{ corresponds to } j \\
0 & \text{otherwise}
\end{cases}
$$

This is mathematically equivalent to the pseudoinverse update restricted to the non-ground indices.

---

## 4. Pseudoinverse Update (Full $n \times n$)

For the Moore-Penrose pseudoinverse $L^+$, the rank-1 Sherman-Morrison formula applies directly because $v \in \text{range}(L)$:

$$
\boxed{L^+_{\text{new}} = L^+ - \frac{(L^+ v)(L^+ v)^T}{1 + v^T L^+ v}}
$$

The denominator satisfies $1 + v^T L^+ v > 0$ since $L^+$ is positive semidefinite on $\text{range}(L)$. The constraint $L^+ \mathbf{1} = 0$ is automatically preserved because $v \perp \mathbf{1}$.

---

## 5. Exact Update for Total Effective Resistance $R_G$

Since $R_G = n \cdot \text{trace}(L^+)$, taking the trace of the pseudoinverse update yields an **exact** formula:

$$
\text{trace}(L^+_{\text{new}}) = \text{trace}(L^+) - \frac{\|L^+ v\|^2}{1 + v^T L^+ v}
$$

Therefore:

$$
\boxed{R_G^{\text{new}} = R_G - n \cdot \frac{\|L^+ v\|^2}{1 + v^T L^+ v}}
$$

In terms of the grounded inverse quantities already computed during the Sherman-Morrison step ($x = M g$, $\texttt{denom} = 1 + g^T x$):

$$
\boxed{R_G^{\text{new}} = R_G - n \cdot \frac{\|x\|^2}{\texttt{denom}}}
$$

This is **exact** — no perturbation approximation. The only additional cost beyond the existing SM update is computing $\|x\|^2$, which is an $O(n)$ dot product.

---

## 6. Eigenvalue Updates via First-Order Perturbation Theory

### 6.1 Non-Degenerate Case

For an isolated eigenvalue $\lambda_k$ with normalized eigenvector $\phi_k$, first-order perturbation theory for $\Delta L = v v^T$ gives:

$$
\boxed{\lambda_k^{\text{new}} \approx \lambda_k + (\phi_k^T v)^2 = \lambda_k + (\phi_k[i] - \phi_k[j])^2}
$$

**Key insight**: The Fiedler gap score $H(i,j)$ already computed in the candidate features **is** the first-order estimate of $\Delta\lambda_2$:

$$
\widehat{\Delta\lambda_2} = (\phi_2[i] - \phi_2[j])^2
$$

The error in this approximation is $O(\|\Delta L\|^2 / \min_{j \neq k} |\lambda_k - \lambda_j|)$. For well-separated eigenvalues, it is excellent.

### 6.2 Degenerate Case (Multiplicity $d > 1$)

Let $\lambda$ have multiplicity $d$ with orthonormal eigenspace basis $P = [\phi_{k+1} \mid \cdots \mid \phi_{k+d}] \in \mathbb{R}^{n \times d}$. The perturbation projected onto this subspace is:

$$
P^T \Delta L \, P = (P^T v)(P^T v)^T
$$

This is a rank-1 $d \times d$ matrix. Its eigenvalues are $\|P^T v\|^2$ (once) and $0$ ($d-1$ times). Therefore:

$$
\boxed{\lambda_{\text{new}} \in \{\lambda + H(i,j),\; \underbrace{\lambda, \ldots, \lambda}_{d-1 \text{ times}}\}}
$$

where $H(i,j) = \|P^T v\|^2 = \sum_{m=1}^d (\phi_{k+m}[i] - \phi_{k+m}[j])^2$ is the **$\lambda$-eigenspace gap** — exactly what the function [`_lambda2_eigenspace_gap`](../rl_train/features/full.py:89-L106) already computes.

This formula is exact (not perturbative) for rank-1 updates: one eigenvalue in the cluster shifts by the total squared gap, and the remaining $d-1$ eigenvalues are unchanged.

The first-order $R_G$ update remains consistent: each $\lambda$ in the cluster contributes $-(\phi_{k+m}[i] - \phi_{k+m}[j])^2 / \lambda^2$, and summing over the $d$ eigenvectors yields $-H(i,j)/\lambda^2$, which matches the first-order Taylor expansion of $1/(\lambda + H) - 1/\lambda$.

---

## 7. Eigenvector Tracking via Inverse Iteration on $L^+$

Since $L$ and $L^+$ share eigenvectors, the dominant eigenvector of $L^+_{\text{new}}$ corresponds to $1/\lambda_2^{\text{new}}$, the smallest nonzero eigenvalue of $L_{\text{new}}$.

### 7.1 Single-Vector Inverse Iteration (for multiplicity $d = 1$)

Starting from the current Fiedler vector $\phi_2$, one step of inverse iteration on the updated pseudoinverse refines the estimate:

1. **Restrict** $\phi_2$ to non-ground nodes, yielding $\tilde{\phi}_2 \in \mathbb{R}^{n-1}$ (drop the ground node component)
2. **Apply** $M_{\text{new}}$: $w = M_{\text{new}} \cdot \tilde{\phi}_2$
3. **Expand** back to full $n$-dimensional space (insert zero at ground node position)
4. **Remove mean**: $w \leftarrow w - \bar{w} \cdot \mathbf{1}$ (to stay in $\mathbf{1}^\perp$)
5. **Normalize**: $\phi_2^{\text{new}} = w / \|w\|$

The matrix-vector multiply with $M_{\text{new}}$ is $O(n^2)$. One or two iterations typically suffice since $\phi_2$ changes slowly with single-edge additions.

### 7.2 Subspace Iteration (for multiplicity $d > 1$)

When $\lambda_2$ has multiplicity $d > 1$, we track the entire eigenspace via block inverse iteration (orthogonal iteration):

1. Form $Q^{(0)} = [\phi_2, \ldots, \phi_{d+1}] \in \mathbb{R}^{n \times d}$ from the last exact eigenspace
2. **Power step**: $Z^{(t+1)} = M_{\text{new}} \cdot Q^{(t)}_{\text{restricted}}$ (apply to each column)
3. **Orthonormalize**: $Q^{(t+1)} R = Z^{(t+1)}$ (QR decomposition), then expand to full $n$-dim
4. **Rayleigh-Ritz**: Form $H = (Q^{(t+1)})^T L_{\text{new}} Q^{(t+1)}$ (a $d \times d$ matrix), diagonalize to get refined eigenvalue estimates

Cost: $O(d \cdot n^2)$ per step. For typical multiplicities ($d \leq 4$), this is still much cheaper than $O(n^3)$.

### 7.3 Refined $\lambda_2$ via Rayleigh Quotient

After inverse iteration, compute the Rayleigh quotient for a sharper $\lambda_2$ estimate:

$$
\lambda_2^{\text{new}} \approx \frac{(\phi_2^{\text{new}})^T L_{\text{new}} \, \phi_2^{\text{new}}}{(\phi_2^{\text{new}})^T \phi_2^{\text{new}}}
$$

Using $L_{\text{new}} = L + v v^T$:

$$
(\phi_2^{\text{new}})^T L_{\text{new}} \, \phi_2^{\text{new}} = (\phi_2^{\text{new}})^T L \, \phi_2^{\text{new}} + (\phi_2^{\text{new}}[i] - \phi_2^{\text{new}}[j])^2
$$

The term $(\phi_2^{\text{new}})^T L \, \phi_2^{\text{new}}$ can be computed either:
- In $O(n^2)$ via direct matrix-vector multiply with $L$, or
- Using the stored eigendecomposition: $\sum_{k=1}^n \lambda_k (\phi_k^T \phi_2^{\text{new}})^2$ in $O(n^2)$

---

## 8. Proposed Three-Tier Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Tier 1  —  O(1) per edge  —  Already Implemented             │
│                                                              │
│  Grounded inverse M via Sherman-Morrison                     │
│  → O(1) effective resistance R_eff(u,v) for any pair         │
│  → Used by CandidateGenerator for ER-based candidate scoring │
├──────────────────────────────────────────────────────────────┤
│ Tier 2  —  O(n²) per edge  —  New Incremental Updates        │
│                                                              │
│  R_G     →  exact via trace identity (from SM intermediates) │
│  λ₂,λ₃,λ₄ →  1st-order perturbation: λ' ≈ λ + gap²          │
│  φ₂,φ₃,φ₄ →  1-step inverse iteration on M_new              │
│  multiplicity →  subspace iteration for d > 1                │
│                                                              │
│  All quantities available in O(1) after Tier 2 update        │
├──────────────────────────────────────────────────────────────┤
│ Tier 3  —  O(n³) every K steps  —  Drift Correction          │
│                                                              │
│  Full np.linalg.eigh(L) to reset accumulated errors          │
│  Stores complete eigenvector matrix for future Tier 2 use    │
│  K = configurable (default 32), also triggered on-demand     │
│  when estimated eigenvalues drift beyond tolerance           │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Implementation Targets

### 9.1 New Module: `SpectralTracker`

A class (placed either in `rl_train/incremental_spectral.py` or extending the existing `IncrementalState` in the baseline) that:

- **Wraps** the grounded inverse $M$ and its Sherman-Morrison update
- **Maintains** cached $\lambda_2, \lambda_3, \lambda_4, \phi_2, \phi_3, \phi_4, R_G$, and multiplicity
- **On `add_edge(i, j)`**: updates $M$ via SM (existing), then applies Tier 2 formulas
- **Exposes** $O(1)$ accessors: `get_lambda2()`, `get_phi2()`, `get_RG()`, `get_multiplicity()`
- **Periodically** triggers Tier 3: `force_exact()` → full `eigh(L)` + store eigenvector matrix
- **Handles** ground-node index mapping for all vector operations

### 9.2 Modifications to Existing Code

| File | Change |
|------|--------|
| [`rl_train/graph_math.py`](../rl_train/graph_math.py) | Add `spectral_features_incremental()` that delegates to `SpectralTracker` |
| [`rl_train/env.py`](../rl_train/env.py:548) | Replace `spectral_features(self.adj)` calls in `step()` with incremental updates; add exact reset every $K$ steps |
| [`baseline/.../incremental_linear_algebra.py`](../baseline/src/algebraic_connectivity/incremental_linear_algebra.py) | Extend `IncrementalState` with Tier 2 eigenvalue/eigenvector tracking and the exact $R_G$ formula |
| [`baseline/.../candidates.py`](../baseline/src/algebraic_connectivity/candidates.py) | Optionally use incremental Fiedler estimates instead of triggering `spectral_chart()` which calls full `eigsh` |

### 9.3 Key Design Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Store full $n \times n$ eigenvector matrix? | Yes, for $n \leq 256$ | Memory is negligible (~512KB for $n=256$); enables exact $R_G$ via trace identity |
| How many inverse iterations per step? | 1 by default, 2-3 when $\Delta\lambda_2$ estimate exceeds threshold | 1 step is $O(n^2)$ and usually sufficient; extra steps only when needed |
| When to trigger exact recomputation? | Every $K = 32$ steps, or when $\|\lambda_k^{\text{estimated}} - \lambda_k^{\text{Rayleigh}}\| > \varepsilon$ | Matches existing `exact_reset_every_steps` pattern; adaptive trigger catches rapid drift |
| Handle $\lambda_3, \lambda_4$ too? | Yes, same pattern as $\lambda_2$ | Needed for global features and multi-spectral candidate scoring |

---

## 10. Edge Cases and Safeguards

### 10.1 Near-Multiplicity

When $|\lambda_k - \lambda_{k+1}| < \epsilon$, the first-order eigenvector perturbation formula has a small denominator. Mitigations:

- Detect at Tier 3 and mark as "near-degenerate"
- Use subspace iteration (treat the cluster as a block) for those eigenvalues
- Increase exact reset frequency for those episodes

### 10.2 Disconnected Graphs

If the graph becomes disconnected (which should not happen during sequential edge addition from a connected initial graph), $\lambda_2 = 0$ and the pseudoinverse update remains valid. The Sherman-Morrison denominator stays positive because $v^T L^+ v$ is still well-defined (though $L^+$ eigenvalue for the ones vector is zero, $v \perp \mathbf{1}$ ensures we stay in $\text{range}(L)$).

### 10.3 Numerical Stability of $M$

After many SM updates, rounding errors accumulate in $M$. The existing code already handles this with `exact_reset_every_steps` and a ridge term $10^{-10} I$. The Tier 2 eigenvalue/eigenvector estimates should similarly be reset at Tier 3.

---

## 11. Expected Performance Impact

| Operation | Current Cost | Proposed Cost | Improvement |
|-----------|-------------|---------------|-------------|
| $\lambda_2, \phi_2$ after edge add | $O(n^3)$ (full `eigh`) | $O(n^2)$ (inverse iteration) | ~$n\times$ speedup |
| $R_G$ after edge add | $O(n^3)$ (full `eigh`) | $O(n)$ (trace identity) | ~$n^2\times$ speedup |
| $R_{\text{eff}}(u,v)$ query | $O(1)$ | $O(1)$ | No change (already optimal) |
| Periodic exact recomputation | $O(n^3)$ every $K$ steps | $O(n^3)$ every $K$ steps | No change |
| **Amortized per-step cost** | $O(n^3)$ | $O(n^2 + n^3/K)$ | ~$K\times$ speedup |
