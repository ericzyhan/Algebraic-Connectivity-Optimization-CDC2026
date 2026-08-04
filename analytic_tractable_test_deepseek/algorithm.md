# Complete Edge Addition Algorithm (Final Air‑tight Specification)

This specification is fully self‑contained and unambiguous. Every numerical routine is spelled out; all edge cases are addressed. It can be handed directly to an implementation team.

## Preliminaries

- **Graph** $G=(V,E)$, $|V| = n$, Laplacian $L$.
- **Eigenvalues** $0 = \lambda_1 < \lambda_2 \le \lambda_3 \le \dots \le \lambda_n$; eigenvectors $q_i$ (columns of $Q$).
- **Non‑edge set** $\mathcal{E}_{\text{non}}$ of size $k$. For each candidate $(i,j)$,
  $a = e_i - e_j \in \mathbb{R}^n$.
- **Reward heuristic** (provided externally):
  $\mathcal{R} = \alpha_1 \frac{\Delta \lambda_2}{n} + \alpha_2 \eta_R \frac{\Delta R_G}{n^2} + \alpha_3 \eta_p \Delta p_{\min}$,
  with default $\alpha_3 = 0.7$. The quantities $\eta_R, \eta_p, p_{\min}$ must be supplied by the caller.
- **Effective resistance** of a non‑edge: $R_e = a^\top L^+ a$.
- **Total effective resistance** $R_G = n \, \text{tr}(L^+)$.

## Initialisation

1. Build Laplacian $L$ from the initial graph.
2. **Full eigendecomposition** (once):
   $L = Q \Lambda Q^\top,\quad \Lambda = \operatorname{diag}(0,\lambda_2,\dots,\lambda_n)$.
   Store $Q$ and all eigenvalues $\{\lambda_i\}$.
3. **Multiplicity** $r$ of $\lambda_2$:
   $r = \#\{\lambda_i \mid |\lambda_i - \lambda_2| < \epsilon_{\text{mult}}\}$
   (typically $\epsilon_{\text{mult}} = 10^{-12}$).
   Let $U \in \mathbb{R}^{n \times r}$ be the columns of $Q$ belonging to this cluster.
   Set $\mu = \lambda_{r+1}$ (first eigenvalue strictly greater than $\lambda_2$).
4. **Pseudoinverse** and auxiliary vectors:
   $L^+ = \sum_{i=2}^n \frac{1}{\lambda_i} q_i q_i^\top$.
   Pre‑compute the row vectors (index $0$ for the zero eigenvalue)
   $\mathbf{inv} = \bigl[0, 1/\lambda_2, \dots, 1/\lambda_n\bigr]$,
   $\mathbf{inv2} = \bigl[0, 1/\lambda_2^2, \dots, 1/\lambda_n^2\bigr]$.
5. **Non‑edge matrix** $B \in \mathbb{R}^{n \times k}$: column $j$ is the incidence vector of the $j$-th non‑edge.
   Compute the full projection matrix
   $Z_{\text{full}} = Q^\top B \in \mathbb{R}^{n \times k}$.
   Column $j$ is $\psi^{(j)} = Q^\top a^{(j)}$.
6. **Initial global metrics**:
   $R_G = n \sum_{i=2}^n \frac{1}{\lambda_i}$,
   $p_{\min}$ (external function of the graph).
7. **Drift control**: choose tolerance $\tau_{\text{drift}}$ (e.g. $10^{-6}$) and a maximum number of incremental steps between forced full recomputations.

## Main Loop

Repeat until the edge budget is exhausted or $\lambda_2$ reaches a target.

### Candidate Screening (Stage 1)

For every unused non‑edge $j$:

- $R_e[j] = \sum_{i=2}^n \frac{\bigl(\psi_i^{(j)}\bigr)^2}{\lambda_i} = \mathbf{inv} \cdot \bigl(\psi^{(j)} \odot \psi^{(j)}\bigr)$.
- Multispectral Fiedler score $F[j] = \sum_{\ell=1}^r \bigl(\psi_\ell^{(j)}\bigr)^2$, where rows $1,\dots,r$ of $Z_{\text{full}}$ correspond to $U$.

Form:
- $\mathcal{I}_R$ = indices of the **top 32** by $R_e$ (descending).
- $\mathcal{I}_F$ = indices of the **top 32** by $F$.

Set $\mathcal{I}_{\text{cand}} = \text{unique}(\mathcal{I}_R \cup \mathcal{I}_F)$, truncate to at most 64.
If fewer than 64 non‑edges remain, keep all.
Build
$B' = B\bigl[:\,,\, \mathcal{I}_{\text{cand}}\bigr], \quad
 Z'_{\text{full}} = Z_{\text{full}}\bigl[:\,,\, \mathcal{I}_{\text{cand}}\bigr].$

### Second‑Stage Selection

#### Case A: $r = 1$ (simple $\lambda_2$)

For each candidate index $c$ (local column of $B'$):

1. Extract $a = B'[:,c]$ and $\psi = Z'_{\text{full}}[:,c]$.
2. **Zero‑projection guard**: if $|\psi_2| < 10^{-14}$, set $\Delta\lambda_2 = 0$, $\Delta R_G = 0$, and skip the secular equation.
   Otherwise:
   - $R_e = \sum_{i=2}^n \psi_i^2 / \lambda_i$.
   - $\Delta R_G = -n \frac{\sum_{i=2}^n \frac{\psi_i^2}{\lambda_i^2}}{1 + \sum_{i=2}^n \frac{\psi_i^2}{\lambda_i}}$.
   - Solve the secular equation
     $f(\lambda) = 1 + \sum_{i=2}^n \frac{\psi_i^2}{\lambda_i - \lambda} = 0$
     on the bracket $(\lambda_2+\epsilon, \mu)$ with $\epsilon = 10^{-12}$,
     using Brent’s method (or BNS safeguarded rational interpolation).
     The root is $\lambda_2'$. Set $\Delta\lambda_2 = \lambda_2' - \lambda_2$.
3. Compute $\Delta p_{\min}$ by tentatively adding the edge (or a cheap approximation acceptable to the caller).
4. Reward:
   $\mathcal{R}_c = \alpha_1 \frac{\Delta\lambda_2}{n} + \alpha_2 \eta_R \frac{\Delta R_G}{n^2} + \alpha_3 \eta_p \Delta p_{\min}$.

Select $c^*$ maximising $\mathcal{R}_c$.
The winning edge is $e^* = \mathcal{I}_{\text{cand}}[c^*]$.

*After selection*, perform the **single‑edge update** (Section Single‑Edge Update).

#### Case B: $r > 1$ (multiple $\lambda_2$)

1. **Build projected matrix**
   $Z_{\text{mult}} = U^\top B' \in \mathbb{R}^{r \times |\mathcal{I}_{\text{cand}}|}$.
2. **Filter high‑potential columns**
   - $n_j = \|Z_{\text{mult}}[:,j]\|_2$.
   - $\tau =$ 75th percentile of $\{n_j\}$.
   - Keep columns with $n_j \ge \tau$. If fewer than $r$ columns survive,
     lower $\tau$ to the 50th, then the 25th percentile, until at least $r$ columns remain.
   - Let $J_{\text{keep}}$ be the local indices (size $m$) and form
     $Z_{\text{keep}} = Z_{\text{mult}}[:, J_{\text{keep}}] \in \mathbb{R}^{r \times m}$.
3. **MaxVol selection**
   - Thin QR: $Z_{\text{keep}}^\top = Q_R R$, $Q_R \in \mathbb{R}^{m \times r}$ orthonormal.
   - LU pivoting: $P Q_R = L U$, with row permutation vector $p$ (0‑based)
     such that the $i$‑th row of $P Q_R$ is row $p[i]$ of $Q_R$.
   - Initial set $I_{\text{init}} = \{p[0], \dots, p[r-1]\}$.
   - MaxVol loop (input $Q_R$, $I_{\text{init}}$, tolerance $10^{-6}$, max iterations 100):
     + $I = I_{\text{init}}$.
     + Compute $C = Q_R \bigl(Q_R[I,:]\bigr)^{-1}$.
     + Find $(i_{\max}, j_{\max}) = \arg\max_{i \notin I, j \in I} |C[i,j]|$.
     + If $\max|C| \le 1+10^{-6}$, stop.
     + Replace $j_{\max}$ by $i_{\max}$ in $I$.
     + Repeat.
   - Output $I_{\text{opt}} = I$.
   - The $r$ candidate edges are
     $\mathcal{E}_r = \bigl\{ \mathcal{I}_{\text{cand}}\bigl[J_{\text{keep}}[i]\bigr] \mid i \in I_{\text{opt}} \bigr\}$.
4. **Choose $r-1$ edges**
   Discard any one edge from $\mathcal{E}_r$ (e.g. the last).
   The remaining $r-1$ edges form the addition list $\mathcal{E}_{\text{add}}$.
5. **Sequential addition of the $r-1$ edges**
   Set $R_G^{\text{acc}} = 0$ (accumulator).
   For each edge $e \in \mathcal{E}_{\text{add}}$ with incidence vector $a$:
   a. **Before the update**, compute the $\Delta R_G$ for this single addition using the current $L^+$:
      $\beta = 1 + a^\top L^+ a$,
      $\Delta R_G^{\text{edge}} = -n \frac{a^\top (L^+)^2 a}{\beta}$.
      (The quantity $a^\top (L^+)^2 a$ is evaluated with two matrix‑vector products using $L^+$; no eigenbasis needed.)
   b. **Update $R_G$**: $R_G \gets R_G + \Delta R_G^{\text{edge}}$.
      Accumulate: $R_G^{\text{acc}} \gets R_G^{\text{acc}} + \Delta R_G^{\text{edge}}$.
   c. **Rank‑1 Laplacian / pseudoinverse update**:
      $L \gets L + a a^\top$,
      $L^+ \gets L^+ - \frac{(L^+ a)(L^+ a)^\top}{\beta}$,
      $L^+ \gets \bigl(I - \frac{1}{n}J\bigr) L^+ \bigl(I - \frac{1}{n}J\bigr), \quad J = \mathbf{1}\mathbf{1}^\top$.
   d. **Rayleigh–Ritz refinement of the $\lambda_2$ eigenspace**:
      + Form $H = U^\top L U \in \mathbb{R}^{r \times r}$ (using the new $L$).
      + Solve $H V = V \Theta$, with $\Theta = \operatorname{diag}(\theta_1,\dots,\theta_r)$.
      + Updated eigenspace: $U_{\text{new}} = U V$.
      + Let $\theta_{\min}$ be the smallest eigenvalue in $\Theta$.
        Check its multiplicity $r_{\text{new}}$.
      + If $r_{\text{new}} = 1$ (simple), set
        $r = 1$,
        $\lambda_2 = \theta_{\min}$,
        $u_2 =$ column of $U_{\text{new}}$ corresponding to $\theta_{\min}$,
        and **break** the edge loop (remaining edges will be added later via the simple‑$\lambda_2$ path, if desired).
      + Otherwise, set $r = r_{\text{new}}$, $U = U_{\text{new}}$ (all $r$ vectors), and continue.
   e. (Optional) One step of inverse iteration on $u_2$ using the new $L$ to polish the vector.

   After processing all $r-1$ edges, if $r$ is still $>1$, either repeat the multiplicity procedure or trigger a full recomputation.
6. **Post‑multiplicity adjustments**
   - **Update $\mu$**: using the final $L$ (after all $r-1$ edges added),
     run a few steps of Lanczos with shift $\sigma = \lambda_2 + \delta$
     to obtain the next eigenvalue above $\lambda_2$.
     Set $\mu$ to that eigenvalue. (If a full recomputation is triggered anyway, $\mu$ will be exact.)
   - **Refresh projection data**: let $U_2$ be the current basis of the $\lambda_2$ eigenspace (if $r>1$, the whole $U$; if $r=1$, just the single Fiedler vector).
     Recompute the corresponding rows of $Z_{\text{full}}$:
     $Z_{\text{full}}[1..r, :] = U_2^\top B$.
   - The $\Delta R_G$ for this step is already fully accumulated in $R_G$ from the sequential loop above. The reward can be computed retrospectively if needed.

### Single‑Edge Update (used in Case A)

Assume an edge with incidence vector $a$ and projection $\psi$ (computed from the *current* eigenbasis before the update) has been selected.

1. $\beta = 1 + a^\top L^+ a$.
2. **Exact $\Delta R_G$**:
   $\Delta R_G = -n \frac{a^\top (L^+)^2 a}{\beta}$.
   $R_G \gets R_G + \Delta R_G$.
3. Rank‑1 Laplacian and pseudoinverse update (identical to step 5c above).
4. **Fiedler vector update** (exact secular formula):
   $u_2' = \sum_{i=2}^n \frac{\psi_i}{\lambda_i - \lambda_2'} q_i$,
   followed by normalisation $u_2' = u_2' / \|u_2'\|_2$.
   (Here $\lambda_2'$ is the root obtained from the secular equation during the selection phase.)
5. **Replace $\lambda_2$ and eigenvector**:
   - Identify the index $\ell$ of $\lambda_2$ in the eigenvalue list.
   - $\lambda_\ell = \lambda_2'$, $Q[:,\ell] = u_2'$.
   - Update row $\ell$ of $Z_{\text{full}}$: $Z_{\text{full}}[\ell, :] = u_2'^\top B$.
6. **Update $\mu$**:
   - After the rank‑1 update, $\mu$ may shift. Perform a few Lanczos steps to find the first eigenvalue $> \lambda_2'$. Assign that to $\mu$.
     (If a drift reset occurs later, $\mu$ becomes exact.)
7. **Drift monitoring**:
   Compute residual $\text{res} = \|L u_2' - \lambda_2' u_2'\|_2$.
   If $\text{res} > \tau_{\text{drift}}$ (or step counter exceeds limit):
   - **Full recomputation**: eigendecomposition of $L$ → new $Q, \{\lambda_i\}, L^+, r, \mu, U$.
   - Rebuild $Z_{\text{full}} = Q^\top B$.
   - Reset drift step counter.
   Otherwise, keep the approximate basis.

## Termination

The loop stops when the allowed number of edges is added or $\lambda_2$ reaches a prescribed value. The final graph state, $\lambda_2$, $R_G$, and $p_{\min}$ are returned.

## Implementation Notes (Clarifications)

- **Zero projection**: In Case A, before entering the secular solver, check if $\psi_2 \approx 0$. If true, the Fiedler vector does not move; set $\Delta\lambda_2 = 0$ and skip root‑finding. The $\Delta R_G$ formula still works (it uses the full $\psi$, not just $\psi_2$).
- **$\Delta R_G$ in multiplicity path**: In Case B, $\Delta R_G$ was *not* pre‑computed during the screening. The sequential loop explicitly computes $a^\top (L^+)^2 a$ and the corresponding $\Delta R_G$ before each rank‑1 update, and accumulates the total into $R_G$. The value of $a^\top (L^+)^2 a$ can be obtained by computing $v = L^+ a$ and then $v^\top v$ (since $L^+$ is symmetric).
- **Eigenvector after multiplicity break**: When the Rayleigh–Ritz step yields a simple $\theta_{\min}$, we set $u_2$ directly to the corresponding Ritz vector *without* applying the secular eigenvector formula. That formula is only used in the pure single‑edge path where $\lambda_2'$ came from solving the secular equation. This avoids mixing exact and approximate quantities.
- **Updating $\mu$**: After any change to $L$, $\mu$ (the first eigenvalue $> \lambda_2$) may shift. We always refresh $\mu$ using a few Lanczos iterations, unless a full recomputation is performed anyway. This ensures the bracket for secular root‑finding remains correct.
- **MaxVol when $m = r$**: If the column‑norm filter yields exactly $r$ candidates, the QR factor $Q_R$ is square. The initial active set $I_{\text{init}}$ will be $\{0,1,\dots,r-1\}$ (after LU pivoting). The MaxVol loop terminates immediately because there are no rows outside $I$ to consider. This is correct: the $r$ chosen edges are exactly those $r$ candidates.