from __future__ import annotations

from .spectral import SpectralTracker


class PotentialShaper:
    """Ng-Harada-Russell potential-based shaping for the r>1 credit-assignment
    gap (docs/algorithm.md S5.2): F = gamma * Phi(s') - Phi(s),
    Phi(s) = alpha_1 * sigma_r(Z_accum).

    Policy-invariant only if `gamma` matches the RL discount factor exactly,
    so the caller must pass the same gamma used by PPO/REINFORCE.

    Phi is defined as 0 whenever multiplicity <= 1: in that regime the real
    reward already carries nonzero signal every step (the secular root is
    generically nonzero), so there is nothing to backfill.
    """

    def __init__(self, alpha1: float, gamma: float):
        self.alpha1 = float(alpha1)
        self.gamma = float(gamma)
        self._prev_phi = 0.0

    def _phi(self, tracker: SpectralTracker) -> float:
        if tracker.multiplicity <= 1:
            return 0.0
        return self.alpha1 * tracker.sigma_r_accum()

    def reset(self, tracker: SpectralTracker) -> None:
        self._prev_phi = self._phi(tracker)

    def step(self, tracker: SpectralTracker) -> float:
        phi_new = self._phi(tracker)
        shaping_reward = self.gamma * phi_new - self._prev_phi
        self._prev_phi = phi_new
        return float(shaping_reward)

    def state_dict(self) -> dict:
        return {"alpha1": self.alpha1, "gamma": self.gamma, "prev_phi": self._prev_phi}

    @classmethod
    def from_state_dict(cls, state: dict) -> "PotentialShaper":
        obj = cls(alpha1=float(state["alpha1"]), gamma=float(state["gamma"]))
        obj._prev_phi = float(state["prev_phi"])
        return obj
