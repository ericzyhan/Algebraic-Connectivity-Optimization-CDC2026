from __future__ import annotations


class RLEnvironment:
    def reset(self, *args, **kwargs):
        raise NotImplementedError("RL environment is intentionally deferred.")

    def step(self, action):
        raise NotImplementedError("RL environment is intentionally deferred.")


class ObservationBuilder:
    def build(self, *args, **kwargs):
        raise NotImplementedError("Node/edge feature design is intentionally deferred.")


class RewardModel:
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("Reward design is intentionally deferred.")


class RLPolicyModel:
    def act(self, *args, **kwargs):
        raise NotImplementedError("RL policy model is intentionally deferred.")
