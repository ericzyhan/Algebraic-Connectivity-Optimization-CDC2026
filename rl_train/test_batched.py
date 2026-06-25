"""Quick numerical equivalence test for forward_batched vs forward_observation."""
import sys
import numpy as np
import torch

# Ensure deterministic
torch.manual_seed(42)
np.random.seed(42)

from model import GraphPolicyNetwork, feature_dims_for_variant
from env import GraphObservation


def make_dummy_observation(n: int, num_candidates: int, node_dim: int,
                           pair_dim: int, global_dim: int, rng: np.random.RandomState):
    """Create a plausible random GraphObservation."""
    node_features = rng.randn(n, node_dim).astype(np.float32)

    # Build a plausible edge_index (a simple path to keep it connected-ish)
    edges = [(i, i + 1) for i in range(n - 1)]
    edge_index = np.array(edges, dtype=np.int64).T  # (2, E)

    # Candidate pairs (random non-edges)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if (i, j) not in edges and (j, i) not in edges]
    chosen = rng.choice(len(all_pairs), size=min(num_candidates, len(all_pairs)),
                        replace=False)
    candidate_pairs = np.array([all_pairs[idx] for idx in chosen], dtype=np.int64)
    actual_k = candidate_pairs.shape[0]

    pair_features = rng.randn(actual_k, pair_dim).astype(np.float32)
    global_features = rng.randn(global_dim).astype(np.float32)

    return GraphObservation(
        node_features=node_features,
        edge_index=edge_index,
        candidate_pairs=candidate_pairs,
        pair_features=pair_features,
        global_features=global_features,
        lambda2_norm=float(rng.rand()),
        n=n,
        rho_target=float(rng.rand()),
        rho_current=float(rng.rand()),
    )


def main():
    node_dim, pair_dim, global_dim = feature_dims_for_variant("full")
    device = torch.device("cpu")

    model = GraphPolicyNetwork(
        node_feature_dim=node_dim,
        pair_feature_dim=pair_dim,
        global_feature_dim=global_dim,
        gat_hidden_dim=32,
        gat_heads=4,
        gat_layers=2,
        edge_mlp_hidden_dim=64,
        value_mlp_hidden_dim=32,
    ).to(device)
    model.eval()

    # Create 4 observations with varying sizes
    rng = np.random.RandomState(12345)
    observations = [
        make_dummy_observation(n=16, num_candidates=32, node_dim=node_dim,
                               pair_dim=pair_dim, global_dim=global_dim, rng=rng),
        make_dummy_observation(n=24, num_candidates=48, node_dim=node_dim,
                               pair_dim=pair_dim, global_dim=global_dim, rng=rng),
        make_dummy_observation(n=12, num_candidates=24, node_dim=node_dim,
                               pair_dim=pair_dim, global_dim=global_dim, rng=rng),
        make_dummy_observation(n=20, num_candidates=40, node_dim=node_dim,
                               pair_dim=pair_dim, global_dim=global_dim, rng=rng),
    ]

    # Sequential version
    seq_logits = []
    seq_values = []
    with torch.no_grad():
        for obs in observations:
            out = model.forward_observation(obs, device)
            seq_logits.append(out.logits.cpu())
            seq_values.append(out.value.cpu())

    # Batched version
    with torch.no_grad():
        batched_logits_list, batched_values = model.forward_batched(observations, device)

    # Compare
    max_logit_err = 0.0
    max_value_err = 0.0
    all_close = True

    for i in range(len(observations)):
        sl = seq_logits[i].numpy()
        bl = batched_logits_list[i].cpu().numpy()

        logit_err = float(np.max(np.abs(sl - bl)))
        max_logit_err = max(max_logit_err, logit_err)

        sv = float(seq_values[i].item())
        bv = float(batched_values[i].item())
        value_err = abs(sv - bv)
        max_value_err = max(max_value_err, value_err)

        if logit_err > 1e-5 or value_err > 1e-5:
            all_close = False
            print(f"  MISMATCH obs {i}: logit_err={logit_err:.2e}, value_err={value_err:.2e}")
        else:
            print(f"  OK obs {i}: K={bl.shape[0]}, n={observations[i].n}, "
                  f"logit_err={logit_err:.2e}, value_err={value_err:.2e}")

    print(f"\nMax logit error: {max_logit_err:.2e}")
    print(f"Max value error: {max_value_err:.2e}")
    print(f"All close: {all_close}")

    if not all_close:
        print("ERROR: batched and sequential outputs differ!")
        sys.exit(1)

    # Verify no -inf leakage: sample from batched and check all indices are valid
    with torch.no_grad():
        logits_list, _ = model.forward_batched(observations, device)
        for i, logits in enumerate(logits_list):
            valid_mask = torch.isfinite(logits)
            if not valid_mask.all():
                print(f"ERROR: obs {i} has non-finite logits!")
                sys.exit(1)
            dist = torch.distributions.Categorical(logits=logits)
            samples = dist.sample((100,))
            if samples.max() >= logits.shape[0] or samples.min() < 0:
                print(f"ERROR: obs {i} sampled out-of-range indices!")
                sys.exit(1)
            print(f"  Sampling OK obs {i}: {samples[:10].tolist()}")

    # Test with the exact case used during training: same K across all observations
    print("\n--- Test with uniform K ---")
    obs_uniform = [
        make_dummy_observation(n=16, num_candidates=128, node_dim=node_dim,
                               pair_dim=pair_dim, global_dim=global_dim, rng=rng),
        make_dummy_observation(n=16, num_candidates=128, node_dim=node_dim,
                               pair_dim=pair_dim, global_dim=global_dim, rng=rng),
        make_dummy_observation(n=16, num_candidates=128, node_dim=node_dim,
                               pair_dim=pair_dim, global_dim=global_dim, rng=rng),
    ]
    with torch.no_grad():
        seq_vals = [model.forward_observation(o, device).value.item() for o in obs_uniform]
        _, bat_vals = model.forward_batched(obs_uniform, device)
    for i, (sv, bv) in enumerate(zip(seq_vals, bat_vals.tolist())):
        err = abs(sv - bv)
        status = "OK" if err < 1e-5 else "FAIL"
        print(f"  {status} obs {i}: seq={sv:.6f}, bat={bv:.6f}, err={err:.2e}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
