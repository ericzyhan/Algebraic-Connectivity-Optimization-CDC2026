from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .backbone_init import BackboneInitializer, default_path_init_metadata, normalize_init_metadata
from .checkpoint import load_checkpoint
from .config import BackboneInitConfig, Config, load_config
from .curriculum import CurriculumScheduler
from .env import GraphEnv
from .graph_math import edge_count, m_target_from_rho
from .model import GraphPolicyNetwork, feature_dims_for_variant
from .selector import PairSelectorNetwork, adaptive_top_k, slice_observation, stable_topk_indices


def _build_model_from_config(cfg: Config, device: torch.device) -> GraphPolicyNetwork:
    node_dim, pair_dim, global_dim = feature_dims_for_variant(cfg.variant.name)
    model = GraphPolicyNetwork(
        node_feature_dim=node_dim,
        pair_feature_dim=pair_dim,
        global_feature_dim=global_dim,
        gat_hidden_dim=cfg.model.gat_hidden_dim,
        gat_heads=cfg.model.gat_heads,
        gat_layers=cfg.model.gat_layers,
        edge_mlp_hidden_dim=cfg.model.edge_mlp_hidden_dim,
        value_mlp_hidden_dim=cfg.model.value_mlp_hidden_dim,
    )
    model = model.to(device)
    model.eval()
    return model


def _build_selector_from_config(cfg: Config, device: torch.device) -> PairSelectorNetwork:
    node_dim, pair_dim, global_dim = feature_dims_for_variant(cfg.variant.name)
    selector = PairSelectorNetwork(
        node_feature_dim=node_dim,
        pair_feature_dim=pair_dim,
        global_feature_dim=global_dim,
        hidden_dim=cfg.lite_v2.selector_hidden_dim,
        layers=cfg.lite_v2.selector_layers,
    )
    selector = selector.to(device)
    selector.eval()
    return selector


def _select_action_pair(
    *,
    cfg: Config,
    model: GraphPolicyNetwork,
    selector: PairSelectorNetwork | None,
    obs,
    device: torch.device,
) -> tuple[int, int]:
    if cfg.variant.name == "lite_v2":
        if selector is None:
            raise RuntimeError("lite_v2 rollout requires selector model.")
        n_candidates = int(obs.candidate_pairs.shape[0])
        k = adaptive_top_k(
            num_candidates=n_candidates,
            ratio=cfg.lite_v2.actor_topk_ratio,
            k_min=cfg.lite_v2.actor_topk_min,
            k_max=cfg.lite_v2.actor_topk_max,
        )
        selector_logits = selector.forward_observation(obs, device=device)
        selected_idx = stable_topk_indices(selector_logits.detach().cpu().numpy(), k)
        policy_obs = slice_observation(obs, selected_idx)
        output = model.forward_observation(policy_obs, device=device)
        action_idx = int(torch.argmax(output.logits).item())
        return tuple(int(x) for x in policy_obs.candidate_pairs[action_idx])

    output = model.forward_observation(obs, device=device)
    action_idx = int(torch.argmax(output.logits).item())
    return tuple(int(x) for x in obs.candidate_pairs[action_idx])


def _edges_from_adj(adj: np.ndarray) -> List[Tuple[int, int]]:
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    return [(int(i), int(j)) for i, j in zip(rows.tolist(), cols.tolist())]


def _build_backbone_init_config(cfg: Config, args: argparse.Namespace) -> BackboneInitConfig:
    base = cfg.backbone_init
    return BackboneInitConfig(
        routing_mode=args.routing_mode if args.routing_mode is not None else base.routing_mode,
        rho_very_low=float(args.rho_very_low if args.rho_very_low is not None else base.rho_very_low),
        rho_low_mix_end=float(
            args.rho_low_mix_end if args.rho_low_mix_end is not None else base.rho_low_mix_end
        ),
        rho_low=float(args.rho_low if args.rho_low is not None else base.rho_low),
        rho_high=float(args.rho_high if args.rho_high is not None else base.rho_high),
        cayley_samples=int(args.cayley_samples if args.cayley_samples is not None else base.cayley_samples),
        cayley_degree_slack=int(
            args.cayley_degree_slack
            if args.cayley_degree_slack is not None
            else base.cayley_degree_slack
        ),
        cayley_multi_index_overlap=bool(
            args.cayley_multi_index_overlap
            if args.cayley_multi_index_overlap is not None
            else base.cayley_multi_index_overlap
        ),
        cayley_enable_index4=bool(
            args.cayley_enable_index4
            if args.cayley_enable_index4 is not None
            else base.cayley_enable_index4
        ),
        cayley_index2_overlap_high=float(
            args.cayley_index2_overlap_high
            if args.cayley_index2_overlap_high is not None
            else base.cayley_index2_overlap_high
        ),
        cayley_index3_overlap_high=float(
            args.cayley_index3_overlap_high
            if args.cayley_index3_overlap_high is not None
            else base.cayley_index3_overlap_high
        ),
        cayley_index4_overlap_low=float(
            args.cayley_index4_overlap_low
            if args.cayley_index4_overlap_low is not None
            else base.cayley_index4_overlap_low
        ),
        cayley_spectral_backend=(
            args.cayley_spectral_backend
            if args.cayley_spectral_backend is not None
            else base.cayley_spectral_backend
        ),
        cayley_eval_batch_size=int(
            args.cayley_eval_batch_size
            if args.cayley_eval_batch_size is not None
            else base.cayley_eval_batch_size
        ),
        cayley_eval_mode=(
            args.cayley_eval_mode
            if args.cayley_eval_mode is not None
            else base.cayley_eval_mode
        ),
        force_connected_backbone=bool(
            args.force_connected_backbone
            if args.force_connected_backbone is not None
            else base.force_connected_backbone
        ),
    )


def _run_episode(
    *,
    cfg: Config,
    model: GraphPolicyNetwork,
    selector: PairSelectorNetwork | None,
    env: GraphEnv,
    device: torch.device,
    n: int,
    rho_target: float,
    target_m: int,
    init_info: dict[str, Any],
    use_fast_inference: bool,
    initial_adj: np.ndarray | None,
) -> dict[str, Any]:
    if initial_adj is None:
        obs = env.reset_with_target(n=n, rho_target=rho_target)
    else:
        obs = env.reset_with_target(
            n=n,
            rho_target=rho_target,
            initial_adj=initial_adj,
            init_mode="backbone",
            init_metadata=init_info,
        )

    trajectory_lambda2: List[float] = [obs.lambda2_norm]
    done = edge_count(env.adj) >= target_m
    info: Dict[str, Any] = {
        "episode_done": done,
        "episode_return": 0.0,
        "episode_len": 0,
        "terminal_lambda2_norm": obs.lambda2_norm,
    }

    start = time.perf_counter()
    while not done:
        with torch.no_grad():
            action_pair = _select_action_pair(
                cfg=cfg,
                model=model,
                selector=selector,
                obs=obs,
                device=device,
            )
        obs, _, done, info = env.step(action_pair)
        if not use_fast_inference:
            trajectory_lambda2.append(obs.lambda2_norm)
    rl_runtime_sec = time.perf_counter() - start
    if use_fast_inference and info.get("terminal_lambda2_norm") is not None:
        trajectory_lambda2.append(float(info["terminal_lambda2_norm"]))

    return {
        "episode_len": int(info.get("episode_len") or 0),
        "terminal_lambda2_norm": float(info.get("terminal_lambda2_norm") or 0.0),
        "edges": _edges_from_adj(env.adj),
        "lambda2_norm_trajectory": trajectory_lambda2,
        "rl_runtime_sec": float(rl_runtime_sec),
        "runtime_sec": float(rl_runtime_sec),
        "runtime_total_sec": float(rl_runtime_sec + float(init_info["init_build_runtime_sec"])),
        "init_info": dict(init_info),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy rollout with trained RL policy")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--n", type=int, required=True, help="Number of nodes")
    parser.add_argument("--rho", type=float, required=True, help="Target normalized density in [0,1]")
    parser.add_argument("--seed", type=int, default=1234, help="Evaluation seed")
    parser.add_argument("--out", type=str, required=True, help="Output JSON file")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference")
    parser.add_argument(
        "--init-mode",
        type=str,
        default=None,
        choices=["path", "backbone"],
        help="Episode initialization graph mode. Defaults to config init.mode.",
    )
    parser.add_argument(
        "--routing-mode",
        default=None,
        choices=["window", "always_both", "always_envelope", "always_cayley"],
        help="Override backbone routing mode when init-mode=backbone.",
    )
    parser.add_argument("--rho-very-low", type=float, default=None)
    parser.add_argument("--rho-low-mix-end", type=float, default=None)
    parser.add_argument("--rho-low", type=float, default=None)
    parser.add_argument("--rho-high", type=float, default=None)
    parser.add_argument("--cayley-samples", type=int, default=None)
    parser.add_argument("--cayley-degree-slack", type=int, default=None)
    parser.add_argument(
        "--cayley-multi-index-overlap",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--cayley-enable-index4",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--cayley-index2-overlap-high", type=float, default=None)
    parser.add_argument("--cayley-index3-overlap-high", type=float, default=None)
    parser.add_argument("--cayley-index4-overlap-low", type=float, default=None)
    parser.add_argument(
        "--cayley-spectral-backend",
        default=None,
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--cayley-eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--cayley-eval-mode",
        default=None,
        choices=["dense", "character"],
    )
    parser.add_argument(
        "--force-connected-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    ckpt_variant = str(ckpt.get("variant", cfg.variant.name))
    if ckpt_variant != cfg.variant.name:
        raise ValueError(
            f"Checkpoint variant mismatch: checkpoint={ckpt_variant} config={cfg.variant.name}"
        )
    model = _build_model_from_config(cfg, device)
    model.load_state_dict(ckpt["model_state"])
    selector: PairSelectorNetwork | None = None
    if cfg.variant.name == "lite_v2":
        selector_state = ckpt.get("selector_state")
        if selector_state is None:
            raise ValueError("lite_v2 checkpoint is missing selector_state.")
        selector = _build_selector_from_config(cfg, device)
        selector.load_state_dict(selector_state)

    scheduler = CurriculumScheduler(cfg.curriculum)
    use_fast_inference = bool(cfg.variant.name == "lite_v2" and cfg.lite_v2.fast_inference)
    env = GraphEnv(
        env_id=0,
        scheduler=scheduler,
        top_k=cfg.env.top_k,
        dist_cap=cfg.env.dist_cap,
        terminal_bonus_coef=cfg.env.terminal_bonus_coef,
        seed=args.seed,
        rl_variant=cfg.variant.name,
        compute_spectral_each_step=not use_fast_inference,
        reward_alpha_1=cfg.env.reward_alpha_1,
        reward_alpha_2=cfg.env.reward_alpha_2,
        reward_alpha_3=cfg.env.reward_alpha_3,
        reward_eta_p=cfg.env.reward_eta_p,
        spectral_oversample=cfg.env.spectral_oversample,
        spectral_exact_reset_every=cfg.env.spectral_exact_reset_every,
        spectral_power_iters=cfg.env.spectral_power_iters,
        spectral_drift_threshold=cfg.env.spectral_drift_threshold,
        spectral_soft_band_rel=cfg.env.spectral_soft_band_rel,
        tier1_topk_dr=cfg.env.tier1_topk_dr,
        tier1_topk_spectral=cfg.env.tier1_topk_spectral,
        tier2_survivor_size=cfg.env.tier2_survivor_size,
        tier3_top=cfg.env.tier3_top,
        tier3_random=cfg.env.tier3_random,
    )

    init_mode = args.init_mode if args.init_mode is not None else cfg.init.mode
    target_m = m_target_from_rho(args.n, args.rho)
    overlap_post_selection = False
    overlap_candidate_count = 1
    overlap_evaluated_families = ""

    if init_mode != "backbone":
        init_info = normalize_init_metadata(
            default_path_init_metadata(args.n),
            n=args.n,
            init_mode="path",
        )
        run_result = _run_episode(
            cfg=cfg,
            model=model,
            selector=selector,
            env=env,
            device=device,
            n=args.n,
            rho_target=args.rho,
            target_m=target_m,
            init_info=init_info,
            use_fast_inference=use_fast_inference,
            initial_adj=None,
        )
    else:
        bb_cfg = _build_backbone_init_config(cfg, args)
        initializer = BackboneInitializer(bb_cfg)
        route_label, route_families = initializer.route_for(n=args.n, rho_target=args.rho)
        if route_label != "both":
            initial_adj, bb_meta = initializer.get_initial_adj(
                n=args.n,
                rho_target=args.rho,
                seed=args.seed,
                use_cache=False,
            )
            init_info = normalize_init_metadata(bb_meta, n=args.n, init_mode="backbone")
            run_result = _run_episode(
                cfg=cfg,
                model=model,
                selector=selector,
                env=env,
                device=device,
                n=args.n,
                rho_target=args.rho,
                target_m=target_m,
                init_info=init_info,
                use_fast_inference=use_fast_inference,
                initial_adj=initial_adj,
            )
        else:
            overlap_results: list[dict[str, Any]] = []
            overlap_errors: list[str] = []
            for family in route_families:
                try:
                    initial_adj, bb_meta = initializer.get_initial_adj_for_family(
                        n=args.n,
                        rho_target=args.rho,
                        seed=args.seed,
                        family=family,
                        use_cache=False,
                    )
                    init_info = normalize_init_metadata(bb_meta, n=args.n, init_mode="backbone")
                    if str(init_info["init_family"]) != str(family):
                        overlap_errors.append(f"{family}:returned_{init_info['init_family']}")
                        continue
                    candidate = _run_episode(
                        cfg=cfg,
                        model=model,
                        selector=selector,
                        env=env,
                        device=device,
                        n=args.n,
                        rho_target=args.rho,
                        target_m=target_m,
                        init_info=init_info,
                        use_fast_inference=use_fast_inference,
                        initial_adj=initial_adj,
                    )
                    overlap_results.append(candidate)
                except Exception as exc:
                    overlap_errors.append(f"{family}:{exc}")

            if not overlap_results:
                initial_adj, bb_meta = initializer.get_initial_adj(
                    n=args.n,
                    rho_target=args.rho,
                    seed=args.seed,
                    use_cache=False,
                )
                init_info = normalize_init_metadata(bb_meta, n=args.n, init_mode="backbone")
                run_result = _run_episode(
                    cfg=cfg,
                    model=model,
                    selector=selector,
                    env=env,
                    device=device,
                    n=args.n,
                    rho_target=args.rho,
                    target_m=target_m,
                    init_info=init_info,
                    use_fast_inference=use_fast_inference,
                    initial_adj=initial_adj,
                )
                if overlap_errors:
                    existing = str(run_result["init_info"].get("init_errors", ""))
                    run_result["init_info"]["init_errors"] = ";".join(
                        x for x in [existing, *overlap_errors] if x
                    )
            else:
                run_result = max(
                    overlap_results,
                    key=lambda item: (
                        float(item["terminal_lambda2_norm"]),
                        float(item["init_info"]["init_lambda2"]),
                        int(item["init_info"]["init_edges"]),
                    ),
                )
                overlap_post_selection = True
                overlap_candidate_count = len(overlap_results)
                overlap_evaluated_families = ",".join(
                    str(item["init_info"]["init_family"]) for item in overlap_results
                )
                run_result["rl_runtime_sec"] = float(
                    max(float(item["rl_runtime_sec"]) for item in overlap_results)
                )
                run_result["runtime_sec"] = float(run_result["rl_runtime_sec"])
                run_result["runtime_total_sec"] = float(
                    max(float(item["runtime_total_sec"]) for item in overlap_results)
                )
                run_result["init_info"]["init_build_runtime_sec"] = float(
                    max(
                        float(item["init_info"]["init_build_runtime_sec"])
                        for item in overlap_results
                    )
                )
                run_result["init_info"]["init_route_label"] = "both"
                existing = str(run_result["init_info"].get("init_errors", ""))
                run_result["init_info"]["init_errors"] = ";".join(
                    x for x in [existing, *overlap_errors] if x
                )
            if not overlap_evaluated_families and overlap_results:
                overlap_evaluated_families = ",".join(
                    str(item["init_info"]["init_family"]) for item in overlap_results
                )

    init_info = normalize_init_metadata(run_result["init_info"], n=args.n, init_mode=init_mode)

    result = {
        "rl_variant": cfg.variant.name,
        "n": int(args.n),
        "rho_target": float(args.rho),
        "m_target": int(target_m),
        "episode_len": int(run_result["episode_len"]),
        "terminal_lambda2_norm": float(run_result["terminal_lambda2_norm"]),
        "edges": run_result["edges"],
        "lambda2_norm_trajectory": run_result["lambda2_norm_trajectory"],
        "init_mode": init_info["init_mode"],
        "init_route_label": init_info["init_route_label"],
        "init_family": init_info["init_family"],
        "init_edges": init_info["init_edges"],
        "init_lambda2": init_info["init_lambda2"],
        "init_build_runtime_sec": init_info["init_build_runtime_sec"],
        "init_cache_hit": init_info["init_cache_hit"],
        "init_selected_index": init_info["init_selected_index"],
        "init_errors": init_info["init_errors"],
        "rl_runtime_sec": float(run_result["rl_runtime_sec"]),
        "runtime_sec": float(run_result["runtime_sec"]),  # Backward-compatible alias.
        "runtime_total_sec": float(run_result["runtime_total_sec"]),
        "overlap_post_completion_selection": bool(overlap_post_selection),
        "overlap_candidate_count": int(overlap_candidate_count),
        "overlap_evaluated_families": overlap_evaluated_families or str(init_info["init_family"]),
        "use_fast_inference": use_fast_inference,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[rollout] wrote result to {out_path}", flush=True)


if __name__ == "__main__":
    main()
