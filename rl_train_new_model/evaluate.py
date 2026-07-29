from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from .backbone_init import BackboneInitializer, default_path_init_metadata, normalize_init_metadata
from .checkpoint import load_checkpoint
from .config import BackboneInitConfig, Config, load_config
from .curriculum import CurriculumScheduler
from .determinism import apply_determinism
from .env import GraphEnv
from .graph_math import edge_count, m_target_from_rho, normalized_density
from .model import GraphPolicyNetwork, feature_dims_for_variant
from .selector import PairSelectorNetwork, adaptive_top_k, slice_observation, stable_topk_indices


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _linspace(start: float, end: float, count: int) -> List[float]:
    if count <= 0:
        raise ValueError("density-count must be positive.")
    if count == 1:
        return [start]
    step = (end - start) / float(count - 1)
    return [start + i * step for i in range(count)]


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
    return model.to(device)


def _build_selector_from_config(cfg: Config, device: torch.device) -> PairSelectorNetwork:
    node_dim, pair_dim, global_dim = feature_dims_for_variant(cfg.variant.name)
    selector = PairSelectorNetwork(
        node_feature_dim=node_dim,
        pair_feature_dim=pair_dim,
        global_feature_dim=global_dim,
        hidden_dim=cfg.lite_v2.selector_hidden_dim,
        layers=cfg.lite_v2.selector_layers,
    )
    return selector.to(device)


def _select_greedy_action_pair(
    cfg: Config,
    model: GraphPolicyNetwork,
    selector: PairSelectorNetwork | None,
    obs,
    device: torch.device,
) -> tuple[int, int]:
    if cfg.variant.name == "lite_v2":
        if selector is None:
            raise RuntimeError("lite_v2 evaluation requires selector model.")
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


def _parse_checkpoint_specs(specs: Sequence[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for raw in specs:
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            label, path = item.split("=", 1)
            label = label.strip()
            path_obj = Path(path.strip()).resolve()
        else:
            path_obj = Path(item).resolve()
            label = path_obj.stem
        if not label:
            raise ValueError(f"Checkpoint label cannot be empty: {raw}")
        out.append((label, path_obj))
    if not out:
        raise ValueError("At least one --checkpoint item is required.")
    return out


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


def _rollout_once(
    *,
    model: GraphPolicyNetwork,
    selector: PairSelectorNetwork | None,
    env: GraphEnv,
    n: int,
    rho_target: float,
    m_target: int,
    device: torch.device,
    cfg: Config,
    init_info: dict[str, Any],
    initial_adj=None,
) -> Dict[str, float | int | str | bool]:
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

    done = edge_count(env.adj) >= m_target
    info: Dict[str, Any] = {
        "episode_done": done,
        "episode_len": 0,
        "terminal_lambda2_norm": obs.lambda2_norm,
    }

    start = time.perf_counter()
    while not done:
        with torch.no_grad():
            action_pair = _select_greedy_action_pair(
                cfg=cfg,
                model=model,
                selector=selector,
                obs=obs,
                device=device,
            )
        obs, _, done, info = env.step(action_pair)
    rl_runtime_sec = time.perf_counter() - start

    m_final = edge_count(env.adj)
    lambda2_norm = float(info.get("terminal_lambda2_norm") or 0.0)
    lambda2 = lambda2_norm * float(max(1, n))
    rho_final = normalized_density(n, m_final)
    init_runtime = float(init_info["init_build_runtime_sec"])
    total_runtime = float(rl_runtime_sec + init_runtime)

    return {
        "m_target": int(m_target),
        "m_final": int(m_final),
        "rho_final": float(rho_final),
        "episode_len": int(info.get("episode_len") or 0),
        "terminal_lambda2_norm": float(lambda2_norm),
        "terminal_lambda2": float(lambda2),
        "rl_runtime_sec": float(rl_runtime_sec),
        "runtime_sec": float(rl_runtime_sec),  # Backward-compatible alias.
        "runtime_total_sec": total_runtime,
        "init_mode": str(init_info["init_mode"]),
        "init_route_label": str(init_info["init_route_label"]),
        "init_family": str(init_info["init_family"]),
        "init_edges": int(init_info["init_edges"]),
        "init_lambda2": float(init_info["init_lambda2"]),
        "init_build_runtime_sec": init_runtime,
        "init_cache_hit": bool(init_info["init_cache_hit"]),
        "init_selected_index": str(init_info["init_selected_index"]),
        "init_errors": str(init_info["init_errors"]),
        "overlap_post_completion_selection": False,
        "overlap_candidate_count": 1,
        "overlap_evaluated_families": str(init_info["init_family"]),
    }


def _pick_best_completed(results: Sequence[Dict[str, float | int | str | bool]]) -> Dict[str, float | int | str | bool]:
    return max(
        results,
        key=lambda item: (
            float(item["terminal_lambda2"]),
            float(item["init_lambda2"]),
            int(item["init_edges"]),
        ),
    )


def _evaluate_overlap_candidates(
    *,
    model: GraphPolicyNetwork,
    selector: PairSelectorNetwork | None,
    env: GraphEnv,
    n: int,
    rho_target: float,
    seed: int,
    device: torch.device,
    cfg: Config,
    init_mode: str = "path",
    backbone_initializer: BackboneInitializer | None = None,
) -> Dict[str, float | int | str | bool]:
    m_target = m_target_from_rho(n, rho_target)
    if init_mode != "backbone":
        init_info = normalize_init_metadata(
            default_path_init_metadata(n),
            n=n,
            init_mode="path",
        )
        return _rollout_once(
            model=model,
            selector=selector,
            env=env,
            n=n,
            rho_target=rho_target,
            m_target=m_target,
            device=device,
            cfg=cfg,
            init_info=init_info,
            initial_adj=None,
        )

    if backbone_initializer is None:
        raise ValueError("init_mode='backbone' requires a backbone initializer.")

    route_label, route_families = backbone_initializer.route_for(n=n, rho_target=rho_target)
    if route_label != "both":
        initial_adj, bb_info = backbone_initializer.get_initial_adj(
            n=n,
            rho_target=rho_target,
            seed=seed,
            use_cache=False,
        )
        init_info = normalize_init_metadata(bb_info, n=n, init_mode="backbone")
        return _rollout_once(
            model=model,
            selector=selector,
            env=env,
            n=n,
            rho_target=rho_target,
            m_target=m_target,
            device=device,
            cfg=cfg,
            init_info=init_info,
            initial_adj=initial_adj,
        )

    overlap_results: list[Dict[str, float | int | str | bool]] = []
    overlap_errors: list[str] = []
    for family in route_families:
        try:
            initial_adj, bb_info = backbone_initializer.get_initial_adj_for_family(
                n=n,
                rho_target=rho_target,
                seed=seed,
                family=family,
                use_cache=False,
            )
            init_info = normalize_init_metadata(bb_info, n=n, init_mode="backbone")
            if str(init_info["init_family"]) != str(family):
                overlap_errors.append(
                    f"{family}:returned_{init_info['init_family']}"
                )
                continue
            result = _rollout_once(
                model=model,
                selector=selector,
                env=env,
                n=n,
                rho_target=rho_target,
                m_target=m_target,
                device=device,
                cfg=cfg,
                init_info=init_info,
                initial_adj=initial_adj,
            )
            overlap_results.append(result)
        except Exception as exc:
            overlap_errors.append(f"{family}:{exc}")

    if not overlap_results:
        initial_adj, bb_info = backbone_initializer.get_initial_adj(
            n=n,
            rho_target=rho_target,
            seed=seed,
            use_cache=False,
        )
        init_info = normalize_init_metadata(bb_info, n=n, init_mode="backbone")
        fallback = _rollout_once(
            model=model,
            selector=selector,
            env=env,
            n=n,
            rho_target=rho_target,
            m_target=m_target,
            device=device,
            cfg=cfg,
            init_info=init_info,
            initial_adj=initial_adj,
        )
        if overlap_errors:
            existing = str(fallback.get("init_errors", ""))
            combined = ";".join(x for x in [existing, *overlap_errors] if x)
            fallback["init_errors"] = combined
        return fallback

    chosen = dict(_pick_best_completed(overlap_results))
    chosen["overlap_post_completion_selection"] = True
    chosen["overlap_candidate_count"] = int(len(overlap_results))
    chosen["overlap_evaluated_families"] = ",".join(
        str(item["init_family"]) for item in overlap_results
    )
    chosen["rl_runtime_sec"] = float(
        max(float(item["rl_runtime_sec"]) for item in overlap_results)
    )
    chosen["runtime_sec"] = float(chosen["rl_runtime_sec"])
    chosen["init_build_runtime_sec"] = float(
        max(float(item["init_build_runtime_sec"]) for item in overlap_results)
    )
    chosen["runtime_total_sec"] = float(
        max(float(item["runtime_total_sec"]) for item in overlap_results)
    )
    chosen["init_route_label"] = "both"
    existing_errors = str(chosen.get("init_errors", ""))
    chosen["init_errors"] = ";".join(x for x in [existing_errors, *overlap_errors] if x)
    return chosen


def _iter_grid(
    *,
    n_values: Iterable[int],
    densities: Iterable[float],
    seeds: Iterable[int],
) -> Iterable[Tuple[int, float, int]]:
    for n in n_values:
        for rho in densities:
            for seed in seeds:
                yield int(n), float(rho), int(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL checkpoints over an (n, density) grid."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help=(
            "Checkpoint spec. Use either '/abs/path/ckpt.pt' or "
            "'label=/abs/path/ckpt.pt'. Repeat flag for multiple checkpoints."
        ),
    )
    parser.add_argument("--n-values", type=str, required=True, help="Comma list, e.g. 32,64")
    parser.add_argument("--densities", type=str, default=None, help="Comma list of rho values")
    parser.add_argument("--density-count", type=int, default=None, help="Use evenly spaced rho grid")
    parser.add_argument("--density-min", type=float, default=0.0)
    parser.add_argument("--density-max", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="1234", help="Comma list of eval seeds")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N rollouts (and first/last).",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable progress prints.",
    )
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_determinism(cfg.determinism)

    checkpoint_specs = _parse_checkpoint_specs(args.checkpoint)
    n_values = _parse_int_list(args.n_values)
    seeds = _parse_int_list(args.seeds)
    if args.density_count is not None:
        densities = _linspace(args.density_min, args.density_max, args.density_count)
    elif args.densities is not None:
        densities = _parse_float_list(args.densities)
    else:
        raise ValueError("Provide either --densities or --density-count.")

    init_mode = args.init_mode if args.init_mode is not None else cfg.init.mode
    if init_mode not in {"path", "backbone"}:
        raise ValueError(f"Unsupported init_mode: {init_mode}")

    device = torch.device(args.device)
    scheduler = CurriculumScheduler(cfg.curriculum)
    backbone_cfg = _build_backbone_init_config(cfg, args) if init_mode == "backbone" else None
    backbone_initializer = BackboneInitializer(backbone_cfg) if backbone_cfg is not None else None

    rows: List[Dict[str, str | int | float | bool]] = []
    total_rollouts = len(checkpoint_specs) * len(n_values) * len(densities) * len(seeds)
    completed_rollouts = 0
    eval_start = time.perf_counter()
    progress_every = max(1, int(args.progress_every))
    use_fast_inference = bool(cfg.variant.name == "lite_v2" and cfg.lite_v2.fast_inference)

    for ckpt_label, ckpt_path in checkpoint_specs:
        checkpoint = load_checkpoint(str(ckpt_path), map_location=device)
        ckpt_variant = str(checkpoint.get("variant", cfg.variant.name))
        if ckpt_variant != cfg.variant.name:
            raise ValueError(
                "Checkpoint variant mismatch for evaluation: "
                f"checkpoint={ckpt_variant} config={cfg.variant.name}"
            )
        model = _build_model_from_config(cfg, device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        selector: PairSelectorNetwork | None = None
        if cfg.variant.name == "lite_v2":
            selector_state = checkpoint.get("selector_state")
            if selector_state is None:
                raise ValueError(f"Checkpoint {ckpt_path} is missing selector_state for lite_v2.")
            selector = _build_selector_from_config(cfg, device)
            selector.load_state_dict(selector_state)
            selector.eval()
        env_by_seed: dict[int, GraphEnv] = {}

        ckpt_algo = str(checkpoint.get("algo", "unknown"))
        ckpt_step = int(checkpoint.get("trainer_state", {}).get("global_env_steps", -1))

        for n, rho, seed in _iter_grid(n_values=n_values, densities=densities, seeds=seeds):
            env = env_by_seed.get(seed)
            if env is None:
                env = GraphEnv(
                    env_id=0,
                    scheduler=scheduler,
                    top_k=cfg.env.top_k,
                    dist_cap=cfg.env.dist_cap,
                    terminal_bonus_coef=cfg.env.terminal_bonus_coef,
                    seed=seed,
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
                env_by_seed[seed] = env
            result = _evaluate_overlap_candidates(
                model=model,
                selector=selector,
                env=env,
                n=n,
                rho_target=rho,
                seed=seed,
                device=device,
                cfg=cfg,
                init_mode=init_mode,
                backbone_initializer=backbone_initializer,
            )
            rows.append(
                {
                    "checkpoint_label": ckpt_label,
                    "checkpoint_path": str(ckpt_path),
                    "checkpoint_algo": ckpt_algo,
                    "checkpoint_global_env_steps": ckpt_step,
                    "config_path": str(Path(args.config).resolve()),
                    "rl_variant": cfg.variant.name,
                    "n": n,
                    "rho_target": rho,
                    "seed": seed,
                    "m_target": int(result["m_target"]),
                    "m_final": int(result["m_final"]),
                    "rho_final": float(result["rho_final"]),
                    "episode_len": int(result["episode_len"]),
                    "terminal_lambda2_norm": float(result["terminal_lambda2_norm"]),
                    "terminal_lambda2": float(result["terminal_lambda2"]),
                    "rl_runtime_sec": float(result["rl_runtime_sec"]),
                    "runtime_sec": float(result["runtime_sec"]),
                    "runtime_total_sec": float(result["runtime_total_sec"]),
                    "init_mode": str(result["init_mode"]),
                    "init_route_label": str(result["init_route_label"]),
                    "init_family": str(result["init_family"]),
                    "init_edges": int(result["init_edges"]),
                    "init_lambda2": float(result["init_lambda2"]),
                    "init_build_runtime_sec": float(result["init_build_runtime_sec"]),
                    "init_cache_hit": bool(result["init_cache_hit"]),
                    "init_selected_index": str(result["init_selected_index"]),
                    "init_errors": str(result["init_errors"]),
                    "overlap_post_completion_selection": bool(
                        result["overlap_post_completion_selection"]
                    ),
                    "overlap_candidate_count": int(result["overlap_candidate_count"]),
                    "overlap_evaluated_families": str(result["overlap_evaluated_families"]),
                    "top_k": cfg.env.top_k,
                    "dist_cap": cfg.env.dist_cap,
                }
            )
            completed_rollouts += 1
            if not args.quiet and (
                completed_rollouts == 1
                or completed_rollouts % progress_every == 0
                or completed_rollouts == total_rollouts
            ):
                elapsed = time.perf_counter() - eval_start
                speed = completed_rollouts / max(elapsed, 1e-9)
                remaining = total_rollouts - completed_rollouts
                eta_sec = remaining / max(speed, 1e-9)
                pct = 100.0 * completed_rollouts / max(1, total_rollouts)
                print(
                    "[evaluate] "
                    f"{completed_rollouts}/{total_rollouts} ({pct:.1f}%) "
                    f"elapsed={elapsed:.1f}s eta={eta_sec:.1f}s "
                    f"ckpt={ckpt_label} n={n} rho={rho:.4f} seed={seed}",
                    flush=True,
                )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint_label",
        "checkpoint_path",
        "checkpoint_algo",
        "checkpoint_global_env_steps",
        "config_path",
        "rl_variant",
        "n",
        "rho_target",
        "seed",
        "m_target",
        "m_final",
        "rho_final",
        "episode_len",
        "terminal_lambda2_norm",
        "terminal_lambda2",
        "rl_runtime_sec",
        "runtime_sec",
        "runtime_total_sec",
        "init_mode",
        "init_route_label",
        "init_family",
        "init_edges",
        "init_lambda2",
        "init_build_runtime_sec",
        "init_cache_hit",
        "init_selected_index",
        "init_errors",
        "overlap_post_completion_selection",
        "overlap_candidate_count",
        "overlap_evaluated_families",
        "top_k",
        "dist_cap",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "config": str(Path(args.config).resolve()),
        "checkpoints": [{label: str(path)} for label, path in checkpoint_specs],
        "n_values": list(n_values),
        "densities": list(densities),
        "seeds": list(seeds),
        "determinism": asdict(cfg.determinism),
        "variant": asdict(cfg.variant),
        "lite_v2": asdict(cfg.lite_v2),
        "use_fast_inference": use_fast_inference,
        "init_mode": init_mode,
        "backbone_init": asdict(backbone_cfg) if backbone_cfg else None,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total_elapsed = time.perf_counter() - eval_start
    print(f"[evaluate] wrote {len(rows)} rows to {out_path}", flush=True)
    print(f"[evaluate] wrote metadata to {meta_path}", flush=True)
    if not args.quiet:
        print(f"[evaluate] total elapsed: {total_elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
