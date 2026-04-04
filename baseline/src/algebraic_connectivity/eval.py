from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .baseline_manager import FrozenModelPolicy
from .baselines import (
    bisection_greedy_complete,
    edge_rewiring_complete,
    er_greedy_complete,
    fiedler_greedy_complete,
    fiedler_greedy_complete_singlevec,
    kopt_exchange_complete,
    lambda2_greedy_complete,
    local_global_maximizer_complete,
    mac_complete,
    mch_complete,
    mdmd_complete,
    min_laplacian_energy_complete,
    oa_exact_complete,
    oa_exact_global,
    random_complete,
    sdp_greedy_rounding_complete,
    sdp_step_rounding_complete,
    wts_complete,
)
from .config import TrainConfig
from .graph_core import deterministic_tree_graph, edge_count, max_edges, path_graph, star_graph
from .incremental_linear_algebra import IncrementalStateConfig
from .model import ModelConfig, PolicyValueNet
from .skeleton import (
    deterministic_warm_start,
    deterministic_warm_start_expander,
    deterministic_warm_start_fib_hybrid,
    deterministic_warm_start_fib_s3_hybrid,
    deterministic_warm_start_rpartite_envelope,
    deterministic_warm_start_smallworld,
)
from .spectral import algebraic_connectivity, lambda2_subspace_edge_scores, two_smallest_nontrivial
from .train import run_policy_episode


DENSITY_SWEEP_METHODS: Dict[str, Tuple[str, str]] = {
    "path_erg": ("path", "erg"),
    "path_fvg": ("path", "fvg"),
    "path_fvg_orig": ("path", "fvg_orig"),
    "path_bisect": ("path", "bisection"),
    "path_rewire": ("path", "rewire"),
    "path_mdmd": ("path", "mdmd"),
    "path_mdmd_noec": ("path", "mdmd_noec"),
    "path_minle": ("path", "minle"),
    "path_lgm": ("path", "lgm"),
    "path_lgm_raw": ("path", "lgm_raw"),
    "path_lgm_fill": ("path", "lgm_fill"),
    "path_mac": ("path", "mac"),
    "path_l2g": ("path", "l2g"),
    "path_kopt": ("path", "kopt"),
    "path_mch": ("path", "mch"),
    "path_oa": ("path", "oa_exact"),
    "global_oa": ("path", "oa_exact_global"),
    "path_wts": ("path", "wts"),
    "path_sdpg": ("path", "sdp_greedy"),
    "path_sdps": ("path", "sdp_step"),
    "star_erg": ("star", "erg"),
    "star_fvg": ("star", "fvg"),
    "star_mdmd": ("star", "mdmd"),
    "star_mdmd_noec": ("star", "mdmd_noec"),
    "star_minle": ("star", "minle"),
    "star_lgm": ("star", "lgm"),
    "star_lgm_raw": ("star", "lgm_raw"),
    "star_lgm_fill": ("star", "lgm_fill"),
    "star_mac": ("star", "mac"),
    "star_bisect": ("star", "bisection"),
    "star_rewire": ("star", "rewire"),
    "star_l2g": ("star", "l2g"),
    "star_kopt": ("star", "kopt"),
    "star_mch": ("star", "mch"),
    "star_oa": ("star", "oa_exact"),
    "star_wts": ("star", "wts"),
    "star_sdpg": ("star", "sdp_greedy"),
    "star_sdps": ("star", "sdp_step"),
    "tree_erg": ("tree", "erg"),
    "tree_fvg": ("tree", "fvg"),
    "tree_mdmd": ("tree", "mdmd"),
    "tree_mdmd_noec": ("tree", "mdmd_noec"),
    "tree_minle": ("tree", "minle"),
    "tree_lgm": ("tree", "lgm"),
    "tree_lgm_raw": ("tree", "lgm_raw"),
    "tree_lgm_fill": ("tree", "lgm_fill"),
    "tree_mac": ("tree", "mac"),
    "tree_bisect": ("tree", "bisection"),
    "tree_rewire": ("tree", "rewire"),
    "tree_l2g": ("tree", "l2g"),
    "tree_kopt": ("tree", "kopt"),
    "tree_mch": ("tree", "mch"),
    "tree_oa": ("tree", "oa_exact"),
    "tree_wts": ("tree", "wts"),
    "tree_sdpg": ("tree", "sdp_greedy"),
    "tree_sdps": ("tree", "sdp_step"),
    "s1_erg": ("deterministic_warm_start_expander", "erg"),
    "s1_fvg": ("deterministic_warm_start_expander", "fvg"),
    "s1_bisect": ("deterministic_warm_start_expander", "bisection"),
    "s1_rewire": ("deterministic_warm_start_expander", "rewire"),
    "s1_l2g": ("deterministic_warm_start_expander", "l2g"),
    "s1_kopt": ("deterministic_warm_start_expander", "kopt"),
    "s1_mch": ("deterministic_warm_start_expander", "mch"),
    "s1_oa": ("deterministic_warm_start_expander", "oa_exact"),
    "s1_wts": ("deterministic_warm_start_expander", "wts"),
    "s1_sdpg": ("deterministic_warm_start_expander", "sdp_greedy"),
    "s1_sdps": ("deterministic_warm_start_expander", "sdp_step"),
    "s2_erg": ("deterministic_warm_start_fib_hybrid", "erg"),
    "s2_fvg": ("deterministic_warm_start_fib_hybrid", "fvg"),
    "s2_bisect": ("deterministic_warm_start_fib_hybrid", "bisection"),
    "s2_rewire": ("deterministic_warm_start_fib_hybrid", "rewire"),
    "s2_l2g": ("deterministic_warm_start_fib_hybrid", "l2g"),
    "s2_kopt": ("deterministic_warm_start_fib_hybrid", "kopt"),
    "s2_mch": ("deterministic_warm_start_fib_hybrid", "mch"),
    "s2_oa": ("deterministic_warm_start_fib_hybrid", "oa_exact"),
    "s2_wts": ("deterministic_warm_start_fib_hybrid", "wts"),
    "s2_sdpg": ("deterministic_warm_start_fib_hybrid", "sdp_greedy"),
    "s2_sdps": ("deterministic_warm_start_fib_hybrid", "sdp_step"),
    "s3_erg": ("deterministic_warm_start_fib_s3_hybrid", "erg"),
    "s3_fvg": ("deterministic_warm_start_fib_s3_hybrid", "fvg"),
    "s3_bisect": ("deterministic_warm_start_fib_s3_hybrid", "bisection"),
    "s3_rewire": ("deterministic_warm_start_fib_s3_hybrid", "rewire"),
    "s3_l2g": ("deterministic_warm_start_fib_s3_hybrid", "l2g"),
    "s3_kopt": ("deterministic_warm_start_fib_s3_hybrid", "kopt"),
    "s3_mch": ("deterministic_warm_start_fib_s3_hybrid", "mch"),
    "s3_oa": ("deterministic_warm_start_fib_s3_hybrid", "oa_exact"),
    "s3_wts": ("deterministic_warm_start_fib_s3_hybrid", "wts"),
    "s3_sdpg": ("deterministic_warm_start_fib_s3_hybrid", "sdp_greedy"),
    "s3_sdps": ("deterministic_warm_start_fib_s3_hybrid", "sdp_step"),
    "sw_erg": ("deterministic_warm_start_smallworld", "erg"),
    "sw_fvg": ("deterministic_warm_start_smallworld", "fvg"),
    "sw_bisect": ("deterministic_warm_start_smallworld", "bisection"),
    "sw_rewire": ("deterministic_warm_start_smallworld", "rewire"),
    "sw_l2g": ("deterministic_warm_start_smallworld", "l2g"),
    "sw_kopt": ("deterministic_warm_start_smallworld", "kopt"),
    "sw_mch": ("deterministic_warm_start_smallworld", "mch"),
    "sw_oa": ("deterministic_warm_start_smallworld", "oa_exact"),
    "sw_wts": ("deterministic_warm_start_smallworld", "wts"),
    "sw_sdpg": ("deterministic_warm_start_smallworld", "sdp_greedy"),
    "sw_sdps": ("deterministic_warm_start_smallworld", "sdp_step"),
    "rp_env_erg": ("deterministic_warm_start_rpartite_envelope", "erg"),
    "rp_env_base": ("deterministic_warm_start_rpartite_envelope", "identity"),
    "rp_env_fvg": ("deterministic_warm_start_rpartite_envelope", "fvg"),
    "rp_env_bisect": ("deterministic_warm_start_rpartite_envelope", "bisection"),
    "rp_env_rewire": ("deterministic_warm_start_rpartite_envelope", "rewire"),
    "rp_env_l2g": ("deterministic_warm_start_rpartite_envelope", "l2g"),
    "rp_env_kopt": ("deterministic_warm_start_rpartite_envelope", "kopt"),
    "rp_env_mch": ("deterministic_warm_start_rpartite_envelope", "mch"),
    "rp_env_oa": ("deterministic_warm_start_rpartite_envelope", "oa_exact"),
    "rp_env_wts": ("deterministic_warm_start_rpartite_envelope", "wts"),
    "rp_env_sdpg": ("deterministic_warm_start_rpartite_envelope", "sdp_greedy"),
    "rp_env_sdps": ("deterministic_warm_start_rpartite_envelope", "sdp_step"),
}


METHOD_LABELS: Dict[str, str] = {
    "path_fvg": "FVG",
    "path_mdmd": "MDMD",
    "path_wts": "WTS",
    "path_lgm": "EIG",
    "path_lgm_fill": "EIG+fill",
    "path_mac": "MAC",
    "path_sdps": "SDP-step",
    "path_kopt": "k-opt",
    "path_oa": "OA-path",
    "global_oa": "OA-exact",
    "path_bisect": "Bisection",
    "path_rewire": "Rewire",
}


def method_label(method: str, completion: str) -> str:
    lbl = METHOD_LABELS.get(str(method))
    if lbl:
        return lbl
    comp = str(completion).strip().lower()
    if comp == "lgm_fill":
        return "EIG+fill"
    if comp in ("lgm", "lgm_raw"):
        return "EIG"
    if comp == "oa_exact":
        return "OA-path"
    if comp == "oa_exact_global":
        return "OA-exact"
    if comp == "sdp_step":
        return "SDP-step"
    if comp == "sdp_greedy":
        return "SDP-greedy"
    if comp == "wts":
        return "WTS"
    if comp == "mdmd":
        return "MDMD"
    if comp == "mdmd_noec":
        return "MDMD-noEC"
    if comp == "mac":
        return "MAC"
    if comp == "kopt":
        return "k-opt"
    if comp == "l2g":
        return "L2G"
    if comp == "bisection":
        return "Bisection"
    if comp == "rewire":
        return "Rewire"
    if comp == "minle":
        return "MinLE"
    if comp == "mch":
        return "MCH"
    if comp == "erg":
        return "ERG"
    if comp == "fvg":
        return "FVG"
    return str(method)


def density_to_m(n: int, rho: float) -> int:
    rho = float(np.clip(rho, 0.0, 1.0))
    m_lo = n - 1
    m_hi = max_edges(n)
    return int(np.clip(round(m_lo + rho * (m_hi - m_lo)), m_lo, m_hi))


def _init_graph(
    n: int,
    m: int,
    cfg: TrainConfig,
    init_mode_override: str | None = None,
    init_seed_offset: int = 0,
) -> np.ndarray:
    init_mode = init_mode_override or cfg.init_mode
    if init_mode == "path":
        return path_graph(n)
    if init_mode == "star":
        return star_graph(n)
    if init_mode == "tree":
        return deterministic_tree_graph(
            n,
            seed=int(cfg.seed + 10_007 * n + int(init_seed_offset)),
        )
    if init_mode == "deterministic_warm_start":
        adj, _ = deterministic_warm_start(n, m, thresholds=cfg.features.density_thresholds)
        return adj
    if init_mode == "deterministic_warm_start_expander":
        adj, _ = deterministic_warm_start_expander(
            n,
            m,
            thresholds=cfg.features.density_thresholds,
            comm_penalty_scale=cfg.features.expander_comm_penalty_scale,
        )
        return adj
    if init_mode == "deterministic_warm_start_smallworld":
        adj, _ = deterministic_warm_start_smallworld(
            n,
            m,
            thresholds=cfg.features.density_thresholds,
            num_samples=cfg.features.smallworld_num_samples,
        )
        return adj
    if init_mode == "deterministic_warm_start_fib_hybrid":
        adj, _ = deterministic_warm_start_fib_hybrid(
            n,
            m,
            thresholds=cfg.features.density_thresholds,
            extra_fraction_by_pocket=cfg.features.fib_hybrid_fractions,
            fib_geo_weight_scale=cfg.features.fib_geo_weight_scale,
            fib_comm_penalty_scale=cfg.features.fib_comm_penalty_scale,
            fib_geo_weight_override=cfg.features.fib_geo_weight_override,
            fib_comm_weight_override=cfg.features.fib_comm_weight_override,
        )
        return adj
    if init_mode == "deterministic_warm_start_fib_s3_hybrid":
        adj, _ = deterministic_warm_start_fib_s3_hybrid(
            n,
            m,
            thresholds=cfg.features.density_thresholds,
            extra_fraction_by_pocket=cfg.features.fib_hybrid_fractions,
            fib_geo_weight_scale=cfg.features.fib_geo_weight_scale,
            fib_comm_penalty_scale=cfg.features.fib_comm_penalty_scale,
            fib_geo_weight_override=cfg.features.fib_geo_weight_override,
            fib_comm_weight_override=cfg.features.fib_comm_weight_override,
        )
        return adj
    if init_mode == "deterministic_warm_start_rpartite_envelope":
        adj, _ = deterministic_warm_start_rpartite_envelope(
            n,
            m,
            thresholds=cfg.features.density_thresholds,
            r_max=cfg.features.rpartite_r_max,
            envelope_rank=cfg.features.rpartite_envelope_rank,
            use_approx_filter=cfg.features.rpartite_use_approx_filter,
            require_acm_cert=cfg.features.rpartite_require_acm_cert,
            allow_uncertified_fallback=cfg.features.rpartite_allow_uncertified_fallback,
        )
        return adj
    raise ValueError(f"unknown init mode: {init_mode}")


def load_model(checkpoint: str, device: str = "cpu") -> PolicyValueNet:
    model = PolicyValueNet(ModelConfig()).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
    model_state = model.state_dict()
    matched = {
        k: v for k, v in state.items() if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape
    }
    model.load_state_dict(matched, strict=False)
    model.eval()
    return model


def evaluate_instances(
    instances: Sequence[Tuple[int, int]],
    cfg: TrainConfig,
    model_checkpoint: str | None = None,
    device: str = "cpu",
    random_trials: int = 3,
) -> List[Dict[str, float | int]]:
    rows: List[Dict[str, float | int]] = []
    rng = np.random.default_rng(cfg.seed + 123)
    model = None
    model_policy = None
    if model_checkpoint is not None:
        model = load_model(model_checkpoint, device=device)
        model_policy = FrozenModelPolicy(model, device=device)

    for n, m in instances:
        init_adj = _init_graph(n, m, cfg)
        row: Dict[str, float | int] = {"n": n, "m": m}
        if model_policy is not None:
            row["model"] = run_policy_episode(n, m, model_policy, cfg)
        er_adj = er_greedy_complete(init_adj.copy(), m)
        fv_adj = fiedler_greedy_complete(init_adj.copy(), m)
        row["er_greedy"] = float(algebraic_connectivity(er_adj))
        row["fiedler_greedy"] = float(algebraic_connectivity(fv_adj))
        rand_scores = []
        for _ in range(max(1, random_trials)):
            rand_adj = random_complete(init_adj.copy(), m, rng)
            rand_scores.append(algebraic_connectivity(rand_adj))
        row["random_mean"] = float(np.mean(rand_scores))
        rows.append(row)
    return rows


def _complete_with_method(
    init_adj: np.ndarray,
    m_target: int,
    cfg: TrainConfig,
    completion_method: str,
    er_top_k: int = 1,
    wts_seed: int = 0,
    wts_iterations: int = 20,
    wts_tabu_tenure: int = 16,
    wts_max_old_edges_per_iter: int = 8,
    wts_neighbor_sample_per_old: int = 3,
    wts_random_jump_per_old: int = 1,
    wts_init_mode: str = "fvg",
    l2g_candidate_cap: int = 0,
    l2g_eval_mode: str = "exact",
    l2g_eigsh_cutoff_n: int = 96,
    l2g_tie_tol: float = 1e-12,
    l2g_tie_break: str = "fvg_among_best",
    bisect_eps: float = 1e-4,
    bisect_sign_tol: float = 1e-12,
    bisect_max_iters: int = 0,
    bisect_final_eval_mode: str = "exact",
    bisect_final_eval_cap: int = 64,
    rewire_fraction: float = 0.07,
    rewire_steps: int = 0,
    rewire_order: str = "remove_then_add",
    rewire_prefill_mode: str = "fvg",
    rewire_seed: int = 0,
    mdmd_seed: int = 0,
    mdmd_random_tie: bool = False,
    mac_fw_iters: int = 20,
    mac_duality_gap_tol: float = 1e-6,
    mac_init_mode: str = "fiedler_topk",
    mac_seed: int = 0,
    kopt_k: int = 1,
    kopt_m: int = 20,
    kopt_max_rounds: int = 20,
    kopt_combo_cap_add: int = 500,
    kopt_combo_cap_del: int = 500,
    kopt_seed: int = 0,
    mch_k: int = 3,
    mch_h1: int = 5,
    mch_h2: int = 3,
    mch_oa_pool_extra: int = 40,
    mch_oa_max_iters: int = 25,
    mch_oa_tol: float = 1e-6,
    mch_oa_solver: str = "HIGHS",
    mch_oa_verbose: bool = False,
    oa_solver: str = "HIGHS",
    oa_max_iters: int = 60,
    oa_tol: float = 1e-6,
    oa_verbose: bool = False,
    oa_warm_start_fvg: bool = True,
    sdp_solver: str = "SCS",
    sdp_max_iters: int = 10_000,
    sdp_eps: float = 1e-5,
    sdp_verbose: bool = False,
) -> np.ndarray:
    if completion_method == "identity":
        # Direct baseline: use warm-start graph as-is.
        return init_adj
    if completion_method == "erg":
        state_cfg = IncrementalStateConfig(
            refresh_every_steps=cfg.spectral.refresh_every_steps,
            exact_reset_every_steps=cfg.spectral.exact_reset_every_steps,
            eigsh_cutoff_n=cfg.spectral.eigsh_cutoff_n,
        )
        return er_greedy_complete(
            init_adj,
            m_target,
            top_k=max(1, int(er_top_k)),
            state_config=state_cfg,
        )
    if completion_method == "fvg":
        return fiedler_greedy_complete(init_adj, m_target)
    if completion_method == "fvg_orig":
        return fiedler_greedy_complete_singlevec(
            init_adj,
            m_target,
            eigsh_cutoff_n=int(l2g_eigsh_cutoff_n),
        )
    if completion_method == "l2g":
        return lambda2_greedy_complete(
            init_adj,
            m_target,
            candidate_cap=int(l2g_candidate_cap),
            eval_mode=str(l2g_eval_mode).strip().lower(),
            eigsh_cutoff_n=int(l2g_eigsh_cutoff_n),
            tie_tol=float(l2g_tie_tol),
            tie_break=str(l2g_tie_break).strip().lower(),
        )
    if completion_method == "bisection":
        return bisection_greedy_complete(
            init_adj,
            m_target,
            bisect_eps=float(bisect_eps),
            bisect_sign_tol=float(bisect_sign_tol),
            bisect_max_iters=int(bisect_max_iters),
            bisect_final_eval_mode=str(bisect_final_eval_mode).strip().lower(),
            bisect_final_eval_cap=int(bisect_final_eval_cap),
            eigsh_cutoff_n=int(l2g_eigsh_cutoff_n),
        )
    if completion_method == "rewire":
        return edge_rewiring_complete(
            init_adj,
            m_target,
            rewire_fraction=float(rewire_fraction),
            rewire_steps=int(rewire_steps),
            rewire_order=str(rewire_order).strip().lower(),
            prefill_mode=str(rewire_prefill_mode).strip().lower(),
            seed=int(rewire_seed),
            eigsh_cutoff_n=int(l2g_eigsh_cutoff_n),
        )
    if completion_method == "mdmd":
        return mdmd_complete(
            init_adj,
            m_target,
            use_ec=True,
            random_tie=bool(mdmd_random_tie),
            seed=int(mdmd_seed),
        )
    if completion_method == "mdmd_noec":
        return mdmd_complete(
            init_adj,
            m_target,
            use_ec=False,
            random_tie=bool(mdmd_random_tie),
            seed=int(mdmd_seed),
        )
    if completion_method == "minle":
        return min_laplacian_energy_complete(init_adj, m_target)
    if completion_method == "lgm":
        return local_global_maximizer_complete(
            init_adj,
            m_target,
            exact_budget=False,
        )
    if completion_method == "lgm_raw":
        return local_global_maximizer_complete(
            init_adj,
            m_target,
            exact_budget=False,
        )
    if completion_method == "lgm_fill":
        return local_global_maximizer_complete(
            init_adj,
            m_target,
            exact_budget=True,
        )
    if completion_method == "mac":
        return mac_complete(
            init_adj,
            m_target,
            fw_iters=int(mac_fw_iters),
            duality_gap_tol=float(mac_duality_gap_tol),
            init_mode=str(mac_init_mode).strip().lower(),
            seed=int(mac_seed),
        )
    if completion_method == "kopt":
        return kopt_exchange_complete(
            init_adj,
            m_target,
            k=int(kopt_k),
            m=int(kopt_m),
            max_rounds=int(kopt_max_rounds),
            combo_cap_add=int(kopt_combo_cap_add),
            combo_cap_del=int(kopt_combo_cap_del),
            seed=int(kopt_seed),
        )
    if completion_method == "wts":
        return wts_complete(
            init_adj,
            m_target,
            seed=int(wts_seed),
            iterations=int(wts_iterations),
            tabu_tenure=int(wts_tabu_tenure),
            max_old_edges_per_iter=int(wts_max_old_edges_per_iter),
            neighbor_sample_per_old=int(wts_neighbor_sample_per_old),
            random_jump_per_old=int(wts_random_jump_per_old),
            init_mode=str(wts_init_mode).strip().lower(),
        )
    if completion_method == "mch":
        return mch_complete(
            init_adj,
            m_target,
            k=int(mch_k),
            h1=int(mch_h1),
            h2=int(mch_h2),
            oa_pool_extra=int(mch_oa_pool_extra),
            oa_max_iters=int(mch_oa_max_iters),
            oa_tol=float(mch_oa_tol),
            oa_solver=str(mch_oa_solver).strip(),
            oa_verbose=bool(mch_oa_verbose),
        )
    if completion_method == "oa_exact":
        return oa_exact_complete(
            init_adj,
            m_target,
            solver=str(oa_solver).strip(),
            max_iters=int(oa_max_iters),
            tol=float(oa_tol),
            verbose=bool(oa_verbose),
            warm_start_fvg=bool(oa_warm_start_fvg),
        )
    if completion_method == "oa_exact_global":
        return oa_exact_global(
            int(init_adj.shape[0]),
            m_target,
            solver=str(oa_solver).strip(),
            max_iters=int(oa_max_iters),
            tol=float(oa_tol),
            verbose=bool(oa_verbose),
            warm_start_fvg=bool(oa_warm_start_fvg),
        )
    if completion_method == "sdp_greedy":
        return sdp_greedy_rounding_complete(
            init_adj,
            m_target,
            solver=str(sdp_solver).strip(),
            max_iters=int(sdp_max_iters),
            eps=float(sdp_eps),
            verbose=bool(sdp_verbose),
        )
    if completion_method == "sdp_step":
        return sdp_step_rounding_complete(
            init_adj,
            m_target,
            solver=str(sdp_solver).strip(),
            max_iters=int(sdp_max_iters),
            eps=float(sdp_eps),
            verbose=bool(sdp_verbose),
        )
    raise ValueError(f"unknown completion_method: {completion_method}")


def evaluate_density_sweep(
    n_values: Sequence[int],
    cfg: TrainConfig,
    methods: Sequence[str] | None = None,
    rho_steps: int = 101,
    rho_values: Sequence[float] | None = None,
    er_top_k: int = 1,
    wts_seed: int = 0,
    wts_iterations: int = 20,
    wts_tabu_tenure: int = 16,
    wts_max_old_edges_per_iter: int = 8,
    wts_neighbor_sample_per_old: int = 3,
    wts_random_jump_per_old: int = 1,
    wts_init_mode: str = "fvg",
    l2g_candidate_cap: int = 0,
    l2g_eval_mode: str = "exact",
    l2g_eigsh_cutoff_n: int = 96,
    l2g_tie_tol: float = 1e-12,
    l2g_tie_break: str = "fvg_among_best",
    bisect_eps: float = 1e-4,
    bisect_sign_tol: float = 1e-12,
    bisect_max_iters: int = 0,
    bisect_final_eval_mode: str = "exact",
    bisect_final_eval_cap: int = 64,
    rewire_fraction: float = 0.07,
    rewire_steps: int = 0,
    rewire_order: str = "remove_then_add",
    rewire_prefill_mode: str = "fvg",
    rewire_seed: int = 0,
    mdmd_seed: int = 0,
    mdmd_random_tie: bool = False,
    mac_fw_iters: int = 20,
    mac_duality_gap_tol: float = 1e-6,
    mac_init_mode: str = "fiedler_topk",
    mac_seed: int = 0,
    kopt_k: int = 1,
    kopt_m: int = 20,
    kopt_max_rounds: int = 20,
    kopt_combo_cap_add: int = 500,
    kopt_combo_cap_del: int = 500,
    kopt_seed: int = 0,
    mch_k: int = 3,
    mch_h1: int = 5,
    mch_h2: int = 3,
    mch_oa_pool_extra: int = 40,
    mch_oa_max_iters: int = 25,
    mch_oa_tol: float = 1e-6,
    mch_oa_solver: str = "HIGHS",
    mch_oa_verbose: bool = False,
    oa_solver: str = "HIGHS",
    oa_max_iters: int = 60,
    oa_tol: float = 1e-6,
    oa_verbose: bool = False,
    oa_warm_start_fvg: bool = True,
    sdp_solver: str = "SCS",
    sdp_max_iters: int = 10_000,
    sdp_eps: float = 1e-5,
    sdp_verbose: bool = False,
    repeats: int = 1,
    tree_randomize_per_repeat: bool = False,
    progress: bool = False,
    progress_every: int = 100,
) -> List[Dict[str, float | int | str]]:
    """
    Evaluate warm-start + completion methods over rho in [0,1].
    rho is interpreted on the feasible connected-edge scale:
      m = round((n-1) + rho * (n(n-1)/2 - (n-1))).
    """
    if methods is None:
        methods = [
            "s1_erg",
            "s1_fvg",
            "s2_erg",
            "s2_fvg",
            "s3_erg",
            "s3_fvg",
        ]
    if rho_values is None:
        rho_arr = np.linspace(0.0, 1.0, max(2, int(rho_steps)), dtype=np.float64)
    else:
        rho_arr = np.asarray(list(rho_values), dtype=np.float64)

    rows: List[Dict[str, float | int | str]] = []
    rep_count = max(1, int(repeats))
    total_jobs = int(len(n_values) * len(rho_arr) * len(methods) * rep_count)
    done_jobs = 0
    t_all_start = time.perf_counter()

    def _progress_log(
        cur_n: int,
        cur_rho_conn: float,
        cur_density: float,
        cur_m: int,
        cur_method: str,
        cur_rep: int,
    ) -> None:
        if not bool(progress):
            return
        if total_jobs <= 0:
            return
        pct = 100.0 * float(done_jobs) / float(total_jobs)
        elapsed = max(1e-12, float(time.perf_counter() - t_all_start))
        rate = float(done_jobs) / elapsed
        rem = max(0, int(total_jobs - done_jobs))
        eta = float(rem) / max(1e-12, rate)
        print(
            f"[density_sweep] {done_jobs}/{total_jobs} ({pct:.1f}%) "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s "
            f"n={cur_n} rho_conn={cur_rho_conn:.4f} density={cur_density:.4f} "
            f"m={cur_m} method={cur_method} rep={cur_rep}",
            flush=True,
        )

    def _append_row(
        *,
        n: int,
        rho_conn: float,
        m: int,
        m_actual: int,
        rep: int,
        method: str,
        init_mode: str,
        completion: str,
        init_e: int,
        lam2: float,
        runtime_init_s: float,
        runtime_complete_s: float,
        runtime_lambda2_s: float,
    ) -> None:
        nonlocal done_jobs
        m_lo = int(n - 1)
        m_hi = int(max_edges(n))
        if m_hi > 0:
            density = float(m) / float(m_hi)
            density_actual = float(m_actual) / float(m_hi)
        else:
            density = 0.0
            density_actual = 0.0
        if m_hi > m_lo:
            rho_conn_actual = float((m_actual - m_lo) / float(m_hi - m_lo))
        else:
            rho_conn_actual = 0.0
        rows.append(
            {
                "n": int(n),
                "density": float(density),
                "rho": float(density),
                "m": int(m),
                "density_actual": float(density_actual),
                "rho_actual": float(density_actual),
                "rho_conn": float(rho_conn),
                "rho_conn_actual": float(rho_conn_actual),
                "m_max": int(m_hi),
                "m_actual": int(m_actual),
                "repeat_idx": int(rep),
                "method": method,
                "method_label": method_label(method, completion),
                "warm_start_mode": init_mode,
                "completion_method": completion,
                "init_edges": int(init_e),
                "residual_budget": int(m - init_e),
                "lambda2": float(lam2),
                "runtime_init_s": float(runtime_init_s),
                "runtime_complete_s": float(runtime_complete_s),
                "runtime_lambda2_s": float(runtime_lambda2_s),
                "runtime_total_s": float(runtime_init_s + runtime_complete_s + runtime_lambda2_s),
            }
        )
        done_jobs += 1
        pe = max(1, int(progress_every))
        if done_jobs == 1 or done_jobs == total_jobs or (done_jobs % pe) == 0:
            _progress_log(
                cur_n=int(n),
                cur_rho_conn=float(rho_conn),
                cur_density=float(density),
                cur_m=int(m),
                cur_method=str(method),
                cur_rep=int(rep),
            )

    def _can_use_fvg_trajectory(init_mode: str, completion: str) -> bool:
        # For these init modes, the initial graph is independent of target m.
        return (completion in ("fvg", "fvg_orig")) and (init_mode in ("path", "star", "tree"))

    def _can_use_mdmd_trajectory(init_mode: str, completion: str) -> bool:
        # MDMD is also sequential edge-addition. A shared trajectory is valid
        # when ties are deterministic (random_tie=False), matching per-m runs.
        return (
            (completion in ("mdmd", "mdmd_noec"))
            and (init_mode in ("path", "star", "tree"))
            and (not bool(mdmd_random_tie))
        )

    for n in n_values:
        rho_meta: List[Tuple[float, int]] = []
        for rho in rho_arr:
            rho_f = float(np.clip(rho, 0.0, 1.0))
            m = int(density_to_m(n, rho_f))
            rho_meta.append((rho_f, m))

        for method in methods:
            if method not in DENSITY_SWEEP_METHODS:
                raise ValueError(
                    f"unknown method '{method}'. supported: {sorted(DENSITY_SWEEP_METHODS.keys())}"
                )
            init_mode, completion = DENSITY_SWEEP_METHODS[method]
            for rep in range(rep_count):
                seed_off = int(9_973 * rep)
                tree_seed_off = int(seed_off) if bool(tree_randomize_per_repeat) and (init_mode == "tree") else 0
                if _can_use_fvg_trajectory(init_mode, completion):
                    # Single sequential trajectory; derive per-(rho,m) metrics from it.
                    t0 = time.perf_counter()
                    # m is irrelevant for path/star/tree initialization.
                    init_adj = _init_graph(
                        n,
                        n - 1,
                        cfg,
                        init_mode_override=init_mode,
                        init_seed_offset=int(tree_seed_off),
                    )
                    t1 = time.perf_counter()
                    init_e = int(edge_count(init_adj))

                    work = init_adj.copy()
                    n_i = int(work.shape[0])
                    iu, iv = np.triu_indices(n_i, k=1)
                    available = ~work[iu, iv]
                    cur_edges = int(init_e)
                    comp_accum = 0.0

                    uniq_m = sorted({int(m) for _, m in rho_meta})
                    per_m: Dict[int, Tuple[int, float, float, float]] = {}
                    # value: (m_actual, lam2, runtime_complete_s, runtime_lambda2_s)
                    for m_t in uniq_m:
                        while cur_edges < int(m_t):
                            cand_idx = np.flatnonzero(available)
                            if int(cand_idx.size) == 0:
                                break
                            uv = np.stack([iu[cand_idx], iv[cand_idx]], axis=1).astype(
                                np.int64, copy=False
                            )
                            t_step0 = time.perf_counter()
                            if completion == "fvg":
                                _, scores = lambda2_subspace_edge_scores(work, uv)
                            else:
                                _, phi2, _ = two_smallest_nontrivial(
                                    work,
                                    eigsh_cutoff_n=int(l2g_eigsh_cutoff_n),
                                )
                                scores = (phi2[uv[:, 0]] - phi2[uv[:, 1]]) ** 2
                            pick_local = int(np.argmax(scores))
                            pick = int(cand_idx[pick_local])
                            u = int(iu[pick])
                            v = int(iv[pick])
                            work[u, v] = True
                            work[v, u] = True
                            available[pick] = False
                            cur_edges += 1
                            comp_accum += float(time.perf_counter() - t_step0)

                        t_l0 = time.perf_counter()
                        lam2 = float(algebraic_connectivity(work))
                        t_l1 = time.perf_counter()
                        per_m[int(m_t)] = (
                            int(cur_edges),
                            float(lam2),
                            float(comp_accum),
                            float(t_l1 - t_l0),
                        )

                    for rho_f, m in rho_meta:
                        m_actual, lam2, rt_comp, rt_lam = per_m[int(m)]
                        _append_row(
                            n=int(n),
                            rho_conn=float(rho_f),
                            m=int(m),
                            m_actual=int(m_actual),
                            rep=int(rep),
                            method=str(method),
                            init_mode=str(init_mode),
                            completion=str(completion),
                            init_e=int(init_e),
                            lam2=float(lam2),
                            runtime_init_s=float(t1 - t0),
                            runtime_complete_s=float(rt_comp),
                            runtime_lambda2_s=float(rt_lam),
                        )
                    continue

                if _can_use_mdmd_trajectory(init_mode, completion):
                    # Single sequential trajectory for deterministic MDMD.
                    t0 = time.perf_counter()
                    init_adj = _init_graph(
                        n,
                        n - 1,
                        cfg,
                        init_mode_override=init_mode,
                        init_seed_offset=int(tree_seed_off),
                    )
                    t1 = time.perf_counter()
                    init_e = int(edge_count(init_adj))

                    work = init_adj.copy()
                    comp_accum = 0.0
                    use_ec = bool(completion == "mdmd")

                    uniq_m = sorted({int(m) for _, m in rho_meta})
                    per_m: Dict[int, Tuple[int, float, float, float]] = {}
                    # value: (m_actual, lam2, runtime_complete_s, runtime_lambda2_s)
                    for m_t in uniq_m:
                        t_step0 = time.perf_counter()
                        work = mdmd_complete(
                            work,
                            int(m_t),
                            use_ec=bool(use_ec),
                            random_tie=False,
                            # seed is irrelevant when random_tie=False
                            seed=int(mdmd_seed),
                        )
                        comp_accum += float(time.perf_counter() - t_step0)

                        t_l0 = time.perf_counter()
                        lam2 = float(algebraic_connectivity(work))
                        t_l1 = time.perf_counter()
                        per_m[int(m_t)] = (
                            int(edge_count(work)),
                            float(lam2),
                            float(comp_accum),
                            float(t_l1 - t_l0),
                        )

                    for rho_f, m in rho_meta:
                        m_actual, lam2, rt_comp, rt_lam = per_m[int(m)]
                        _append_row(
                            n=int(n),
                            rho_conn=float(rho_f),
                            m=int(m),
                            m_actual=int(m_actual),
                            rep=int(rep),
                            method=str(method),
                            init_mode=str(init_mode),
                            completion=str(completion),
                            init_e=int(init_e),
                            lam2=float(lam2),
                            runtime_init_s=float(t1 - t0),
                            runtime_complete_s=float(rt_comp),
                            runtime_lambda2_s=float(rt_lam),
                        )
                    continue

                # Generic path: evaluate each (rho,m) independently.
                for rho_f, m in rho_meta:
                    t0 = time.perf_counter()
                    init_adj = _init_graph(
                        n,
                        m,
                        cfg,
                        init_mode_override=init_mode,
                        init_seed_offset=int(tree_seed_off),
                    )
                    t1 = time.perf_counter()

                    init_e = edge_count(init_adj)
                    final_adj = _complete_with_method(
                        init_adj.copy(),
                        m_target=m,
                        cfg=cfg,
                        completion_method=completion,
                        er_top_k=er_top_k,
                        wts_seed=int(wts_seed + 100_003 * int(n) + int(m) + seed_off),
                        wts_iterations=wts_iterations,
                        wts_tabu_tenure=wts_tabu_tenure,
                        wts_max_old_edges_per_iter=wts_max_old_edges_per_iter,
                        wts_neighbor_sample_per_old=wts_neighbor_sample_per_old,
                        wts_random_jump_per_old=wts_random_jump_per_old,
                        wts_init_mode=wts_init_mode,
                        l2g_candidate_cap=l2g_candidate_cap,
                        l2g_eval_mode=l2g_eval_mode,
                        l2g_eigsh_cutoff_n=l2g_eigsh_cutoff_n,
                        l2g_tie_tol=l2g_tie_tol,
                        l2g_tie_break=l2g_tie_break,
                        bisect_eps=bisect_eps,
                        bisect_sign_tol=bisect_sign_tol,
                        bisect_max_iters=bisect_max_iters,
                        bisect_final_eval_mode=bisect_final_eval_mode,
                        bisect_final_eval_cap=bisect_final_eval_cap,
                        rewire_fraction=rewire_fraction,
                        rewire_steps=rewire_steps,
                        rewire_order=rewire_order,
                        rewire_prefill_mode=rewire_prefill_mode,
                        rewire_seed=int(rewire_seed + 11_573 * int(n) + int(m) + seed_off),
                        mdmd_seed=int(mdmd_seed + 29_993 * int(n) + int(m) + seed_off),
                        mdmd_random_tie=mdmd_random_tie,
                        mac_fw_iters=mac_fw_iters,
                        mac_duality_gap_tol=mac_duality_gap_tol,
                        mac_init_mode=mac_init_mode,
                        mac_seed=int(mac_seed + 53_633 * int(n) + int(m) + seed_off),
                        kopt_k=kopt_k,
                        kopt_m=kopt_m,
                        kopt_max_rounds=kopt_max_rounds,
                        kopt_combo_cap_add=kopt_combo_cap_add,
                        kopt_combo_cap_del=kopt_combo_cap_del,
                        kopt_seed=int(kopt_seed + 67_097 * int(n) + int(m) + seed_off),
                        mch_k=mch_k,
                        mch_h1=mch_h1,
                        mch_h2=mch_h2,
                        mch_oa_pool_extra=mch_oa_pool_extra,
                        mch_oa_max_iters=mch_oa_max_iters,
                        mch_oa_tol=mch_oa_tol,
                        mch_oa_solver=mch_oa_solver,
                        mch_oa_verbose=mch_oa_verbose,
                        oa_solver=oa_solver,
                        oa_max_iters=oa_max_iters,
                        oa_tol=oa_tol,
                        oa_verbose=oa_verbose,
                        oa_warm_start_fvg=oa_warm_start_fvg,
                        sdp_solver=sdp_solver,
                        sdp_max_iters=sdp_max_iters,
                        sdp_eps=sdp_eps,
                        sdp_verbose=sdp_verbose,
                    )
                    t2 = time.perf_counter()

                    t_l0 = time.perf_counter()
                    lam2 = float(algebraic_connectivity(final_adj))
                    t_l1 = time.perf_counter()

                    _append_row(
                        n=int(n),
                        rho_conn=float(rho_f),
                        m=int(m),
                        m_actual=int(edge_count(final_adj)),
                        rep=int(rep),
                        method=str(method),
                        init_mode=str(init_mode),
                        completion=str(completion),
                        init_e=int(init_e),
                        lam2=float(lam2),
                        runtime_init_s=float(t1 - t0),
                        runtime_complete_s=float(t2 - t1),
                        runtime_lambda2_s=float(t_l1 - t_l0),
                    )
    return rows


def write_csv(rows: Sequence[Dict[str, float | int]], path: str) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
