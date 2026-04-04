from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from algebraic_connectivity.config import load_config  # noqa: E402
from algebraic_connectivity.eval import evaluate_density_sweep, write_csv  # noqa: E402
from algebraic_connectivity.graph_core import max_edges  # noqa: E402


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def nontrivial_m_points(n: int, target_points: int) -> List[int]:
    n_i = int(n)
    m_lo = int(n_i - 1)
    m_hi = int(max_edges(n_i))
    all_nontrivial = list(range(m_lo + 1, m_hi))
    if len(all_nontrivial) <= int(target_points):
        return all_nontrivial

    idx = np.linspace(0, len(all_nontrivial) - 1, int(target_points), dtype=np.float64)
    idx_i = np.round(idx).astype(np.int64)
    idx_u = sorted(set(int(i) for i in idx_i.tolist()))
    if len(idx_u) < int(target_points):
        used = set(idx_u)
        for i in range(len(all_nontrivial)):
            if i not in used:
                idx_u.append(i)
                used.add(i)
            if len(idx_u) >= int(target_points):
                break
        idx_u = sorted(idx_u[: int(target_points)])
    return [int(all_nontrivial[i]) for i in idx_u]


def rho_conn_from_m(n: int, m: int) -> float:
    n_i = int(n)
    m_lo = int(n_i - 1)
    m_hi = int(max_edges(n_i))
    den = int(m_hi - m_lo)
    if den <= 0:
        return 0.0
    return float((int(m) - m_lo) / float(den))


def methods_for_n(n: int) -> List[str]:
    methods = ["path_mac", "path_mdmd", "path_fvg", "path_wts", "path_kopt"]
    if int(n) <= 32:
        methods.append("path_sdps")
    if int(n) == 8:
        methods.append("global_oa")  # OA-PMC-like global OA run
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_values", type=str, default="8,16,32,64,128")
    parser.add_argument("--points_per_n", type=int, default=100, help="Non-trivial density points per n when feasible.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--progress_every", type=int, default=200)
    parser.add_argument(
        "--out_csv",
        type=str,
        default="logs/baseline_comparison/eval_big_type123_raw.csv",
    )
    parser.add_argument(
        "--out_meta",
        type=str,
        default="logs/baseline_comparison/eval_big_type123_raw.meta.json",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    n_values = parse_int_list(args.n_values)
    if not n_values:
        raise ValueError("n_values is empty")

    points_per_n = max(1, int(args.points_per_n))
    repeats = max(1, int(args.repeats))
    progress_every = max(1, int(args.progress_every))

    plan: Dict[int, Dict[str, object]] = {}
    total_jobs = 0
    for n in n_values:
        m_points = nontrivial_m_points(n, points_per_n)
        methods = methods_for_n(n)
        jobs_n = len(m_points) * len(methods) * repeats
        total_jobs += jobs_n
        plan[int(n)] = {
            "n": int(n),
            "m_points": len(m_points),
            "m_min": int(min(m_points)),
            "m_max": int(max(m_points)),
            "methods": methods,
            "method_count": len(methods),
            "repeats": repeats,
            "jobs_n": int(jobs_n),
        }

    print("Big Experiment Plan")
    print(json.dumps({"n_values": n_values, "points_per_n": points_per_n, "repeats": repeats, "total_jobs": total_jobs, "plan": plan}, indent=2))
    if args.dry_run:
        return

    cfg = load_config(None)
    all_rows: List[Dict[str, float | int | str]] = []

    common_params = {
        # MDMD
        "mdmd_seed": 0,
        "mdmd_random_tie": False,
        # WTS
        "wts_seed": 0,
        "wts_iterations": 1000,
        "wts_tabu_tenure": 100,
        "wts_max_old_edges_per_iter": 8,
        "wts_neighbor_sample_per_old": 3,
        "wts_random_jump_per_old": 1,
        "wts_init_mode": "random",
        # MAC
        "mac_fw_iters": 50,
        "mac_duality_gap_tol": 1e-6,
        "mac_init_mode": "fiedler_topk",
        "mac_seed": 0,
        # K-opt
        "kopt_k": 2,
        "kopt_m": 10,
        "kopt_max_rounds": 10,
        "kopt_combo_cap_add": 500,
        "kopt_combo_cap_del": 500,
        "kopt_seed": 0,
        # SDP-step
        "sdp_solver": "MOSEK",
        "sdp_max_iters": 50_000,
        "sdp_eps": 1e-6,
        "sdp_verbose": False,
        # OA (global)
        "oa_solver": "GUROBI",
        "oa_max_iters": 60,
        "oa_tol": 1e-6,
        "oa_verbose": False,
        "oa_warm_start_fvg": True,
    }

    for n in n_values:
        m_points = nontrivial_m_points(n, points_per_n)
        rho_values = [rho_conn_from_m(n, m) for m in m_points]
        methods = methods_for_n(n)
        print(
            f"[run] n={n}, methods={methods}, nontrivial_points={len(m_points)}, repeats={repeats}",
            flush=True,
        )
        rows_n = evaluate_density_sweep(
            n_values=[int(n)],
            cfg=cfg,
            methods=methods,
            rho_values=rho_values,
            repeats=repeats,
            progress=True,
            progress_every=progress_every,
            **common_params,
        )
        # Safety filter: keep only non-trivial target points.
        m_set = set(int(m) for m in m_points)
        rows_n = [r for r in rows_n if int(r["m"]) in m_set]
        all_rows.extend(rows_n)
        print(f"[run] done n={n}, rows_kept={len(rows_n)}", flush=True)

    out_csv = (ROOT / args.out_csv) if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
    out_meta = (ROOT / args.out_meta) if not Path(args.out_meta).is_absolute() else Path(args.out_meta)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    write_csv(all_rows, str(out_csv))
    meta = {
        "name": "big_type123_unified",
        "n_values": n_values,
        "points_per_n_requested": points_per_n,
        "repeats": repeats,
        "total_rows": len(all_rows),
        "plan": plan,
        "params": common_params,
        "notes": [
            "Non-trivial points only: m in [n, C(n,2)-1].",
            "If non-trivial count < points_per_n (e.g., n=8), use all available points.",
            "Methods by n: base(mac,mdmd,fvg,wts,kopt), +sdp-step for n<=32, +global_oa for n=8.",
        ],
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()

