# lite_trials_new — three-way spectral-tracker comparison

Head-to-head lite trials between three RL agents. "Lite" means fewer training
episodes, not a reduced pipeline: each arm gets a real PPO training loop with
curriculum, periodic evaluation and checkpointing, capped at 20K environment
steps.

## Arms

| tag | code base | spectral tracking |
|---|---|---|
| `old_model` | `../old_model/rl_train` | full `eigh` every step. Reward is `Δλ₂/n` + terminal bonus — no blend terms exist in its `EnvConfig` |
| `analytic` | `analytic_tractable_test_deepseek` | full eigenbasis + exact secular root; λ₂ tracked to ~1e-15 |
| `subspace` | `rl_train_new_model` | bottom-q invariant subspace by block power iteration; λ₂ approximate but with a certified two-sided bracket |

Not a sweep. Both analytic arms are pinned to the **same** blend
α = (0.15, 0.15, 0.70) so the reward does not confound the comparison —
`analytic`'s own default is (0.10, 0.20, 0.70) and is overridden here.
`old_model` has no α at all.

Everything else is held identical across arms: curriculum (n=8–12), network
sizes, PPO hyperparameters, seed (1234), `num_envs`, `top_k`, `dist_cap`,
`terminal_bonus_coef`, `init.mode=path`, and the evaluation grid.

`evaluate.py` is byte-identical between `old_model` and `analytic`.
`rl_train_new_model`'s copy differs by exactly 16 lines, all of them forwarding
its extra tracker/candidacy config into the env constructor — the protocol
itself (rollout loop, density grid, metrics, CSV schema) is unchanged. So the
evaluation is genuinely shared across all three arms.

## Run it

From the project root:

```bash
python -m lite_trials_new.run_all
```

That is train → evaluate → visualize. The stages are also individually runnable:

```bash
python -m lite_trials_new.compare_train                 # ~2-4 min per arm
python -m lite_trials_new.compare_experiment            # the long stage
python -m lite_trials_new.compare_visualize
```

Useful flags:

```bash
# quick smoke test of the whole pipeline (~2 min total)
python -m lite_trials_new.run_all --max-env-steps 2048 --n-values 12 --points-per-n 3 --repeats 1

# re-evaluate / re-plot without retraining
python -m lite_trials_new.run_all --skip-train

# see the evaluation plan and rollout count without running it
python -m lite_trials_new.compare_experiment --dry-run

# one arm only
python -m lite_trials_new.compare_train --arms analytic
```

### Resuming an interrupted run

`--resume` picks up where a run stopped, and works on either stage:

```bash
python -m lite_trials_new.run_all --resume              # both stages
python -m lite_trials_new.compare_train --resume        # training only
python -m lite_trials_new.compare_experiment --resume   # evaluation only
```

**Training** continues each arm from its `runs/<tag>/checkpoints/last.pt`, which
carries optimizer, LR scheduler, env manager, rollout buffer and RNG state — so
the resumed leg continues the same run rather than starting a second one at step
0. Arms with no `last.pt` train from scratch. An arm already at
`--max-env-steps` exits immediately, so raise it to actually train further.
(`--skip-trained` is the different, coarser thing: skip any arm that has a
`best.pt` at all.)

**Evaluation** is sharded per `(arm, n, init_mode)` into
`runs/<tag>/results/per_n_mode/eval_<mode>_n<N>.csv`, and `evaluate.py` writes a
shard only once every rollout in it succeeds — so a shard that exists is
complete. `--resume` reuses the shards matching the grid you asked for and
re-runs only the rest, reporting each decision:

```
[subspace] resume: reusing eval_path_n32.csv (30 rows)
[subspace] resume: re-running eval_path_n64.csv (no csv)
```

A shard is reused only if its sidecar `.meta.json` records the same `n`, seeds,
density grid and init mode, **and** it is newer than the arm's `best.pt`. That
last check matters because retraining rewrites `best.pt` in place, leaving its
path unchanged — without it, `run_all --resume` would happily pair fresh weights
with stale evaluation rows. Summaries (`raw.csv`, `summary_*.csv`) are always
rebuilt from the full shard set, reused or not.

Note that `run_all --resume` still re-runs the visualize stage, which is cheap.

The evaluation grid runs to n=64 deliberately: the three trackers are closest at
n≤32, so a grid that stops there understates the differences between them.

If the old model is not at `../old_model`, point at it:

```bash
export LITE_TRIALS_OLD_MODEL_ROOT=/path/to/old_model     # PowerShell: $env:LITE_TRIALS_OLD_MODEL_ROOT=...
```

## Layout

```
lite_trials_new/
  arms.py                  arm definitions + the shared config template
  compare_train.py         one full (short) training loop per arm -> manifest.json
  compare_experiment.py    path-vs-backbone evaluation on one shared grid
  compare_visualize.py     figures + summary tables
  run_all.py               train -> evaluate -> visualize
  configs/<tag>.yaml       generated; regenerated on every compare_train run
  runs/<tag>/              checkpoints, metrics.csv, tensorboard, results/
  comparison/              cross-arm figures and CSVs
```

## Outputs

`comparison/`:

| file | what it shows |
|---|---|
| `runtime_comparison.png` | **the headline cost figure** — episode wall time vs n, per-edge-decision cost, cost vs density, training throughput |
| `runtime_summary.csv` | per-(arm, method, n) cost table with speedup vs baseline |
| `lambda2_vs_density.png` | λ₂ vs density, binned mean ± std, small multiples by n |
| `density_bucket_bars.png` | λ₂ averaged over sparse/medium/dense regimes |
| `head_to_head_delta.png` | paired Δλ₂ vs the baseline arm |
| `head_to_head.csv` | the same, tabulated, with win rates |
| `training_curves.png` | learning progress, raw eval λ₂, and episode return vs env steps |
| `comparison_summary.csv` | one row per arm |

Three notes on reading these:

- **λ₂ is reported raw**, never divided by n — algebraic connectivity is an
  absolute measure.
- **Mean episode return is not comparable across arms.** The two arms optimize
  different reward functions, so their return scales differ by construction.
  That panel is a per-arm convergence check, not a comparison. λ₂ is the
  comparable quality metric.
- **The old model's `train.py` logs no raw-λ₂ eval column** — only
  `best_eval_metric`, which is normalized (λ₂/n). The new model logs both. So
  `training_curves.png` compares arms on `best_eval_metric` in its top panel
  and shows raw eval λ₂ only for the arm that records it. Raw λ₂ for *both*
  arms comes from the evaluation stage, which runs the shared `evaluate.py` —
  that is where the quality comparison actually lives.

Head-to-head pairing is exact: same n, same target edge count, same evaluation
seed, same init mode. Runtime figures use `rl_runtime_sec` (the policy's own
rollout time) rather than `runtime_total_sec`, because backbone construction is
shared identical code and would dilute the measured difference.

## Note on `run_path_vs_backbone_experiment.py`

`compare_experiment.py` reimplements that script's driver loop rather than
calling it. Two reasons: it hard-codes `-m rl_train.evaluate`, which under
`analytic_tractable_test_deepseek` would silently evaluate the *wrong* package
(this repo also has an `rl_train/`); and each arm must run with its own repo
root as cwd, which that script can't express since it derives the root from its
own file location. The density grid, per-(n, init_mode) eval calls, and the
cost/quality summaries all match it.
