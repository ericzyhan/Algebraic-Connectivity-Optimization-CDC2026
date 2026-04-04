# baseline_comparison (HPC)

Manifest-driven baseline experiment runner.

## Main Runner

`run_manifest_hpc.py` reads a JSON manifest and executes density sweeps.

### Usage

```bash
python experiments/baseline_comparison/run_manifest_hpc.py \
  --manifest experiments/baseline_comparison/manifests/baseline_comparison.json
```

Dry run:

```bash
python experiments/baseline_comparison/run_manifest_hpc.py \
  --manifest experiments/baseline_comparison/manifests/baseline_comparison.json \
  --dry_run
```

Optional output override:

```bash
python experiments/baseline_comparison/run_manifest_hpc.py \
  --manifest experiments/baseline_comparison/manifests/fvg_n8_128_rep5.json \
  --out_csv logs/baseline_comparison/custom_fvg.csv \
  --out_meta logs/baseline_comparison/custom_fvg.meta.json
```

## Included Manifests

- `baseline_comparison.json`: full experiment (type 1+2+3 mix)
- `fvg_n8_128_rep5.json`
- `mac_n8_128_rep5.json`
- `mdmd_n8_128_rep5.json`
- `wts_n8_128_rep5.json`
- `kopt_n8_128_rep5.json`
- `sdps_n8_32_rep5.json`
- `oa_pmc_n8_rep5.json`

## Summary

```bash
python experiments/baseline_comparison/summarize_big_experiment.py \
  --in_csv logs/baseline_comparison/eval_big_type123_raw.csv \
  --out_cost_csv logs/baseline_comparison/summary_cost.csv \
  --out_quality_csv logs/baseline_comparison/summary_quality.csv
```

