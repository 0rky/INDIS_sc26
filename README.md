# Characterizing Network Behavior of Distributed LLM Inference on a Four-Node GB10 Edge Cluster

Artifact for the INDIS @ SC26 paper. Contains the measurement pipeline, the raw
results, and the analysis code that produces every figure and table in the paper.

## What is here

```
scripts/phase0/     microbenchmark sweep (iperf3, perftest, nccl-tests)
scripts/phase1/     inference campaign: cluster setup, matrix driver, instrumentation
analysis/           analysis and plotting code, plus the analytical traffic model
data/tables/        derived result tables (CSV) — these regenerate every figure
data/raw.zip        zip file of per-cell raw measurements: manifests, client results, counters, time series
figures/            the figures as they appear in the paper
logs/               software versions and evidence excerpts for two key claims
```

## Reproducing the figures without hardware (5 minutes)

This is the level most readers will want. No GB10 cluster required. Raw per-cell measurements are in `raw_measurements.tar.gz`
(`tar xzf data/raw_measurements.tar.gz`). The derived tables in `data/tables/` are sufficient to regenerate every figure.

We expect the models inside of /opt/models/ look at the config.tsv file to look at the names of the folder the script is expecting. They are similar to huggingface name for the model.  

```bash
conda create -n indis python=3.12 -y && conda activate indis
pip install pandas matplotlib

# Phase 1 figures
python3 analysis/reanalyze_v2.py        --root data --line-rate-gbps 200
python3 analysis/concurrency_figures.py --root data
python3 analysis/roofline_concurrency.py --root data
python3 analysis/roofline_redesign.py   --root data

# Phase 0 figures
python3 analysis/plot_nccl_v2.py --root data --line-rate-gbps 100 --annotate-protocols
```

Output appears in `data/tables/figures_corrected/`. Compare against `figures/`.

## Re-running the experiments (requires the hardware)

Four GB10 nodes with dual-rail RoCEv2 through one switch. See
`scripts/phase1/` and follow in order:

1. `setup_container.sh` — build and distribute the serving container
2. `setup_client_env.sh` — CPU-only client environment on the launch node
3. `start_ray.sh` — form the cluster (set `MODE=dual|single|tcp`)
4. `run_matrix.sh configs.tsv` — run the campaign; Please look at the tsv file and uncomment the line that you want to learn
5. `collect_all.sh --fresh` — gather results to one node
6. `analyze_phase1.py` then `reanalyze_v2.py` — build the tables

**Two validation checks must pass before trusting any result:**

- Single-node configurations must record **exactly zero** RDMA bytes.
  Confirm with `verify_run.sh <run_id>`.
- The transport NCCL actually used must match the intended mode. Confirm with
  `grep "via NET" server_logs/*.log` — expect `NET/IB` for RoCE modes and
  `NET/Socket` for TCP. See `logs/transport_verification.txt` for ours.

## Key data files

| File | Contents |
|---|---|
| `data/tables/P1_master_corrected.csv` | 110 rows, one per measurement cell, all metrics |
| `data/tables/P1_pernode_bytes.csv` | per-node, per-rail byte counts (rail split, imbalance) |
| `data/tables/T6_nccl_full.csv` | full collective sweep, 1 KB – 1 GB, all transports |
| `data/tables/TBC2_transport_comparison.csv` | RoCE vs TCP per configuration |

Run identifiers encode the configuration, e.g.
`p1_l70_tp4_dual_decode_in128_out1024_c32` is Llama-70B, TP4, dual-rail RoCE,
decode workload, 128-token prompt, 1024 generated tokens, concurrency 32.

## Models

Weights are not redistributable. Download from Hugging Face:

- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`
- `Qwen/Qwen3-30B-A3B-Instruct-2507`

Place at the **same absolute path on every node** and update `configs.tsv`.

## Environment

- vLLM `0.23.1rc1.dev1551+g61aca5d75` (development build; later releases may
  select different collective algorithms or expert-dispatch backends)
- NVIDIA driver 580.x
- Container: https://github.com/eugr/spark-vllm-docker Follow the instructions in this repo on how to use their containers. You may have to build it from scratch. For this we did build it from scratch.

Full version details in `logs/versions_*.txt`.

## Notes

`data/raw.zip` excludes `ethtool` dumps and NCCL debug logs, which are large and
redundant with the RDMA counters used in the analysis. Excerpts documenting the
transport verification and the expert-parallelism finding are in `logs/`.

## Citation

[FILL IN once the paper has a DOI or arXiv identifier]

## License

[Choose one — MIT or Apache-2.0 is conventional for artifacts]
