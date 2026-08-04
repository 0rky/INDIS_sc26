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

## B. Re-running Phase 0: the communication envelope

Phase 0 measures what the fabric can do, independently of any model. It runs
first, because the Phase 1 results are interpreted against this envelope. Hardware: 4GB10 connected to a switch with star topology.The connection bandwidth is 200 Gbps per GB10 node.

### B.1 One-time setup (all four nodes)

```bash
sudo apt install -y iperf3 perftest ethtool sysstat chrony jq \
                    openmpi-bin libopenmpi-dev numactl
```

Set up passwordless SSH from the launch node to all four nodes, using host
aliases `node1`–`node4` in `~/.ssh/config`. Create a working directory that
exists at the same path everywhere and deploy the scripts:

```bash
for n in node1 node2 node3 node4; do
  ssh $n 'sudo mkdir -p /opt/phase0/runs && sudo chown -R $USER /opt/phase0'
  scp scripts/phase0/{env.sh,snapshot.sh,capture_local.sh} $n:/opt/phase0/
  ssh $n 'chmod +x /opt/phase0/*.sh'
done
cp scripts/phase0/*.sh /opt/phase0/ && chmod +x /opt/phase0/*.sh
```

Build `nccl-tests` **with MPI support** at `/opt/nccl-tests` on every node:

```bash
make MPI=1 MPI_HOME=/usr/lib/aarch64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda -j
```

MPI hostfiles on the launch node:

```bash
printf "node1 slots=1\nnode2 slots=1\nnode3 slots=1\nnode4 slots=1\n" > ~/hosts4.txt
printf "node1 slots=1\nnode2 slots=1\n" > ~/hosts2.txt
```

### B.2 Preflight checks (do not skip)

**Link speeds must match across all eight links.** Mixed rates invalidate every
collective measurement:

```bash
for n in node1 node2 node3 node4; do
  ssh $n 'ethtool enp1s0f0np0 | grep Speed; ethtool enP2p1s0f0np0 | grep Speed'
done
```

**Jumbo frames end-to-end, both rails:**

```bash
for i in 1 3 4; do
  ping -M do -s 8972 -c 2 192.168.100.$i    # rail A
  ping -M do -s 8972 -c 2 192.168.101.$i    # rail B
done
```

**Find the RoCE v2 GID index** and set it in `env.sh` on every node:

```bash
for D in rocep1s0f0 roceP2p1s0f0; do
  for g in /sys/class/infiniband/$D/ports/1/gid_attrs/types/*; do
    i=$(basename $g)
    echo "$i $(cat $g 2>/dev/null) $(cat /sys/class/infiniband/$D/ports/1/gids/$i)"
  done
done
```

Use the index whose type is `RoCE v2` and whose GID contains the rail's IPv4
address. It must be the same index on every node.

Reduce noise: `sudo systemctl disable --now irqbalance`, set the CPU governor to
performance, and leave the switch at factory defaults (this is deliberate — the
paper characterizes the default configuration).

### B.3 Run the three experiments

```bash
cd /opt/phase0

# A. TCP point-to-point envelope  (~1 hour)
for n in node1 node2 node3 node4; do
  ssh $n 'pkill iperf3; iperf3 -s -p 5201 -D; iperf3 -s -p 5202 -D'
done
./run_iperf.sh

# B. RDMA point-to-point, all six pairs and both rails  (~2 hours)
./run_perftest.sh

# C. NCCL collectives: 3 collectives x {2,4} nodes x 3 transports x 3 reps  (~4-6 hours)
./run_nccl.sh
```

**Validity check after the first NCCL run of each transport mode.** A RoCE run
that silently fell back to sockets is invalid and must be discarded:

```bash
grep -h "via NET" /opt/phase0/runs/<run_id>/nccl.*.log | sort -u
```

Expect `NET/IB` for the `dual` and `single` modes and `NET/Socket` for `tcp`.

Sanity gates before trusting the data: per-rail RDMA bandwidth should approach
line rate, the dual-rail aggregate roughly twice that, 2-byte latency should be
single-digit microseconds, and all six node pairs should be symmetric.

### B.4 Collect and analyze

```bash
./collect_results.sh                      # -> ~/phase0_results/
python3 analysis/parse_nccl_v2.py --root ~/phase0_results --verbose
python3 analysis/plot_nccl_v2.py  --root ~/phase0_results \
        --line-rate-gbps 100 --annotate-protocols
```

`parse_nccl_v2.py` writes `T6_nccl_full.csv` (every message size) and
`T6_nccl_summary.csv`, and reports any run that failed to parse. The plotting
script produces the bandwidth curve, the small-message latency figure, and the
per-collective rail-scaling figure used in the paper.

---

## C. Re-running Phase 1: the inference campaign

Requires the Phase 0 testbed plus model weights.

```bash
cd /opt/phase1
./setup_container.sh          # build and distribute the serving container
bash setup_client_env.sh      # CPU-only client environment on the launch node
./start_ray.sh                # form the cluster; MODE=dual|single|tcp
./run_matrix.sh configs.tsv   # the campaign; resumable after interruption
./collect_all.sh --fresh      # gather results to one node
```

Build the tables:

```bash
python3 analysis/analyze_phase1.py --root ~/phase1_results
python3 analysis/reanalyze_v2.py   --root ~/phase1_results --line-rate-gbps 200
```

For the transport comparison, restart the cluster in TCP mode and re-run the two
four-node tensor-parallel configurations:

```bash
./stop_ray.sh && MODE=tcp ./start_ray.sh
MODE=tcp ./run_matrix.sh configs_tcp.tsv
./stop_ray.sh && MODE=dual ./start_ray.sh
```

**Two validation checks must pass before trusting any Phase 1 result:**

- Single-node configurations must record **exactly zero** RDMA bytes. Confirm
  with `verify_run.sh <run_id>`.
- The transport NCCL actually used must match the intended mode:
  `grep "via NET" server_logs/*.log`. See `logs/transport_verification.txt`
  for ours.

**Estimated time:** Phase 0 about one day; Phase 1 several days, dominated by
loading model weights across nodes.


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

[FUTURE]

## License

Apache-2.0 
