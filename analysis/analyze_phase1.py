#!/usr/bin/env python3
"""
analyze_phase1.py — build the Phase 1 master tables from collected runs.

Reads  ~/phase1_results/<node>/runs/<run_id>/   (from collect_phase1.sh)
Writes <root>/tables/:
  P1_master.csv        one row per cell: config + app metrics + wire bytes +
                       bytes/token (measured) + bytes/token (predicted) + GPU util
  P1_pernode_bytes.csv per-node TX/RX bytes per run (EP imbalance evidence)

Measured bytes convention: TX-only, summed over all nodes and both rails,
delta(port_xmit_data)*4 across the cell window. Matches predict_traffic.py.

Usage:
  python3 analyze_phase1.py --root ~/phase1_results
Optional: --devA/--devB to override RDMA device names.
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_traffic import load_cfg, predict  # noqa: E402

HW_KEYS = ["out_of_sequence", "packet_seq_err", "local_ack_timeout_err",
           "np_cnp_sent", "rp_cnp_handled", "np_ecn_marked_roce_packets"]


def read_kv(path):
    d = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        d[parts[0]] = int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    return d


def build_index(root):
    runs = defaultdict(list)
    for d in glob.glob(os.path.join(root, "*", "runs", "*")):
        if os.path.isdir(d):
            runs[os.path.basename(d)].append(d)
    return runs


def per_node_deltas(run_dirs, devs):
    """{(host, dev): {'tx': bytes, 'rx': bytes, hw...}} across all node dirs."""
    out = {}
    for d in run_dirs:
        for dev in devs:
            for before in glob.glob(os.path.join(d, f"*_{dev}_counters_before.txt")):
                host = os.path.basename(before).split(f"_{dev}_")[0]
                after = before.replace("_before.txt", "_after.txt")
                b, a = read_kv(before), read_kv(after)
                tx = (a.get("port_xmit_data", 0) - b.get("port_xmit_data", 0)) * 4
                rx = (a.get("port_rcv_data", 0) - b.get("port_rcv_data", 0)) * 4
                rec = out.setdefault((host, dev), dict(tx=0, rx=0))
                rec["tx"] += max(0, tx)
                rec["rx"] += max(0, rx)
            for before in glob.glob(os.path.join(d, f"*_{dev}_hw_counters_before.txt")):
                host = os.path.basename(before).split(f"_{dev}_")[0]
                after = before.replace("_before.txt", "_after.txt")
                b, a = read_kv(before), read_kv(after)
                rec = out.setdefault((host, dev), dict(tx=0, rx=0))
                for k in HW_KEYS:
                    if k in a and k in b:
                        rec[k] = rec.get(k, 0) + (a[k] - b[k])
    return out


def gpu_util(run_dirs):
    """Mean GPU util % per host from <host>_gpu.csv files."""
    utils = {}
    for d in run_dirs:
        for f in glob.glob(os.path.join(d, "*_gpu.csv")):
            host = os.path.basename(f).replace("_gpu.csv", "")
            vals = []
            try:
                for line in open(f):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2 and parts[1].endswith("%"):
                        try:
                            vals.append(float(parts[1].rstrip(" %")))
                        except ValueError:
                            pass
            except OSError:
                continue
            if vals:
                utils.setdefault(host, []).extend(vals)
    return {h: sum(v) / len(v) for h, v in utils.items() if v}



def net_rates(run_dirs):
    """From <host>_net.csv time series: peak/mean cluster TX rate (Gb/s) and a
    burstiness ratio (peak/mean). Returns {} if no series were captured."""
    per_host = {}
    for d in run_dirs:
        for f in glob.glob(os.path.join(d, "*_net.csv")):
            host = os.path.basename(f).replace("_net.csv", "")
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if df.empty or "xmit_bytes" not in df:
                continue
            # sum both devices per timestamp, then difference over time
            g = df.groupby("ts")["xmit_bytes"].sum().sort_index()
            if len(g) < 3:
                continue
            ts = g.index.to_numpy(dtype=float)
            by = g.to_numpy(dtype=float)
            dt = ts[1:] - ts[:-1]
            db = by[1:] - by[:-1]
            ok = (dt > 0) & (db >= 0)
            if not ok.any():
                continue
            per_host[host] = (db[ok] * 8 / 1e9) / dt[ok]   # Gb/s samples
    if not per_host:
        return {}
    # align by sample index across hosts to approximate cluster-wide rate
    n = min(len(v) for v in per_host.values())
    if n < 2:
        return {}
    cluster = [sum(v[i] for v in per_host.values()) for i in range(n)]
    mean = sum(cluster) / len(cluster)
    peak = max(cluster)
    return dict(net_gbps_mean=round(mean, 3), net_gbps_peak=round(peak, 3),
                net_burstiness=round(peak / mean, 2) if mean else None,
                net_samples=n)

def first(run_dirs, name):
    for d in run_dirs:
        p = os.path.join(d, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/phase1_results"))
    ap.add_argument("--devA", default="rocep1s0f0")
    ap.add_argument("--devB", default="roceP2p1s0f0")
    args = ap.parse_args()
    devs = [args.devA, args.devB]

    runs = build_index(args.root)
    if not runs:
        sys.exit(f"no runs under {args.root}/*/runs/")
    print(f"indexed {len(runs)} runs")

    rows, pernode_rows = [], []
    cfg_cache = {}
    for rid, dirs in sorted(runs.items()):
        mpath = first(dirs, "manifest.json")
        cpath = first(dirs, "client.json")
        if not mpath:
            print(f"  [warn] {rid}: no manifest, skipped"); continue
        man = json.load(open(mpath))
        agg = {}
        if cpath:
            agg = json.load(open(cpath)).get("aggregates", {})
        else:
            print(f"  [warn] {rid}: no client.json (failed cell?)")

        deltas = per_node_deltas(dirs, devs)
        tx_total = sum(v["tx"] for v in deltas.values())
        rx_total = sum(v["rx"] for v in deltas.values())
        hw = {k: sum(v.get(k, 0) for v in deltas.values()) for k in HW_KEYS}
        for (host, dev), v in sorted(deltas.items()):
            pernode_rows.append(dict(run=rid, host=host, dev=dev,
                                     tx_bytes=v["tx"], rx_bytes=v["rx"]))

        out_toks = agg.get("total_output_tokens") or 0
        in_toks = agg.get("total_input_tokens") or 0
        bpt_out = tx_total / out_toks if out_toks else None
        bpt_all = tx_total / (out_toks + in_toks) if (out_toks + in_toks) else None

        # prediction from the model's own config.json
        pred = {}
        cfg_file = os.path.join(man["model_path"], "config.json")
        try:
            if cfg_file not in cfg_cache:
                cfg_cache[cfg_file] = load_cfg(cfg_file)
            pred = predict(cfg_cache[cfg_file], man["tp"], man["pp"], man["ep"])
        except Exception as e:
            print(f"  [warn] {rid}: prediction failed ({e})")

        util = gpu_util(dirs)
        rates = net_rates(dirs)
        rows.append(dict(
            run=rid, label=man.get("label"), model=os.path.basename(man["model_path"]),
            tp=man["tp"], pp=man["pp"], ep=man["ep"], mode=man.get("mode"),
            workload=man.get("workload"), input_len=man.get("input_len"),
            output_len=man.get("output_len"), concurrency=man.get("concurrency"),
            completed=agg.get("completed"), failed=agg.get("failed"),
            output_tok_per_s=agg.get("output_tok_per_s"),
            ttft_s_p50=agg.get("ttft_s_p50"), ttft_s_p95=agg.get("ttft_s_p95"),
            itl_ms_p50=agg.get("itl_ms_p50"), itl_ms_p95=agg.get("itl_ms_p95"),
            itl_ms_mean=agg.get("itl_ms_mean"),
            total_output_tokens=out_toks, total_input_tokens=in_toks,
            wire_tx_bytes=tx_total, wire_rx_bytes=rx_total,
            bytes_per_output_token=bpt_out, bytes_per_total_token=bpt_all,
            pred_bytes_per_token=pred.get("total_bytes_per_token"),
            pred_tp_bytes=pred.get("tp_allreduce_bytes_per_token"),
            pred_ep_bytes=pred.get("ep_alltoall_bytes_per_token"),
            pred_pp_bytes=pred.get("pp_boundary_bytes_per_token"),
            is_moe=pred.get("is_moe"),
            gpu_util_mean=(sum(util.values()) / len(util) if util else None),
            **rates,
            **{f"hw_{k}": hw[k] for k in HW_KEYS},
        ))

    outdir = os.path.join(args.root, "tables")
    os.makedirs(outdir, exist_ok=True)
    master = pd.DataFrame(rows)
    master.to_csv(os.path.join(outdir, "P1_master.csv"), index=False)
    pd.DataFrame(pernode_rows).to_csv(
        os.path.join(outdir, "P1_pernode_bytes.csv"), index=False)
    print(f"wrote {outdir}/P1_master.csv ({len(master)} rows)")
    print(f"wrote {outdir}/P1_pernode_bytes.csv ({len(pernode_rows)} rows)")

    # console preview: measured vs predicted bytes/token for decode cells
    if not master.empty:
        dec = master[(master.workload == "decode")]
        cols = ["run", "bytes_per_output_token", "pred_bytes_per_token",
                "itl_ms_p50", "output_tok_per_s"]
        cols = [c for c in cols if c in dec.columns]
        if not dec.empty:
            print("\n=== decode cells: measured vs predicted bytes/token ===")
            with pd.option_context("display.width", 150):
                print(dec[cols].to_string(index=False))


if __name__ == "__main__":
    main()
