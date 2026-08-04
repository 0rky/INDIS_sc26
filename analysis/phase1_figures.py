#!/usr/bin/env python3
"""
phase1_figures.py — paper figures that (a) derive new metrics from P1_master.csv
and (b) BRIDGE Phase 0's microbenchmark envelope to Phase 1's serving results.

    python3 phase1_figures.py --root ~/phase1_results \
        --phase0 /path/to/T6_nccl_full.csv

Requires: analyze_phase1.py has already produced tables/P1_master.csv.

Figures written to <root>/tables/figures/:
  P10_distribution_tax.png        latency cost of distributing, vs 1-node baseline
  P11_phase0_bridge.png           Phase 0 predicted collective time vs measured ITL
  P12_ep_matched_pairs.png        EP on/off at identical TP/PP (3 matched pairs)
  P13_efficiency_frontier.png     throughput per MB of network traffic
  P14_model_accuracy.png          measured/predicted traffic ratio per config
  P15_decode_msgsize_map.png      where each config lands on Phase 0's curve

Tables written to <root>/tables/:
  TB1_config_inventory.csv        every config + predicted traffic + status
  TB2_headline_results.csv        the numbers that go in the paper's main table
  TB3_ep_pairs.csv                EP on/off deltas
  TB4_phase0_bridge.csv           per-config predicted vs measured collective time
"""
import argparse, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# model geometry (hidden size, layers) — used for message-size math
GEOM = {
    "Llama-3.1-8B-Instruct":      dict(h=4096, L=32, moe=False),
    "Llama-3.3-70B-Instruct":     dict(h=8192, L=80, moe=False),
    "Qwen3-30B-A3B-Instruct-2507": dict(h=2048, L=48, moe=True, topk=8),
}
PALETTE = {"Llama-3.1-8B-Instruct": "#1f77b4",
           "Llama-3.3-70B-Instruct": "#ff7f0e",
           "Qwen3-30B-A3B-Instruct-2507": "#2ca02c"}


def load_master(root):
    p = os.path.join(root, "tables", "P1_master.csv")
    if not os.path.exists(p):
        sys.exit("P1_master.csv not found — run analyze_phase1.py first")
    m = pd.read_csv(p)
    # normalise: some columns may be absent if a cell failed
    for c in ["itl_ms_p50", "ttft_s_p50", "output_tok_per_s",
              "bytes_per_output_token", "pred_bytes_per_token", "gpu_util_mean"]:
        if c not in m.columns:
            m[c] = pd.NA
    return m


def geom_for(model):
    for k, v in GEOM.items():
        if k in str(model):
            return v
    return None


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- P10
def p10_distribution_tax(m, out):
    """ITL increase relative to the single-node (tp1) baseline of the same model.
       This isolates the cost of DISTRIBUTING, which is the paper's core claim."""
    d = m[(m.workload == "decode") & (m.concurrency == 1) & m.itl_ms_p50.notna()]
    if d.empty:
        return None
    base = {}
    for model, g in d.groupby("model"):
        b = g[(g.tp == 1) & (g.pp == 1)]
        if not b.empty:
            base[model] = b.itl_ms_p50.iloc[0]
    if not base:
        print("  P10 skipped: no single-node baseline found")
        return None
    rows = []
    for _, r in d.iterrows():
        if r.model not in base:
            continue
        rows.append(dict(label=r.label, model=r.model, itl=r.itl_ms_p50,
                         baseline=base[r.model],
                         tax_ms=r.itl_ms_p50 - base[r.model],
                         tax_pct=100 * (r.itl_ms_p50 / base[r.model] - 1),
                         mb=r.bytes_per_output_token / 1e6
                         if pd.notna(r.bytes_per_output_token) else None))
    t = pd.DataFrame(rows).sort_values(["model", "tax_ms"])
    fig, ax = plt.subplots(figsize=(max(9, .6 * len(t) + 4), 5))
    x = range(len(t))
    ax.bar(list(x), t.tax_ms, color=[PALETTE.get(mm, "#777") for mm in t.model])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(t.label, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("ITL increase vs single-node (ms/token)")
    ax.set_title("Distribution tax: latency cost of going multi-node (concurrency=1)")
    ax.grid(True, axis="y", alpha=.3)
    save(fig, out, "P10_distribution_tax.png")
    return t


# ---------------------------------------------------------------- P11 / P15
def phase0_lookup(p0, collective, nodes, mode, size):
    """Mean measured time_us for the nearest measured message size <= size."""
    s = p0[(p0.collective == collective) & (p0.nodes == nodes) & (p0["mode"] == mode)]
    if s.empty:
        return None, None
    g = s.groupby("size")["time_us"].mean()
    cands = [k for k in g.index if k <= size]
    k = max(cands) if cands else min(g.index)
    return float(g[k]), int(k)


def p11_bridge(m, p0, out):
    """For each TP config, use Phase 0's measured all-reduce latency at the
       message size that config actually generates, times the number of
       all-reduces per token, and compare with the measured ITL."""
    rows = []
    d = m[(m.workload == "decode") & m.itl_ms_p50.notna()]
    for _, r in d.iterrows():
        g = geom_for(r.model)
        if g is None or r.tp < 2 or r.pp != 1 or r.ep != 0:
            continue           # pure-TP dense/MoE configs only: cleanest bridge
        nodes = int(r.tp)
        if nodes not in (2, 4):
            continue
        # decode all-reduce payload per op ~ batch x hidden x 2 bytes
        size = int(r.concurrency) * g["h"] * 2
        t_us, used = phase0_lookup(p0, "all_reduce", nodes, "dual", size)
        if t_us is None:
            continue
        n_ops = 2 * g["L"]                       # 2 all-reduces per layer
        pred_ms = n_ops * t_us / 1000.0
        rows.append(dict(label=r.label, model=r.model, tp=nodes,
                         concurrency=int(r.concurrency),
                         msg_bytes=size, phase0_size_used=used,
                         phase0_us_per_op=round(t_us, 1), ops_per_token=n_ops,
                         pred_collective_ms=round(pred_ms, 2),
                         measured_itl_ms=round(r.itl_ms_p50, 2),
                         collective_share_pct=round(100 * pred_ms / r.itl_ms_p50, 1)
                         if r.itl_ms_p50 else None))
    if not rows:
        print("  P11 skipped: no matching TP configs / Phase 0 sizes")
        return None
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for model, g in t.groupby("model"):
        ax.scatter(g.pred_collective_ms, g.measured_itl_ms, s=80,
                   color=PALETTE.get(model, "#777"), edgecolors="k",
                   linewidths=.5, label=model)
        for _, r in g.iterrows():
            ax.annotate(f"{r.label} c{r.concurrency}",
                        (r.pred_collective_ms, r.measured_itl_ms),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    lim = max(t.pred_collective_ms.max(), t.measured_itl_ms.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=.8, label="ITL = collective time")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Phase 0 predicted collective time (ms/token)")
    ax.set_ylabel("Phase 1 measured inter-token latency (ms)")
    ax.set_title("Phase 0 envelope predicts Phase 1 serving latency")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=.3)
    save(fig, out, "P11_phase0_bridge.png")
    return t


def p15_msgsize_map(m, p0, out):
    """Phase 0's all-reduce latency curve with Phase 1's actual message sizes
       marked — shows which configs land in the LL128 dip."""
    s = p0[(p0.collective == "all_reduce") & (p0.nodes == 4) & (p0["mode"] == "dual")]
    if s.empty:
        print("  P15 skipped: no Phase 0 4-node dual all_reduce data")
        return
    g = s.groupby("size")["time_us"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(g.index, g.values, "-o", ms=3, color="#444",
            label="Phase 0: 4-node dual-rail all-reduce")
    d = m[(m.workload == "decode") & (m.tp == 4) & (m.pp == 1)]
    seen = set()
    for _, r in d.iterrows():
        geo = geom_for(r.model)
        if geo is None:
            continue
        size = int(r.concurrency) * geo["h"] * 2
        cands = [k for k in g.index if k <= size]
        k = max(cands) if cands else min(g.index)
        key = (r.model, size)
        if key in seen:
            continue
        seen.add(key)
        ax.scatter([size], [g[k]], s=90, zorder=5,
                   color=PALETTE.get(r.model, "#777"), edgecolors="k", linewidths=.6)
        ax.annotate(f"{r.label}\nc={int(r.concurrency)}", (size, g[k]),
                    fontsize=6.5, xytext=(5, -12), textcoords="offset points")
    ax.axvspan(262144, 4194304, color="orange", alpha=.12)
    ax.text(3e5, g.values.max() * .95, "Phase 0 Ring/LL128 regime\n(busbw dip)",
            fontsize=7, color="#a15c00")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("all-reduce message size (bytes)")
    ax.set_ylabel("Phase 0 measured time per op (us)")
    ax.set_title("Where Phase 1 decode workloads land on the Phase 0 curve")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=.25)
    save(fig, out, "P15_decode_msgsize_map.png")


# ---------------------------------------------------------------- P12
def p12_ep_pairs(m, out):
    """Matched pairs: identical TP/PP, EP off vs on. Three pairs in this matrix."""
    d = m[(m.workload == "decode") & m.itl_ms_p50.notna()]
    if d.empty:
        return None
    rows = []
    for (model, tp, pp, conc), g in d.groupby(["model", "tp", "pp", "concurrency"]):
        off = g[g.ep == 0]; on = g[g.ep == 1]
        if off.empty or on.empty:
            continue
        o, e = off.iloc[0], on.iloc[0]
        rows.append(dict(model=model, tp=int(tp), pp=int(pp), concurrency=int(conc),
                         label_off=o.label, label_on=e.label,
                         itl_off=o.itl_ms_p50, itl_on=e.itl_ms_p50,
                         itl_delta_pct=round(100*(e.itl_ms_p50/o.itl_ms_p50-1), 1),
                         mb_off=(o.bytes_per_output_token or 0)/1e6,
                         mb_on=(e.bytes_per_output_token or 0)/1e6,
                         tps_off=o.output_tok_per_s, tps_on=e.output_tok_per_s))
    if not rows:
        print("  P12 skipped: no EP on/off pairs")
        return None
    t = pd.DataFrame(rows)
    sub = t[t.concurrency == t.concurrency.min()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    lbl = [f"TP{r.tp}PP{r.pp}" for _, r in sub.iterrows()]
    x = range(len(sub))
    axes[0].bar([i-.2 for i in x], sub.mb_off, width=.4, label="EP off (all-reduce)")
    axes[0].bar([i+.2 for i in x], sub.mb_on,  width=.4, label="EP on (+all-to-all)")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(lbl, fontsize=8)
    axes[0].set_ylabel("MB per generated token"); axes[0].legend(fontsize=8)
    axes[0].set_title("Traffic: expert parallelism on/off")
    axes[0].grid(True, axis="y", alpha=.3)
    axes[1].bar([i-.2 for i in x], sub.itl_off, width=.4, label="EP off")
    axes[1].bar([i+.2 for i in x], sub.itl_on,  width=.4, label="EP on")
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(lbl, fontsize=8)
    axes[1].set_ylabel("ITL p50 (ms)"); axes[1].legend(fontsize=8)
    axes[1].set_title("Latency: expert parallelism on/off")
    axes[1].grid(True, axis="y", alpha=.3)
    fig.suptitle("MoE matched pairs: same model, same layout, EP toggled", fontsize=11)
    save(fig, out, "P12_ep_matched_pairs.png")
    return t


# ---------------------------------------------------------------- P13 / P14
def p13_efficiency(m, out):
    d = m[(m.workload == "decode") & m.output_tok_per_s.notna()
          & m.bytes_per_output_token.notna()].copy()
    d = d[d.bytes_per_output_token > 0]
    if d.empty:
        return
    d = d.sort_values("concurrency").groupby("label").tail(1)
    d["tok_per_MB"] = d.output_tok_per_s / (d.bytes_per_output_token / 1e6)
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, g in d.groupby("model"):
        ax.scatter(g.bytes_per_output_token/1e6, g.output_tok_per_s, s=80,
                   color=PALETTE.get(model, "#777"), edgecolors="k",
                   linewidths=.5, label=model)
        for _, r in g.iterrows():
            ax.annotate(r.label, (r.bytes_per_output_token/1e6, r.output_tok_per_s),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("MB of network traffic per generated token (log)")
    ax.set_ylabel("output tokens / s")
    ax.set_title("Efficiency frontier: throughput achieved per unit of traffic")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=.3)
    save(fig, out, "P13_efficiency_frontier.png")


def p14_accuracy(m, out):
    d = m[(m.workload == "decode") & m.bytes_per_output_token.notna()
          & m.pred_bytes_per_token.notna()].copy()
    d = d[d.pred_bytes_per_token > 0]
    if d.empty:
        return
    d = d.sort_values("concurrency").groupby("label").tail(1).sort_values(["model", "label"])
    d["ratio"] = d.bytes_per_output_token / d.pred_bytes_per_token
    fig, ax = plt.subplots(figsize=(max(9, .6*len(d)+4), 5))
    x = range(len(d))
    ax.bar(list(x), d.ratio, color=[PALETTE.get(mm, "#777") for mm in d.model])
    ax.axhline(1.0, color="k", ls="--", lw=1, label="perfect agreement")
    ax.axhspan(0.8, 2.0, color="green", alpha=.08, label="expected band (0.8-2x)")
    ax.set_xticks(list(x)); ax.set_xticklabels(d.label, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("measured / predicted bytes per token")
    ax.set_title("Analytical model accuracy per configuration")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=.3)
    save(fig, out, "P14_model_accuracy.png")
    return d[["label", "model", "tp", "pp", "ep", "bytes_per_output_token",
              "pred_bytes_per_token", "ratio"]]


# ---------------------------------------------------------------- tables
def write_tables(root, m, tax, bridge, eppairs, acc):
    td = os.path.join(root, "tables")
    inv = (m.groupby(["label", "model", "tp", "pp", "ep"], dropna=False)
             .agg(cells=("run", "count"),
                  pred_MB_per_token=("pred_bytes_per_token",
                                     lambda s: round((s.dropna().iloc[0] or 0)/1e6, 4)
                                     if s.notna().any() else None))
             .reset_index().sort_values(["model", "label"]))
    inv.to_csv(os.path.join(td, "TB1_config_inventory.csv"), index=False)
    print("  wrote TB1_config_inventory.csv")

    cols = ["label", "model", "tp", "pp", "ep", "workload", "concurrency",
            "itl_ms_p50", "itl_ms_p95", "ttft_s_p50", "output_tok_per_s",
            "bytes_per_output_token", "pred_bytes_per_token",
            "net_gbps_mean", "net_gbps_peak", "net_burstiness", "gpu_util_mean"]
    cols = [c for c in cols if c in m.columns]
    m[cols].sort_values(["model", "label", "workload", "concurrency"]) \
        .to_csv(os.path.join(td, "TB2_headline_results.csv"), index=False)
    print("  wrote TB2_headline_results.csv")

    for obj, name in ((tax, "TB_distribution_tax.csv"),
                      (bridge, "TB4_phase0_bridge.csv"),
                      (eppairs, "TB3_ep_pairs.csv"),
                      (acc, "TB_model_accuracy.csv")):
        if obj is not None and len(obj):
            obj.to_csv(os.path.join(td, name), index=False)
            print("  wrote", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/phase1_results"))
    ap.add_argument("--phase0", help="Phase 0 T6_nccl_full.csv (enables P11/P15)")
    args = ap.parse_args()

    m = load_master(args.root)
    out = os.path.join(args.root, "tables", "figures")
    os.makedirs(out, exist_ok=True)
    print(f"loaded {len(m)} cells, {m.label.nunique()} configs")

    tax = p10_distribution_tax(m, out)
    eppairs = p12_ep_pairs(m, out)
    p13_efficiency(m, out)
    acc = p14_accuracy(m, out)

    bridge = None
    if args.phase0 and os.path.exists(args.phase0):
        p0 = pd.read_csv(args.phase0)
        bridge = p11_bridge(m, p0, out)
        p15_msgsize_map(m, p0, out)
    else:
        print("  (no --phase0 given: skipping P11 and P15 bridge figures)")

    write_tables(args.root, m, tax, bridge, eppairs, acc)
    print("\ndone. figures in", out)


if __name__ == "__main__":
    main()
