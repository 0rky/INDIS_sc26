#!/usr/bin/env python3
"""
plot_phase1.py — Phase 1 figures from P1_master.csv / P1_pernode_bytes.csv.

  python3 plot_phase1.py --root ~/phase1_results

Writes to <root>/tables/figures/:
  CUMULATIVE (all models on one axis — good while you have few models, and
  good for the paper's headline comparisons):
    P1_bytes_per_token_validation.png   measured vs predicted, grouped by model
    P4_throughput_vs_concurrency.png    tokens/s scaling, colour=model
    P6_gpu_util_by_config.png           GPU utilisation by config
    P7_dense_vs_moe.png                 dense vs MoE traffic contrast (NEW)
    P8_traffic_composition.png          predicted TP/EP/PP byte breakdown (NEW)
  PER-MODEL (written to figures/by_model/<model>/ when >1 model is present,
  so crowded plots stay readable as the model list grows):
    P2_decode_itl_by_config.png         decode ITL across parallelism configs
    P3_prefill_ttft_by_config.png       prefill TTFT across configs
    P4_throughput_vs_concurrency.png    per-model scaling curves
    P5_pernode_imbalance.png            per-node TX bytes (EP imbalance)

Figures whose data is absent are skipped silently.
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def load(root, name):
    p = os.path.join(root, "tables", name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def model_colors(m):
    models = sorted(m.model.dropna().unique())
    return {mod: PALETTE[i % len(PALETTE)] for i, mod in enumerate(models)}


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)


def steady(d):
    """highest-concurrency cell per label (steadiest state)."""
    return d.sort_values("concurrency").groupby("label").tail(1)


def lightest(d):
    """concurrency=1 cell per label (pure latency)."""
    return d.sort_values("concurrency").groupby("label").head(1)


# ---------------------------------------------------------------- cumulative
def p1_validation(m, out, colors):
    d = m[(m.workload == "decode") & m.bytes_per_output_token.notna()
          & m.pred_bytes_per_token.notna()]
    if d.empty:
        return
    d = steady(d).sort_values(["model", "label"])
    x = range(len(d))
    fig, ax = plt.subplots(figsize=(max(8.5, 0.75 * len(d) + 4), 5))
    ax.bar([i - .2 for i in x], d.bytes_per_output_token / 1e6, width=.4,
           color=[colors.get(mm, "#1f77b4") for mm in d.model],
           label="measured (RDMA counters)")
    ax.bar([i + .2 for i in x], d.pred_bytes_per_token / 1e6, width=.4,
           color="#cccccc", edgecolor="#555", label="predicted (analytical)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(d.label, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("cluster TX bytes per generated token (MB)")
    ax.set_title("Measured vs predicted network traffic per token (decode)")
    # model separators + labels
    prev, start = None, 0
    for i, mm in enumerate(list(d.model) + [None]):
        if mm != prev and prev is not None:
            ax.axvline(i - .5, color="#999", lw=.7, ls=":")
            ax.text((start + i - 1) / 2, ax.get_ylim()[1] * .97, prev,
                    ha="center", va="top", fontsize=7, color="#333")
            start = i
        if prev is None:
            start = i
        prev = mm
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=.3)
    save(fig, out, "P1_bytes_per_token_validation.png")


def p7_dense_vs_moe(m, out, colors):
    """The paper's core contrast: traffic per token, dense vs MoE."""
    d = m[(m.workload == "decode") & m.bytes_per_output_token.notna()
          & m.is_moe.notna()]
    if d.empty or d.is_moe.nunique() < 1:
        return
    d = steady(d).copy()
    d["arch"] = d.is_moe.map({True: "MoE", False: "Dense"})
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for arch, mk in (("Dense", "o"), ("MoE", "s")):
        s = d[d.arch == arch]
        if s.empty:
            continue
        ax.scatter(s.itl_ms_p50, s.bytes_per_output_token / 1e6, marker=mk,
                   s=70, alpha=.85,
                   color=[colors.get(mm, "#1f77b4") for mm in s.model],
                   edgecolors="k", linewidths=.5, label=arch)
        for _, r in s.iterrows():
            ax.annotate(r.label, (r.itl_ms_p50, r.bytes_per_output_token / 1e6),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("inter-token latency p50 (ms)")
    ax.set_ylabel("cluster TX bytes per generated token (MB)")
    ax.set_title("Dense vs MoE: network traffic vs decode latency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=.3)
    save(fig, out, "P7_dense_vs_moe.png")


def p8_composition(m, out):
    """Predicted traffic split into TP all-reduce / EP all-to-all / PP."""
    cols = ["pred_tp_bytes", "pred_ep_bytes", "pred_pp_bytes"]
    if not all(c in m.columns for c in cols):
        return
    d = m[(m.workload == "decode")].dropna(subset=cols)
    if d.empty:
        return
    d = steady(d).sort_values(["model", "label"])
    x = range(len(d))
    fig, ax = plt.subplots(figsize=(max(8.5, 0.7 * len(d) + 4), 5))
    bottom = [0.0] * len(d)
    for c, lbl, col in zip(cols,
                           ["TP all-reduce", "EP all-to-all", "PP boundary"],
                           ["#1f77b4", "#d62728", "#2ca02c"]):
        vals = (d[c] / 1e6).tolist()
        ax.bar(list(x), vals, bottom=bottom, label=lbl, color=col, width=.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(d.label, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("predicted MB per token")
    ax.set_title("Predicted traffic composition by parallelism mechanism")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=.3)
    save(fig, out, "P8_traffic_composition.png")



def p9_traffic_timeseries(root, m, out, colors):
    """Cluster TX rate over time for representative cells — shows burstiness
    and the prefill-vs-decode traffic profile that aggregate bytes hide."""
    import glob as _glob
    d = m[m.workload.isin(["decode", "prefill"])]
    if d.empty:
        return
    # one representative cell per (label, workload): highest concurrency
    sel = d.sort_values("concurrency").groupby(["label", "workload"]).tail(1)
    series = []
    for _, r in sel.iterrows():
        files = _glob.glob(os.path.join(root, "*", "runs", r["run"], "*_net.csv"))
        if not files:
            continue
        per_host = []
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if df.empty or "xmit_bytes" not in df:
                continue
            g = df.groupby("ts")["xmit_bytes"].sum().sort_index()
            if len(g) < 3:
                continue
            ts = g.index.to_numpy(dtype=float); by = g.to_numpy(dtype=float)
            dt = ts[1:] - ts[:-1]; db = by[1:] - by[:-1]
            ok = (dt > 0) & (db >= 0)
            if ok.any():
                per_host.append((db[ok] * 8 / 1e9) / dt[ok])
        if not per_host:
            continue
        n = min(len(v) for v in per_host)
        cluster = [sum(v[i] for v in per_host) for i in range(n)]
        series.append((r["label"], r["workload"], r["model"], cluster))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, wl, mod, vals in series:
        ls = "-" if wl == "decode" else "--"
        ax.plot(range(len(vals)), vals, ls=ls, lw=1.3,
                color=colors.get(mod, "#1f77b4"), label=f"{label} ({wl})")
    ax.set_xlabel("time into measured window (s)")
    ax.set_ylabel("cluster TX rate (Gb/s)")
    ax.set_title("Network traffic over time (solid=decode, dashed=prefill)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=.3)
    save(fig, out, "P9_traffic_timeseries.png")

def p4_throughput(m, out, colors, title_suffix=""):
    d = m[(m.workload == "decode") & m.output_tok_per_s.notna()]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5))
    styles = ["-", "--", ":", "-."]
    for i, (label, g) in enumerate(d.groupby("label")):
        g = g.sort_values("concurrency")
        mod = g.model.iloc[0]
        ax.plot(g.concurrency, g.output_tok_per_s, marker="o", ms=4,
                color=colors.get(mod, "#1f77b4"), ls=styles[i % len(styles)],
                label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("output tokens / s")
    ax.set_title("Decode throughput vs concurrency" + title_suffix)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=.3)
    save(fig, out, "P4_throughput_vs_concurrency.png")


def p6_gpu(m, out, colors):
    d = m[(m.workload == "decode") & m.gpu_util_mean.notna()]
    if d.empty:
        return
    d = steady(d).sort_values(["model", "label"])
    fig, ax = plt.subplots(figsize=(max(8.5, 0.7 * len(d) + 3), 5))
    x = range(len(d))
    ax.bar(list(x), d.gpu_util_mean,
           color=[colors.get(mm, "#1f77b4") for mm in d.model])
    ax.set_xticks(list(x))
    ax.set_xticklabels(d.label, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("mean GPU utilization (%)")
    ax.set_title("GPU utilization by configuration (low = waiting on network)")
    ax.grid(True, axis="y", alpha=.3)
    save(fig, out, "P6_gpu_util_by_config.png")


# ---------------------------------------------------------------- per-model
def p2_itl(m, out, title_suffix=""):
    d = m[(m.workload == "decode") & m.itl_ms_p50.notna()]
    if d.empty:
        return
    d = lightest(d).sort_values("label")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = range(len(d))
    ax.bar(list(x), d.itl_ms_p50, color="#1f77b4", width=.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(d.label, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("inter-token latency p50 (ms), concurrency=1")
    ax.set_title("Decode latency by parallelism configuration" + title_suffix)
    ax.grid(True, axis="y", alpha=.3)
    save(fig, out, "P2_decode_itl_by_config.png")


def p3_ttft(m, out, title_suffix=""):
    d = m[(m.workload == "prefill") & m.ttft_s_p50.notna()]
    if d.empty:
        return
    d = lightest(d).sort_values("label")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = range(len(d))
    ax.bar(list(x), d.ttft_s_p50, color="#ff7f0e", width=.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(d.label, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"TTFT p50 (s), input={int(d.input_len.iloc[0])} tok, c=1")
    ax.set_title("Prefill latency by configuration" + title_suffix)
    ax.grid(True, axis="y", alpha=.3)
    save(fig, out, "P3_prefill_ttft_by_config.png")


def p5_imbalance(m, pn, out, title_suffix=""):
    if pn.empty or m.empty:
        return
    sel = steady(m[m.workload == "decode"])
    labels = sel.set_index("run").label.to_dict()
    d = pn[pn.run.isin(labels)]
    if d.empty:
        return
    agg = d.groupby(["run", "host"]).tx_bytes.sum().reset_index()
    runs = sorted(agg.run.unique())
    hosts = sorted(agg.host.unique())
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(runs) + 3), 5))
    w = .8 / max(1, len(hosts))
    for i, h in enumerate(hosts):
        vals = [agg[(agg.run == r) & (agg.host == h)].tx_bytes.sum() / 1e9
                for r in runs]
        ax.bar([j + i * w for j in range(len(runs))], vals, width=w, label=h)
    ax.set_xticks([j + .4 - w / 2 for j in range(len(runs))])
    ax.set_xticklabels([labels[r] for r in runs], rotation=35, ha="right",
                       fontsize=8)
    ax.set_ylabel("TX GB during cell")
    ax.set_title("Per-node transmit volume (routing/load imbalance)" + title_suffix)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=.3)
    save(fig, out, "P5_pernode_imbalance.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/phase1_results"))
    ap.add_argument("--no-per-model", action="store_true",
                    help="skip the per-model figure sets")
    args = ap.parse_args()

    m = load(args.root, "P1_master.csv")
    if m.empty:
        raise SystemExit("P1_master.csv not found — run analyze_phase1.py first")
    pn = load(args.root, "P1_pernode_bytes.csv")
    out = os.path.join(args.root, "tables", "figures")
    os.makedirs(out, exist_ok=True)
    colors = model_colors(m)

    # cumulative
    p1_validation(m, out, colors)
    p7_dense_vs_moe(m, out, colors)
    p8_composition(m, out)
    p4_throughput(m, out, colors)
    p6_gpu(m, out, colors)
    p9_traffic_timeseries(args.root, m, out, colors)
    # single-model runs: also emit the per-config figures at top level
    if m.model.nunique() == 1:
        p2_itl(m, out)
        p3_ttft(m, out)
        p5_imbalance(m, pn, out)

    # per-model sets (only meaningful with several models)
    if not args.no_per_model and m.model.nunique() > 1:
        for mod, g in m.groupby("model"):
            sub = os.path.join(out, "by_model", str(mod).replace("/", "_"))
            sfx = f"\n{mod}"
            p2_itl(g, sub, sfx)
            p3_ttft(g, sub, sfx)
            p4_throughput(g, sub, colors, sfx)
            p5_imbalance(g, pn, sub, sfx)

    print("figures written to", out)
    for root_, _, files in os.walk(out):
        for f in sorted(files):
            print("  ", os.path.relpath(os.path.join(root_, f), out))


if __name__ == "__main__":
    main()
