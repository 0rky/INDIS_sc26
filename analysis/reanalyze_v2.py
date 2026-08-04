#!/usr/bin/env python3
"""
reanalyze_v2.py — corrected Phase 1 analysis.

Fixes three problems found in the first pass and adds the analyses the data
actually supports:

  FIX 1  bytes/token used total_output_tokens as the denominator, but the byte
         counters also cover (a) the 4 warm-up requests and (b) every prefill
         forward pass. Corrected denominator = "token-passes" =
         (num_prompts + warmup) x (input_len + output_len).
  FIX 2  gpu_util_mean averaged over all four nodes including IDLE ones, so it
         read ~24% for 1-node configs and ~95% for 4-node configs purely as an
         artifact. Corrected: average over ACTIVE nodes only.
  FIX 3  the analytical model counts PAYLOAD bytes; NCCL's LL protocol puts
         2 bytes on the wire per byte of payload (LL128 ~1.07x). We now
         classify each cell by the Phase 0 protocol regime and report the
         protocol-corrected comparison.

New analyses:
  * memory-bandwidth roofline: predicted tokens/s from weight bytes read per
    node per token vs measured (explains why TP speeds decode up)
  * rail-B utilisation share per config/concurrency (production confirmation of
    Phase 0's rail-scaling threshold)
  * fabric utilisation (measured Gb/s vs line rate)

Usage:
  python3 reanalyze_v2.py --root <dir containing tables/> [--line-rate-gbps 200]
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WARMUP = 4                      # bench_client default
NUM_PROMPTS = {1: 8, 8: 32, 32: 64}   # from run_matrix CELLS
MEM_BW_GBPS = 273.0             # GB10 unified LPDDR5x, GB/s

# model geometry + weight bytes actually READ per token (fp16)
MODELS = {
    "Llama-3.1-8B-Instruct":       dict(h=4096, L=32, gb_per_token=16.0,  moe=False),
    "Llama-3.3-70B-Instruct":      dict(h=8192, L=80, gb_per_token=140.0, moe=False),
    "Qwen3-30B-A3B-Instruct-2507": dict(h=2048, L=48, gb_per_token=6.0,   moe=True),
}
COLORS = {"Llama-3.1-8B-Instruct": "#1f77b4",
          "Llama-3.3-70B-Instruct": "#ff7f0e",
          "Qwen3-30B-A3B-Instruct-2507": "#2ca02c"}


def phase0_protocol(size):
    """NCCL algo/proto regime measured in Phase 0 (4-node dual-rail)."""
    if size <= 2048:      return "Tree/LL"
    if size < 262144:     return "Ring/LL"
    if size < 4194304:    return "Ring/LL128"
    return "Ring/Simple"

PROTO_WIRE = {"Tree/LL": 2.0, "Ring/LL": 2.0, "Ring/LL128": 1.067, "Ring/Simple": 1.0}


def enrich(root):
    t = os.path.join(root, "tables")
    m = pd.read_csv(os.path.join(t, "P1_master.csv"))
    pn = pd.read_csv(os.path.join(t, "P1_pernode_bytes.csv"))

    m["nodes"] = m.tp * m.pp
    m["num_prompts"] = m.concurrency.map(NUM_PROMPTS)
    m["token_passes"] = (m.num_prompts + WARMUP) * (m.input_len + m.output_len)
    m["bytes_per_token_pass"] = m.wire_tx_bytes / m.token_passes

    m["h"] = m.model.map({k: v["h"] for k, v in MODELS.items()})
    m["layers"] = m.model.map({k: v["L"] for k, v in MODELS.items()})
    m["gb_per_token_total"] = m.model.map({k: v["gb_per_token"] for k, v in MODELS.items()})

    # decode all-reduce message size = batch x hidden x 2 bytes
    m["msg_bytes"] = m.concurrency * m.h * 2
    m["phase0_proto"] = m.msg_bytes.map(phase0_protocol)
    m["wire_factor"] = m.phase0_proto.map(PROTO_WIRE)
    m["pred_wire_bytes"] = m.pred_bytes_per_token * m.wire_factor
    m["ratio_raw"] = m.bytes_per_token_pass / m.pred_bytes_per_token
    m["ratio_proto_corrected"] = m.bytes_per_token_pass / m.pred_wire_bytes

    # FIX 2: GPU util over ACTIVE nodes only (analysis averaged over all 4)
    m["gpu_util_active"] = m.gpu_util_mean * 4.0 / m.nodes

    # memory-bandwidth roofline (TP shards weights per node; PP does not)
    #   TP=n  -> each node reads total/n per token
    #   PP=p  -> each node reads total/p, but the token traverses all p stages
    #            sequentially, so the CRITICAL PATH still reads the full model
    m["gb_read_critical_path"] = m.gb_per_token_total / m.tp
    m["roofline_tok_s"] = MEM_BW_GBPS / m.gb_read_critical_path
    m["measured_tok_s"] = 1000.0 / m.itl_ms_p50
    m["roofline_efficiency"] = m.measured_tok_s / m.roofline_tok_s

    # rail split
    idx = m.set_index("run")[["label", "workload", "concurrency", "tp", "pp", "ep", "nodes"]]
    pn = pn.join(idx, on="run")
    rail = pn.groupby(["run", "dev"]).tx_bytes.sum().unstack().fillna(0)
    if "roceP2p1s0f0" in rail and "rocep1s0f0" in rail:
        rail["railB_share"] = 100 * rail["roceP2p1s0f0"] / (
            rail["roceP2p1s0f0"] + rail["rocep1s0f0"]).replace(0, pd.NA)
        m = m.merge(rail[["railB_share"]], left_on="run", right_index=True, how="left")

    # per-node imbalance (max/min tx among ACTIVE nodes)
    imb = {}
    for run, g in pn.groupby("run"):
        s = g.groupby("host").tx_bytes.sum()
        s = s[s > 0]
        imb[run] = (s.max() / s.min()) if len(s) > 1 and s.min() > 0 else None
    m["node_imbalance"] = m.run.map(imb)
    return m, pn


def fig_roofline(m, out):
    d = m[(m.workload == "decode") & (m.concurrency == 1)]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for model, g in d.groupby("model"):
        ax.scatter(g.roofline_tok_s, g.measured_tok_s, s=90,
                   color=COLORS.get(model, "#777"), edgecolors="k",
                   linewidths=.5, label=model.split("-Instruct")[0])
        for _, r in g.iterrows():
            ax.annotate(r.label, (r.roofline_tok_s, r.measured_tok_s), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
    lim = max(d.roofline_tok_s.max(), d.measured_tok_s.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="memory-bandwidth roofline")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("roofline tokens/s  =  273 GB/s  /  (weights read per node per token)")
    ax.set_ylabel("measured tokens/s (concurrency=1)")
    ax.set_title("Decode is memory-bandwidth bound; TP raises the roofline, PP does not")
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "R1_memory_roofline.png"), dpi=150)
    plt.close(fig); print("  R1_memory_roofline.png")


def fig_speedup(m, out):
    d = m[(m.workload == "decode") & (m.concurrency == 1)]
    rows = []
    for model, g in d.groupby("model"):
        b = g[(g.tp == 1) & (g.pp == 1)]
        if b.empty:
            continue
        base = b.itl_ms_p50.iloc[0]
        for _, r in g.iterrows():
            rows.append(dict(model=model, label=r.label, tp=r.tp, pp=r.pp,
                             speedup=base / r.itl_ms_p50,
                             kind=("TP" if r.tp > 1 and r.pp == 1 else
                                   "PP" if r.pp > 1 and r.tp == 1 else
                                   "TP+PP" if r.tp > 1 else "single")))
    if not rows:
        return
    t = pd.DataFrame(rows).sort_values(["model", "speedup"])
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = {"single": "#999", "PP": "#d62728", "TP": "#1f77b4", "TP+PP": "#9467bd"}
    x = range(len(t))
    ax.bar(list(x), t.speedup, color=[cmap[k] for k in t.kind])
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks(list(x)); ax.set_xticklabels(t.label, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("decode speed-up vs single node (concurrency=1)")
    ax.set_title("Tensor parallelism accelerates decode; pipeline parallelism does not")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in cmap.items()], fontsize=8)
    ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "R2_tp_vs_pp_speedup.png"), dpi=150)
    plt.close(fig); print("  R2_tp_vs_pp_speedup.png")


def fig_rail(m, out):
    if "railB_share" not in m.columns:
        return
    d = m[(m.workload == "decode") & m.railB_share.notna() & (m.nodes > 1)]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for model, g in d.groupby("model"):
        gg = g.groupby("msg_bytes").railB_share.mean()
        ax.plot(gg.index, gg.values, "o-", ms=6,
                color=COLORS.get(model, "#777"), label=model.split("-Instruct")[0])
    ax.axhline(50, color="g", ls="--", lw=1, label="balanced (50%)")
    ax.axhline(0, color="r", ls=":", lw=1, label="single-rail only")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("all-reduce message size (bytes)")
    ax.set_ylabel("share of traffic carried by rail B (%)")
    ax.set_title("The second rail is barely used until messages grow\n"
                 "(production confirmation of the Phase 0 rail-scaling threshold)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "R3_rail_utilisation.png"), dpi=150)
    plt.close(fig); print("  R3_rail_utilisation.png")


def fig_protocol(m, out):
    d = m[(m.workload == "decode") & (m.tp > 1) & m.ratio_raw.notna()]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for proto, g in d.groupby("phase0_proto"):
        ax.scatter(g.msg_bytes, g.ratio_raw, s=70, label=f"{proto} (expect {PROTO_WIRE[proto]}x)",
                   edgecolors="k", linewidths=.4)
    ax.axhline(2.0, color="#1f77b4", ls="--", lw=1, label="LL protocol: 2 wire bytes / payload byte")
    ax.axhline(1.067, color="#ff7f0e", ls="--", lw=1, label="LL128: 1.07x")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("all-reduce message size (bytes)")
    ax.set_ylabel("measured wire bytes / predicted payload bytes")
    ax.set_title("Measured traffic exceeds payload by the NCCL protocol's wire factor")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "R4_protocol_overhead.png"), dpi=150)
    plt.close(fig); print("  R4_protocol_overhead.png")


def fig_utilisation(m, out, line_rate):
    d = m[m.workload.isin(["decode", "prefill"]) & m.net_gbps_mean.notna()]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for wl, mk in (("decode", "o"), ("prefill", "s")):
        g = d[d.workload == wl]
        ax.scatter(g.concurrency, 100 * g.net_gbps_peak / line_rate, marker=mk,
                   s=60, alpha=.75, label=f"{wl} (peak)")
        ax.scatter(g.concurrency, 100 * g.net_gbps_mean / line_rate, marker=mk,
                   s=30, alpha=.45, label=f"{wl} (mean)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency"); ax.set_ylabel(f"% of {line_rate:.0f} Gb/s fabric capacity")
    ax.set_title("The fabric is far from saturated during inference")
    ax.legend(fontsize=8); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "R5_fabric_utilisation.png"), dpi=150)
    plt.close(fig); print("  R5_fabric_utilisation.png")



def fig_transport(m, out, tdir):
    """TCP vs dual-rail RoCE for configs measured under both transports."""
    d = m[m.workload == "decode"]
    if d["mode"].nunique() < 2:
        print("  (only one transport present - skipping R6)")
        return
    piv = d.pivot_table(index=["label", "concurrency"], columns="mode",
                        values=["itl_ms_p50", "output_tok_per_s",
                                "bytes_per_token_pass"])
    rows = []
    for (label, conc), r in piv.iterrows():
        try:
            itl_d = r[("itl_ms_p50", "dual")]; itl_t = r[("itl_ms_p50", "tcp")]
            tps_d = r[("output_tok_per_s", "dual")]; tps_t = r[("output_tok_per_s", "tcp")]
        except KeyError:
            continue
        if pd.isna(itl_d) or pd.isna(itl_t):
            continue
        rows.append(dict(label=label, concurrency=conc,
                         itl_roce=itl_d, itl_tcp=itl_t,
                         itl_penalty_x=round(itl_t / itl_d, 2),
                         tps_roce=tps_d, tps_tcp=tps_t,
                         tps_ratio=round(tps_d / tps_t, 2) if tps_t else None))
    if not rows:
        print("  (no matched dual/tcp pairs - skipping R6)")
        return
    tt = pd.DataFrame(rows).sort_values(["label", "concurrency"])
    tt.to_csv(os.path.join(tdir, "TBC2_transport_comparison.csv"), index=False)
    print("  wrote TBC2_transport_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    lbl = [f"{r.label}\nc={int(r.concurrency)}" for _, r in tt.iterrows()]
    x = range(len(tt))
    axes[0].bar([i - .2 for i in x], tt.itl_roce, width=.4, label="dual-rail RoCE")
    axes[0].bar([i + .2 for i in x], tt.itl_tcp, width=.4, label="TCP")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(lbl, fontsize=7)
    axes[0].set_ylabel("inter-token latency p50 (ms)")
    axes[0].set_title("Decode latency: RoCE vs TCP")
    axes[0].legend(fontsize=8); axes[0].grid(True, axis="y", alpha=.3)

    axes[1].bar(list(x), tt.itl_penalty_x, color="#d62728")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(lbl, fontsize=7)
    axes[1].set_ylabel("TCP latency / RoCE latency")
    axes[1].set_title("TCP penalty factor\n(Phase 0 microbenchmark predicted ~20x at these message sizes)")
    axes[1].grid(True, axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "R6_transport_comparison.png"), dpi=150)
    plt.close(fig); print("  R6_transport_comparison.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--line-rate-gbps", type=float, default=200.0)
    args = ap.parse_args()

    m, pn = enrich(args.root)
    t = os.path.join(args.root, "tables")
    out = os.path.join(t, "figures_corrected")
    os.makedirs(out, exist_ok=True)

    m.to_csv(os.path.join(t, "P1_master_corrected.csv"), index=False)
    print("wrote P1_master_corrected.csv")

    d = m[m.workload == "decode"]
    summ = (d.groupby(["model", "label", "tp", "pp", "ep", "mode"])
              .agg(itl_c1=("itl_ms_p50", "first"),
                   measured_tok_s=("measured_tok_s", "first"),
                   roofline_tok_s=("roofline_tok_s", "first"),
                   roofline_eff=("roofline_efficiency", "first"),
                   MB_per_token_pass=("bytes_per_token_pass", lambda s: s.mean()/1e6),
                   pred_MB=("pred_bytes_per_token", "first"),
                   ratio_raw=("ratio_raw", "mean"),
                   gpu_util_active=("gpu_util_active", "mean"),
                   railB_share=("railB_share", "mean"),
                   node_imbalance=("node_imbalance", "mean"))
              .round(3).reset_index())
    summ.to_csv(os.path.join(t, "TBC1_corrected_summary.csv"), index=False)
    print("wrote TBC1_corrected_summary.csv")

    fig_roofline(m, out); fig_speedup(m, out); fig_rail(m, out)
    fig_protocol(m, out); fig_utilisation(m, out, args.line_rate_gbps)
    fig_transport(m, out, t)

    print("\n=== HEADLINES ===")
    print(f"protocol-regime mean ratio:\n{d[d.tp>1].groupby('phase0_proto').ratio_raw.mean().round(2).to_string()}")
    print(f"\nfabric peak utilisation: {100*m.net_gbps_peak.max()/args.line_rate_gbps:.1f}% of {args.line_rate_gbps:.0f} Gb/s")
    print(f"congestion events (all counters): {int(m[[c for c in m.columns if c.startswith('hw_')]].sum().sum())}")
    print(f"\ntransports present: {sorted(m['mode'].unique())}")
    print(f"\nfigures in {out}")


if __name__ == "__main__":
    main()
