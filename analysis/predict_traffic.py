#!/usr/bin/env python3
"""
predict_traffic.py — analytical bytes-per-token model for distributed LLM
inference. Reads a HuggingFace config.json and a parallelism layout, predicts
CLUSTER-WIDE TRANSMITTED bytes per token during decode. This is the model that
Phase 1 validates against measured RDMA counters.

Model (activations in fp16, b = 2 bytes; h = hidden_size, L = layers):

  Tensor parallelism, degree t (Megatron-style, 2 all-reduces per layer):
    one ring all-reduce of payload S puts 2*(t-1)*S bytes on the wire
    cluster-wide (each of t ranks transmits 2*(t-1)/t * S).
    dense:      per token  =  L * 2 * 2*(t-1) * h * b
    MoE w/ EP:  attention still TP -> 1 all-reduce/layer = L * 2*(t-1) * h * b

  Pipeline parallelism, p stages:
    (p-1) boundary transfers of h*b per token = (p-1) * h * b

  Expert parallelism over e ranks, top-k experts, L_moe MoE layers:
    dispatch + combine move k expert-activations out and back per MoE layer;
    expected off-rank fraction (e-1)/e under uniform routing:
    per token = L_moe * 2 * k * h * b * (e-1)/e

Conventions: predictions are TX-only, cluster-wide (sum over nodes), decode
phase, batch-independent per token. Compare against sum over nodes of
delta(port_xmit_data)*4 divided by generated tokens.

CLI:  python predict_traffic.py --config /opt/models/X/config.json \
          --tp 4 --pp 1 [--ep] [--nodes 4]
"""
import argparse, json


def load_cfg(path):
    with open(path) as f:
        cfg = json.load(f)
    # some configs nest under "text_config" (multimodal wrappers)
    if "hidden_size" not in cfg and "text_config" in cfg:
        cfg = cfg["text_config"]
    return cfg


def moe_params(cfg):
    """Return (num_experts, top_k, n_moe_layers) or (0,0,0) for dense."""
    n_exp = (cfg.get("num_local_experts") or cfg.get("num_experts")
             or cfg.get("n_routed_experts") or 0)
    top_k = (cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token")
             or cfg.get("moe_top_k") or 0)
    L = cfg["num_hidden_layers"]
    if not n_exp or not top_k:
        return 0, 0, 0
    # some models interleave dense layers; default: all layers MoE unless
    # config exposes an interleave/step pattern
    step = cfg.get("moe_layer_freq") or cfg.get("decoder_sparse_step") or 1
    first_dense = cfg.get("first_k_dense_replace") or 0
    n_moe = max(0, (L - first_dense)) // max(1, step)
    return n_exp, top_k, n_moe


def predict(cfg, tp, pp, ep, act_bytes=2):
    """Return dict of predicted cluster-wide TX bytes/token by component."""
    h = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    n_exp, top_k, n_moe = moe_params(cfg)
    is_moe = n_exp > 0

    out = dict(hidden_size=h, layers=L, is_moe=is_moe,
               num_experts=n_exp, top_k=top_k, moe_layers=n_moe)

    ar_per_layer = 2 * (tp - 1) * h * act_bytes if tp > 1 else 0
    if is_moe and ep and tp > 1:
        # attention TP all-reduce only (MoE FFN handled by EP all-to-all)
        tp_bytes = L * ar_per_layer
        e = tp  # vLLM EP size = TP size (single DP group)
        a2a = n_moe * 2 * top_k * h * act_bytes * (e - 1) / e if e > 1 else 0
    else:
        # dense TP, or MoE run in pure-TP mode (experts sharded, still 2 AR/layer)
        tp_bytes = L * 2 * ar_per_layer
        a2a = 0.0

    pp_bytes = (pp - 1) * h * act_bytes if pp > 1 else 0

    out.update(
        tp_allreduce_bytes_per_token=tp_bytes,
        ep_alltoall_bytes_per_token=a2a,
        pp_boundary_bytes_per_token=pp_bytes,
        total_bytes_per_token=tp_bytes + a2a + pp_bytes,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.json")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    res = predict(cfg, args.tp, args.pp, 1 if args.ep else 0)
    for k, v in res.items():
        if "bytes" in k:
            print(f"{k:36s} {v/1e6:10.3f} MB/token")
        else:
            print(f"{k:36s} {v}")


if __name__ == "__main__":
    main()
