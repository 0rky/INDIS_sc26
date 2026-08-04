#!/bin/bash
# /opt/phase0/env.sh — shared testbed constants. Copy to ALL nodes.
# EDIT GID after Stage 1.7 (RoCE v2 GID index, must be identical on all nodes/devices).

export GID=3                       # <-- EDIT after discovery (Stage 1.7)

export DEVA=rocep1s0f0             # rail A RDMA device
export DEVB=roceP2p1s0f0           # rail B RDMA device
export IFA=enp1s0f0np0             # rail A netdev (192.168.100.0/24)
export IFB=enP2p1s0f0np0           # rail B netdev (192.168.101.0/24)

export NCCL_BIN=/opt/nccl-tests/build
export P0=/opt/phase0
