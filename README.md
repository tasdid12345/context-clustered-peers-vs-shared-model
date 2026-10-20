# Peer-context clustering vs. a shared model on Covertype

Motivation: In "Causeways: Building Bridges with Context in Peer-To-Peer Federated Learning" (2025), peers with different visit contexts are argued to benefit from context-aware grouping rather than one global model. This repository is a small, self-contained test of that idea on a public tabular dataset.

Question: Does grouping heterogeneous peers by visit-context similarity and training one model per group beat a single shared model on every peer's held-out data, without leaving low-data peers worse off?

Dataset: Covertype from OpenML (id 150), subsampled for CPU speed. It is not the paper's own dataset. Peers are formed by dominant wilderness area, and each peer gets a stratified held-out split.

Baseline: one shared MLP trained on pooled peer training data (a FedAvg-style stand-in, not the paper's algorithm).

Improved model: k-means clusters peers by their context summaries (per-peer feature statistics). One MLP is trained per cluster, and each peer's prediction blends its cluster model with its own local model, weighted toward the local model when the peer has more samples.

Main metric is macro-F1 on the pooled test split. Secondary metrics are accuracy and the worst per-peer accuracy.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts go to `figures/`).

## Reference

This project is motivated by the following paper:

> M Geçer, B Garbinato (2025). *Causeways: Building Bridges with Context in Peer-To-Peer Federated Learning*. 2025 International Joint Conference on. https://ieeexplore.ieee.org/abstract/document/11229017/

Paper author: Prof. Benoît Garbinato, University of Lausanne (benoit.garbinato@unil.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
