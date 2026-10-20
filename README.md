# Peer-context clustering vs. shared model on Covertype

Question: does grouping peers (wilderness areas) by visit-context similarity and training one model per group beat a single shared model on each peer's held-out data?

Setup: OpenML Covertype (id 150), 60k-row seeded subsample, per-peer stratified 25% test split, seeds 42–44, metric macro-F1 on the pooled test split.

- Baseline: one shared MLPClassifier (unchanged).
- Improved: one LightGBM per k-means cluster of peer context summaries (k chosen by silhouette on training data), blended per peer with a local LightGBM using weight n_p/(n_p+1000). Hyperparameters are fixed a priori; nothing is tuned on the test set.

Run: `python analysis.py` (writes `results.json` and `figures/baseline_vs_improved.png`).

Caveat: the gain is driven largely by the stronger learner; the clustering contribution was not isolated by an ablation.

## Reference

This project is motivated by the following paper:

> M Geçer, B Garbinato (2025). *Causeways: Building Bridges with Context in Peer-To-Peer Federated Learning*. 2025 International Joint Conference on. https://ieeexplore.ieee.org/abstract/document/11229017/

Paper author: Prof. Benoît Garbinato, University of Lausanne (benoit.garbinato@unil.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
