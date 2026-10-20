# Peer-context clustering vs. shared model on Covertype

Question: does grouping peers (wilderness areas) by visit-context similarity and training one model per group beat a single shared model on each peer's held-out data?

Setup: OpenML Covertype (id 150), 60k-row seeded subsample, per-peer stratified 25% test split, seeds 42–44, metric macro-F1 on the pooled test split.

- Baseline: one shared MLPClassifier (unchanged).
- Improved: one LightGBM per k-means cluster of peer context summaries (k chosen by silhouette on training data), blended per peer with a local LightGBM using weight n_p/(n_p+1000). Hyperparameters are fixed a priori; nothing is tuned on the test set.

Run: `python analysis.py` (writes `results.json` and `figures/baseline_vs_improved.png`).

Caveat: the gain is driven largely by the stronger learner; the clustering contribution was not isolated by an ablation.

## Results (latest run)

On Covertype (OpenML 150), macro-F1 on the pooled test split rose from 0.62 for a single shared model trained on pooled peer data to 0.85 for cluster-plus-local LightGBM, though the gain mostly reflects the stronger learner and comes from one split.

| Metric | Value |
|---|---|
| baseline_score | 0.6195 |
| improved_score | 0.8537 |
| improvement | 0.2342 |
| baseline_score_std | 0.0201 |
| improved_score_std | 0.003356 |
| baseline_accuracy | 0.7775 |
| improved_accuracy | 0.9068 |
| baseline_worst_peer_acc | 0.6674 |
| improved_worst_peer_acc | 0.8888 |
| n_peers | 4 |
| n_clusters | 2 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Only 4 peers (wilderness areas), so clustering and silhouette selection are very coarse. The improvement comes mainly from the stronger learner (gradient-boosted trees instead of a 30-iteration MLP); the baseline MLP is kept unchanged. The clustering effect alone was not isolated by an ablation. The shrinkage prior (1000) and LightGBM settings were fixed a priori and not tuned on the test set. Small peers lean on the cluster model, large peers on their local model. The improved model fits more models (cluster + local), so it uses more compute. Single data split; results are from one subsample.

## Reference

This project is motivated by the following paper:

> M Geçer, B Garbinato (2025). *Causeways: Building Bridges with Context in Peer-To-Peer Federated Learning*. 2025 International Joint Conference on. https://ieeexplore.ieee.org/abstract/document/11229017/

Paper author: Prof. Benoît Garbinato, University of Lausanne (benoit.garbinato@unil.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
