"""Peer-context clustering vs. one shared model on OpenML Covertype (CPU, ~60k-row subsample).

Peers = dominant wilderness area. Baseline = one MLP on pooled peer train data.
Improved = one MLP per k-means cluster of peer context summaries, blended per peer
with that peer's own local MLP (local weight grows with the peer's sample count).
"""
import io, json, sys, time
import numpy as np, pandas as pd, requests
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED, SEEDS = 42, [42, 43, 44]
N_SUB = 60_000          # CPU-friendly subsample of the 581k rows
N_PRIOR = 1000          # shrinkage prior strength (fixed a priori, not tuned on test)
CLASSES = np.arange(1, 8)
META_URL = "https://www.openml.org/api/v1/json/data/150"


def get(url, tries=5):
    """HTTP GET with timeout=60 and exponential backoff on 429/5xx or network errors."""
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
                return r
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = str(e)
        print(f"  retry {i + 1} after error: {err}")
        time.sleep(2 ** i)
    raise RuntimeError(f"download failed: {url}")


def load_covertype():
    """Read the OpenML ARFF (data section is CSV) into a DataFrame."""
    meta = get(META_URL).json()["data_set_description"]
    print("downloading Covertype ARFF ...")
    lines = get(meta["url"]).text.splitlines()
    names = [l.split()[1].strip("'\"") for l in lines if l[:10].lower() == "@attribute"]
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == "@data") + 1
    return pd.read_csv(io.StringIO("\n".join(lines[start:])), header=None, names=names)


def proba(model, X):
    """Predicted probabilities expanded to all 7 classes (local models may miss some)."""
    out = np.zeros((len(X), len(CLASSES)))
    out[:, model.classes_ - 1] = model.predict_proba(X)
    return out


def mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(64,), max_iter=30, early_stopping=True, random_state=seed)


def main():
    df = load_covertype().sample(n=N_SUB, random_state=SEED).reset_index(drop=True)
    print(f"subsampled to {len(df)} rows")
    feats = [c for c in df.columns if c != "class"]
    wild = [c for c in feats if c.startswith("Wilderness_Area")]
    numeric = [c for c in feats if c not in wild]
    X = df[feats].to_numpy(float)
    y = df["class"].to_numpy(int)
    peer = df[wild].to_numpy().argmax(1)          # peer = dominant wilderness area
    peers = np.unique(peer)

    # ONE stratified 25% test split per peer (seeded); pooled train/test = concatenation
    split = {p: train_test_split(np.where(peer == p)[0], test_size=0.25,
                                 stratify=y[peer == p], random_state=SEED) for p in peers}
    tr_all = np.concatenate([split[p][0] for p in peers])
    te_all = np.concatenate([split[p][1] for p in peers])
    peer_te, yt = peer[te_all], y[te_all]
    print("peer sizes (train/test):", {int(p): (len(split[p][0]), len(split[p][1])) for p in peers})

    scaler = StandardScaler().fit(X[tr_all])       # fit on training rows only
    Xs = scaler.transform(X)

    # Context summaries: per-peer mean and std of numeric features (training rows only)
    num_idx = [feats.index(c) for c in numeric]
    S = np.array([np.r_[Xs[split[p][0]][:, num_idx].mean(0), Xs[split[p][0]][:, num_idx].std(0)]
                  for p in peers])
    Z = StandardScaler().fit_transform(S)
    # Choose number of clusters from {2,3} by silhouette on training-side summaries (no test data)
    fits = {k: KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z) for k in (2, 3)}
    k = max(fits, key=lambda k: silhouette_score(Z, fits[k]))
    labels = fits[k]
    cluster_of = dict(zip(peers, labels))
    print(f"context clusters (k={k}):", {int(p): int(c) for p, c in cluster_of.items()})

    def score(pred):
        accs = [np.mean(pred[peer_te == p] == yt[peer_te == p]) for p in peers]
        return {"f1": f1_score(yt, pred, average="macro"),
                "acc": accuracy_score(yt, pred), "worst": min(accs)}

    def run(seed):
        """One seed: fit baseline and improved models, return their test scores."""
        # BASELINE: single shared MLP on pooled peer training data
        pred_base = mlp(seed).fit(Xs[tr_all], y[tr_all]).predict(Xs[te_all])
        # IMPROVED: one MLP per context cluster (union of member peers' training data)
        cl_model = {}
        for c in np.unique(labels):
            idx = np.concatenate([split[p][0] for p in peers[labels == c]])
            cl_model[c] = mlp(seed).fit(Xs[idx], y[idx])
        # ...then each peer's model = local MLP blended with its cluster MLP.
        # Local weight lam = n_p / (n_p + N_PRIOR): small peers lean on the cluster model.
        pred_imp = np.empty(len(te_all), dtype=int)
        for p in peers:
            tr, te = split[p]
            local = mlp(seed).fit(Xs[tr], y[tr])
            lam = len(tr) / (len(tr) + N_PRIOR)
            prob = lam * proba(local, Xs[te]) + (1 - lam) * proba(cl_model[cluster_of[p]], Xs[te])
            pred_imp[peer_te == p] = CLASSES[prob.argmax(1)]
        return score(pred_base), score(pred_imp)

    runs = []
    for s in SEEDS:
        print(f"seed {s}: fitting baseline and improved models ...")
        runs.append(run(s))
    B = [r[0] for r in runs]
    I = [r[1] for r in runs]

    def agg(rs, key):
        v = np.array([r[key] for r in rs])
        return float(v.mean()), float(v.std())

    bf, bf_sd = agg(B, "f1"); imf, imf_sd = agg(I, "f1")
    ba, _ = agg(B, "acc"); ia, _ = agg(I, "acc")
    bw, _ = agg(B, "worst"); iw, _ = agg(I, "worst")
    vals = [bf, imf, ba, ia, bw, iw]
    if not all(np.isfinite(vals)):
        sys.exit("non-finite metrics; cannot produce meaningful results")

    results = {
        "question": "Does grouping heterogeneous peers by visit-context similarity and training one "
                    "model per group beat a single shared model on every peer's held-out data, "
                    "without leaving low-data peers worse off?",
        "method": "Covertype subsample; peers = dominant wilderness area; per-peer stratified 25% test "
                  "split; k-means on per-peer context summaries; cluster MLPs blended per peer with "
                  "local MLPs; seeds 42-44.",
        "dataset": {"name": "Covertype (OpenML id 150)", "source_url": "https://www.openml.org/d/150",
                    "n_rows": int(len(df)), "n_train": int(len(tr_all)), "n_test": int(len(te_all)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "macro-F1 on pooled test split",
        "higher_is_better": True,
        "baseline": "One shared MLPClassifier (64 hidden units) trained on pooled peer training data "
                    "(FedAvg-style stand-in).",
        "improved": "One MLP per k-means cluster of peer context summaries (k chosen by silhouette), "
                    "each peer's prediction = local MLP blended with its cluster MLP, local weight "
                    "n_p/(n_p+1000).",
        "metrics": {
            "baseline_score": bf, "improved_score": imf, "improvement": imf - bf,
            "baseline_score_std": bf_sd, "improved_score_std": imf_sd,
            "baseline_accuracy": ba, "improved_accuracy": ia,
            "baseline_worst_peer_acc": bw, "improved_worst_peer_acc": iw,
            "n_peers": int(len(peers)), "n_clusters": int(k),
        },
        "notes": "Only 4 peers (wilderness areas), so clustering and silhouette selection are very "
                 "coarse. The MLP is a stand-in for FedAvg, not the paper's algorithm. The shrinkage "
                 "prior (1000) was fixed a priori and not tuned; no hyperparameter was tuned on the "
                 "test set. Shrinkage direction: small peers lean on the cluster model, large peers "
                 "on their local model. The improved model fits more models (cluster + local), so it "
                 "uses more compute. Single data split; results are from one subsample.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Optional bar chart of the three metrics (means over seeds)
    import os
    os.makedirs("figures", exist_ok=True)
    labels_ = ["macro-F1", "accuracy", "worst-peer acc"]
    x = np.arange(3)
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, [bf, ba, bw], 0.4, label="baseline (shared)")
    plt.bar(x + 0.2, [imf, ia, iw], 0.4, label="improved (clusters)")
    plt.xticks(x, labels_); plt.ylim(0, 1); plt.legend(); plt.title("Covertype: test-set metrics")
    plt.tight_layout(); plt.savefig("figures/baseline_vs_improved.png", dpi=120)
    print("done; wrote results.json")


if __name__ == "__main__":
    main()
