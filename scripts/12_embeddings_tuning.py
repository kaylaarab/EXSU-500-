"""
Step 12: Can entity embeddings or more hyperparameter tuning give a further gain?

Fair comparison under a fixed budget, tuned on VALIDATION only; test scored once.
  1. MLP on one-hot features       (tuned, same budget)         <- like script 04, but tuned
  2. MLP with entity embeddings    (tuned, same budget)         <- EmbPred30's idea, leak-free protocol
  3. CatBoost, wider random search (vs the script-05 default)
Each neural model: random search over RANDOM_CONFIGS configs (seed 0), best config
refit with 3 seeds. Paired patient-level bootstrap for the differences that matter.
Uses the same patient-level splits as scripts 02-06 (run 01_preprocess.py first).
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

TARGET, DEVICE = "readmit_30d", ("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = bool(os.environ.get("SMOKE"))               # SMOKE=1: tiny run to check the code only
RANDOM_CONFIGS, N_SEEDS, MAX_EPOCHS, PATIENCE = (1, 1, 1, 1) if SMOKE else (8, 3, 60, 8)
rng = np.random.default_rng(0)

# Checkpoint cache so a machine restart does not lose finished work (data/ is git-ignored)
CACHE = Path("data/cache_12_smoke.json" if SMOKE else "data/cache_12.json")
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def cached(key, fn):
    if key not in cache:
        cache[key] = fn()
        CACHE.write_text(json.dumps(cache))
    return cache[key]


def at_half(yt, p):
    """accuracy / sensitivity / specificity at threshold 0.5 (always report accuracy WITH sensitivity)"""
    yp = p >= 0.5
    tp, tn = ((yp == 1) & (yt == 1)).sum(), ((yp == 0) & (yt == 0)).sum()
    return dict(accuracy=(tp + tn) / len(yt), sensitivity=tp / (yt == 1).sum(), specificity=tn / (yt == 0).sum())

tr, va, te = [pd.read_csv(f"data/{s}.csv", keep_default_na=False) for s in ("train", "val", "test")]
y = {k: d[TARGET].to_numpy().astype(np.float32) for k, d in (("tr", tr), ("va", va), ("te", te))}
groups = te["patient_nbr"].to_numpy()
X = {k: d.drop(columns=[TARGET, "patient_nbr"]) for k, d in (("tr", tr), ("va", va), ("te", te))}
cats = X["tr"].select_dtypes(exclude="number").columns.tolist()
nums = [c for c in X["tr"].columns if c not in cats]
SKEWED = [c for c in ["number_outpatient", "number_emergency", "number_inpatient", "total_prior_visits", "num_med_changes"] if c in nums]

# integer codes per categorical column; 0 = unseen/unknown
vocab = {c: {v: i + 1 for i, v in enumerate(sorted(X["tr"][c].unique()))} for c in cats}
card = [len(vocab[c]) + 1 for c in cats]
mu, sd = {}, {}
def num_matrix(df, fit=False):
    M = df[nums].astype(float).copy()
    M[SKEWED] = np.log1p(M[SKEWED])
    if fit:
        mu["m"], sd["m"] = M.mean(), M.std().replace(0, 1)
    return ((M - mu["m"]) / sd["m"]).to_numpy(np.float32)
N = {"tr": num_matrix(X["tr"], True), "va": num_matrix(X["va"]), "te": num_matrix(X["te"])}
C = {k: np.stack([X[k][c].map(vocab[c]).fillna(0).astype(int).to_numpy() for c in cats], 1) for k in X}
def onehot(codes):
    return np.concatenate([np.eye(card[j], dtype=np.float32)[codes[:, j]] for j in range(len(cats))], 1)
OH = {k: np.concatenate([N[k], onehot(C[k])], 1) for k in X}


class Net(nn.Module):
    def __init__(self, kind, hidden, drop, emb_drop):
        super().__init__()
        self.kind = kind
        if kind == "emb":
            dims = [min(50, int(round(1.6 * n ** 0.56))) for n in card]   # rule used by EmbPred30
            self.embs = nn.ModuleList([nn.Embedding(n, d) for n, d in zip(card, dims)])
            self.edrop = nn.Dropout(emb_drop)
            n_in = N["tr"].shape[1] + sum(dims)
        else:
            n_in = OH["tr"].shape[1]
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(drop)]
            prev = h
        self.net = nn.Sequential(*layers, nn.Linear(prev, 1))

    def forward(self, num, cat, oh):
        if self.kind == "emb":
            e = self.edrop(torch.cat([m(cat[:, j]) for j, m in enumerate(self.embs)], 1))
            return self.net(torch.cat([num, e], 1)).squeeze(1)
        return self.net(oh).squeeze(1)


def tensors(k):
    return (torch.tensor(N[k]).to(DEVICE), torch.tensor(C[k]).to(DEVICE), torch.tensor(OH[k]).to(DEVICE))

T = {k: tensors(k) for k in ("tr", "va", "te")}
Y = {k: torch.tensor(y[k]).to(DEVICE) for k in y}


def predict(model, k):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(*T[k])).cpu().numpy()


def fit(kind, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(kind, cfg["hidden"], cfg["drop"], cfg["emb_drop"]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y["tr"] == 0).sum() / (y["tr"] == 1).sum()).to(DEVICE))
    best, state, bad = -1, None, 0
    n = len(y["tr"])
    for ep in range(MAX_EPOCHS):
        m.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            if len(b) < 2: continue
            opt.zero_grad()
            lossf(m(T["tr"][0][b], T["tr"][1][b], T["tr"][2][b]), Y["tr"][b]).backward()
            opt.step()
        score = average_precision_score(y["va"], predict(m, "va"))
        if score > best: best, state, bad = score, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    m.load_state_dict(state)
    return m, best


def sample_cfg():
    return dict(hidden=[(256, 128), (128, 64), (512, 256, 128)][rng.integers(3)],
                drop=float(rng.choice([0.2, 0.3, 0.5])), lr=float(rng.choice([3e-4, 1e-3, 2e-3])),
                wd=float(rng.choice([1e-5, 1e-4, 1e-3])), emb_drop=float(rng.choice([0.1, 0.3])))


cfgs = [sample_cfg() for _ in range(RANDOM_CONFIGS)]
rows, preds = [], {}


def pboot(p1, p0, n=1000):
    d = pd.DataFrame({"y": y["te"], "a": p1, "b": p0, "g": groups})
    idx = [v.index.to_numpy() for _, v in d.groupby("g")]
    out, r = [], np.random.default_rng(1)
    for _ in range(n):
        s = d.iloc[np.concatenate([idx[i] for i in r.integers(0, len(idx), len(idx))])]
        if s.y.nunique() > 1: out.append(roc_auc_score(s.y, s.a) - roc_auc_score(s.y, s.b))
    return np.percentile(out, [2.5, 97.5])


for kind, name in (("onehot", "MLP one-hot (tuned)"), ("emb", "MLP entity embeddings (tuned)")):
    results = []
    for i, cfg in enumerate(cfgs):
        v = cached(f"{kind}_cfg{i}", lambda: float(fit(kind, cfg, 0)[1]))
        results.append((v, cfg)); print(f"  [{name}] config {i}: val AUPRC={v:.4f} {cfg}")
    bestcfg = max(results, key=lambda t: t[0])[1]
    ps = [np.array(cached(f"{kind}_seed{s}", lambda: predict(fit(kind, bestcfg, s)[0], "te").tolist())) for s in range(N_SEEDS)]
    p = np.mean(ps, 0); preds[name] = p
    aucs = [roc_auc_score(y["te"], q) for q in ps]
    rows.append(dict(model=name, auroc_seed_mean=np.mean(aucs), auroc_seed_sd=np.std(aucs),
                     auroc_seed_ensemble=roc_auc_score(y["te"], p), auprc_seed_ensemble=average_precision_score(y["te"], p),
                     **at_half(y["te"], p), best_config=str(bestcfg)))
    print(f"{name:34s} AUROC per seed {np.round(aucs, 3)}  seed-ensemble={rows[-1]['auroc_seed_ensemble']:.3f}")

lo, hi = pboot(preds["MLP entity embeddings (tuned)"], preds["MLP one-hot (tuned)"])
print(f"paired AUROC, embeddings minus one-hot: {roc_auc_score(y['te'], preds['MLP entity embeddings (tuned)']) - roc_auc_score(y['te'], preds['MLP one-hot (tuned)']):+.4f} CI [{lo:+.4f}, {hi:+.4f}]")

# CatBoost: default (script 05) vs wider random search
Xc = {k: X[k].copy() for k in X}
for k in Xc:
    for c in cats: Xc[k][c] = Xc[k][c].astype(str)
pool = {k: Pool(Xc[k], y[k], cat_features=cats) for k in Xc}
w = (y["tr"] == 0).sum() / (y["tr"] == 1).sum()
def cb(depth, lr, l2, oh):
    return cached(f"cb_{depth}_{lr}_{l2}_{oh}", lambda: cb_fit(depth, lr, l2, oh))


def cb_fit(depth, lr, l2, oh):
    m = CatBoostClassifier(depth=depth, learning_rate=lr, l2_leaf_reg=l2, one_hot_max_size=oh, iterations=20 if SMOKE else 1000,
                           loss_function="Logloss", eval_metric="AUC", scale_pos_weight=w, random_seed=42,
                           verbose=False, early_stopping_rounds=50)
    m.fit(pool["tr"], eval_set=pool["va"])
    return [average_precision_score(y["va"], m.predict_proba(pool["va"])[:, 1]), m.predict_proba(pool["te"])[:, 1].tolist()]
bv, base = cb(4, 0.1, 3, 2)
best, bestv, bestp = base, bv, (4, 0.1, 3, 2)
for _ in range(RANDOM_CONFIGS):
    p_ = (int(rng.choice([4, 6, 8])), float(rng.choice([0.03, 0.05, 0.1])), float(rng.choice([1, 3, 10, 20])), int(rng.choice([2, 10])))
    v, m = cb(*p_); print(f"  [CatBoost] {p_}: val AUPRC={v:.4f}")
    if v > bestv: best, bestv, bestp = m, v, p_
pb, pt = np.array(base), np.array(best)
lo, hi = pboot(pt, pb)
rows += [dict(model="CatBoost default (depth 4, lr 0.1)", auroc_seed_ensemble=roc_auc_score(y["te"], pb), auprc_seed_ensemble=average_precision_score(y["te"], pb), **at_half(y["te"], pb)),
         dict(model="CatBoost tuned", auroc_seed_ensemble=roc_auc_score(y["te"], pt), auprc_seed_ensemble=average_precision_score(y["te"], pt), **at_half(y["te"], pt), best_config=str(bestp))]
print(f"CatBoost default AUROC={rows[-2]['auroc_seed_ensemble']:.3f}; tuned {bestp} AUROC={rows[-1]['auroc_seed_ensemble']:.3f}; paired diff CI [{lo:+.4f}, {hi:+.4f}]")
if not SMOKE:
    pd.DataFrame(rows).to_csv("5_results/ablations/embeddings_tuning.csv", index=False)
print("Saved 5_results/ablations/embeddings_tuning.csv")
