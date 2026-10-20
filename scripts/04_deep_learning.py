"""
Step 4: Deep learning model (MLP) for 30-day readmission prediction.

Uses the same preprocessing (same train/val/test splits, same feature
encoding) as the classical models, so the comparison is fair. Trained with
early stopping on validation AUPRC, evaluated once on the held-out test set.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

torch.manual_seed(42)
np.random.seed(42)

TARGET = "readmit_30d"
ID_COLS = ["patient_nbr"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]
X_train = train.drop(columns=[TARGET] + ID_COLS)
X_val = val.drop(columns=[TARGET] + ID_COLS)
X_test = test.drop(columns=[TARGET] + ID_COLS)

cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
X_train_t = preprocess.fit_transform(X_train).astype(np.float32)
X_val_t = preprocess.transform(X_val).astype(np.float32)
X_test_t = preprocess.transform(X_test).astype(np.float32)
if hasattr(X_train_t, "toarray"):
    X_train_t, X_val_t, X_test_t = X_train_t.toarray(), X_val_t.toarray(), X_test_t.toarray()

n_features = X_train_t.shape[1]
print(f"Input dimension after encoding: {n_features}")


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


X_train_d = to_tensor(X_train_t, DEVICE)
y_train_d = to_tensor(y_train.to_numpy(), DEVICE)
X_val_d = to_tensor(X_val_t, DEVICE)
X_test_d = to_tensor(X_test_t, DEVICE)

model = MLP(n_features).to(DEVICE)
pos_weight = torch.tensor(
    (y_train == 0).sum() / (y_train == 1).sum(), dtype=torch.float32, device=DEVICE
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

batch_size = 512
n_train = X_train_d.shape[0]
best_val_auprc, best_state, patience, patience_counter = -1, None, 10, 0
max_epochs = 100

for epoch in range(max_epochs):
    model.train()
    perm = torch.randperm(n_train, device=DEVICE)
    epoch_loss = 0.0
    for i in range(0, n_train, batch_size):
        idx = perm[i:i + batch_size]
        xb, yb = X_train_d[idx], y_train_d[idx]
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(idx)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_d)
        val_prob = torch.sigmoid(val_logits).cpu().numpy()
    val_auprc = average_precision_score(y_val, val_prob)

    if val_auprc > best_val_auprc:
        best_val_auprc, best_state, patience_counter = val_auprc, model.state_dict().copy(), 0
    else:
        patience_counter += 1

    if epoch % 5 == 0 or patience_counter == 0:
        print(f"Epoch {epoch:3d}  train_loss={epoch_loss/n_train:.4f}  val_AUPRC={val_auprc:.4f}"
              f"{'  (best)' if patience_counter == 0 else ''}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
        break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_logits = model(X_test_d)
    test_prob = torch.sigmoid(test_logits).cpu().numpy()

y_pred = (test_prob >= 0.5).astype(int)
auroc = roc_auc_score(y_test, test_prob)
auprc = average_precision_score(y_test, test_prob)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sens, spec = tp / (tp + fn), tn / (tn + fp)

print(f"\nMLP (deep learning)                 AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
      f"Sens={sens:.3f}  Spec={spec:.3f}  (thr=0.5)")

# Bootstrap 95% CI, same protocol as the XGBoost evaluation
rng = np.random.default_rng(42)
y_test_arr = y_test.to_numpy()
boot_aurocs = []
for _ in range(1000):
    idx = rng.integers(0, len(y_test_arr), len(y_test_arr))
    if len(np.unique(y_test_arr[idx])) < 2:
        continue
    boot_aurocs.append(roc_auc_score(y_test_arr[idx], test_prob[idx]))
ci_lo, ci_hi = np.percentile(boot_aurocs, [2.5, 97.5])
print(f"MLP test AUROC: {auroc:.3f} (95% bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}])")

# Append to the shared results table
results_df = pd.read_csv("results/model_comparison.csv")
new_row = pd.DataFrame([{
    "model": "MLP (deep learning)", "auroc": auroc, "auprc": auprc,
    "sensitivity": sens, "specificity": spec,
}])
results_df = results_df[results_df["model"] != "MLP (deep learning)"]
results_df = pd.concat([results_df, new_row], ignore_index=True)
results_df.to_csv("results/model_comparison.csv", index=False)
print("\nUpdated results/model_comparison.csv")
