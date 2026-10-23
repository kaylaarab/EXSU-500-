"""
Step 6: Ensemble (weighted blend) of CatBoost + XGBoost + Logistic Regression.

Trains the three models with their best-known hyperparameters, finds blend
weights that maximize val AUPRC, then evaluates the blend once on test.
"""
import itertools
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier

TARGET = "readmit_30d"
ID_COLS = ["patient_nbr"]

train = pd.read_csv("data/train.csv", keep_default_na=False)
val = pd.read_csv("data/val.csv", keep_default_na=False)
test = pd.read_csv("data/test.csv", keep_default_na=False)

y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]
X_train = train.drop(columns=[TARGET] + ID_COLS)
X_val = val.drop(columns=[TARGET] + ID_COLS)
X_test = test.drop(columns=[TARGET] + ID_COLS)

cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# --- Logistic Regression ---
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
])
logreg.fit(X_train, y_train)
logreg_val = logreg.predict_proba(X_val)[:, 1]
logreg_test = logreg.predict_proba(X_test)[:, 1]
print("Logistic Regression trained.")

# --- XGBoost (best params from 02_train_evaluate.py) ---
xgb = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(
        max_depth=5, n_estimators=400, learning_rate=0.03, eval_metric="aucpr",
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
        random_state=42, n_jobs=-1,
    )),
])
xgb.fit(X_train, y_train)
xgb_val = xgb.predict_proba(X_val)[:, 1]
xgb_test = xgb.predict_proba(X_test)[:, 1]
print("XGBoost trained.")

# --- CatBoost (best params from 05_catboost.py) ---
X_train_cb, X_val_cb, X_test_cb = X_train.copy(), X_val.copy(), X_test.copy()
for df_ in (X_train_cb, X_val_cb, X_test_cb):
    for c in cat_cols:
        df_[c] = df_[c].astype(str)
train_pool = Pool(X_train_cb, y_train, cat_features=cat_cols)
val_pool = Pool(X_val_cb, cat_features=cat_cols)
test_pool = Pool(X_test_cb, cat_features=cat_cols)
catboost = CatBoostClassifier(
    depth=6, learning_rate=0.03, iterations=791, loss_function="Logloss",
    eval_metric="AUC", scale_pos_weight=pos_weight, random_seed=42, verbose=False,
)
catboost.fit(train_pool)
cb_val = catboost.predict_proba(val_pool)[:, 1]
cb_test = catboost.predict_proba(test_pool)[:, 1]
print("CatBoost trained.\n")

# --- Find blend weights that maximize val AUPRC (grid search, step 0.1) ---
preds_val = {"logreg": logreg_val, "xgb": xgb_val, "catboost": cb_val}
preds_test = {"logreg": logreg_test, "xgb": xgb_test, "catboost": cb_test}
names = list(preds_val.keys())

best_auprc, best_weights = -1, None
steps = [i / 10 for i in range(11)]
for w1, w2 in itertools.product(steps, steps):
    w3 = round(1 - w1 - w2, 2)
    if w3 < 0 or w3 > 1:
        continue
    weights = {names[0]: w1, names[1]: w2, names[2]: w3}
    blend_val = sum(weights[n] * preds_val[n] for n in names)
    auprc = average_precision_score(y_val, blend_val)
    if auprc > best_auprc:
        best_auprc, best_weights = auprc, weights

print(f"Best blend weights (by val AUPRC): {best_weights}, val_AUPRC={best_auprc:.3f}\n")

blend_test = sum(best_weights[n] * preds_test[n] for n in names)
y_pred = (blend_test >= 0.5).astype(int)
auroc = roc_auc_score(y_test, blend_test)
auprc = average_precision_score(y_test, blend_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sens, spec = tp / (tp + fn), tn / (tn + fp)

print(f"Ensemble (weighted blend)           AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
      f"Sens={sens:.3f}  Spec={spec:.3f}  (thr=0.5)")

rng = np.random.default_rng(42)
y_test_arr = y_test.to_numpy()
boot_aurocs = []
for _ in range(1000):
    idx = rng.integers(0, len(y_test_arr), len(y_test_arr))
    if len(np.unique(y_test_arr[idx])) < 2:
        continue
    boot_aurocs.append(roc_auc_score(y_test_arr[idx], blend_test[idx]))
ci_lo, ci_hi = np.percentile(boot_aurocs, [2.5, 97.5])
print(f"Ensemble test AUROC: {auroc:.3f} (95% bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}])")

print(f"\nFor comparison -- individual models on test:")
print(f"  Logistic Regression: AUROC={roc_auc_score(y_test, logreg_test):.3f}")
print(f"  XGBoost:             AUROC={roc_auc_score(y_test, xgb_test):.3f}")
print(f"  CatBoost:            AUROC={roc_auc_score(y_test, cb_test):.3f}")

results_df = pd.read_csv("results/model_comparison.csv")
new_row = pd.DataFrame([{
    "model": "Ensemble (CatBoost+XGBoost+LogReg blend)", "auroc": auroc, "auprc": auprc,
    "sensitivity": sens, "specificity": spec,
}])
results_df = results_df[results_df["model"] != "Ensemble (CatBoost+XGBoost+LogReg blend)"]
results_df = pd.concat([results_df, new_row], ignore_index=True)
results_df.to_csv("results/model_comparison.csv", index=False)
print("\nUpdated results/model_comparison.csv")
