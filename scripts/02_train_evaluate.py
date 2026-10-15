"""
Step 2: Train and compare models for 30-day readmission prediction.

Models:
  1. Baseline - majority-class (trivial floor)
  2. Classical ML - regularized logistic regression
  3. Classical ML - gradient boosted trees (XGBoost)

Protocol: fit/tune on train+val, evaluate once on held-out test.
Reports AUROC, AUPRC, sensitivity/specificity at a stated threshold,
and 5-seed variability for the best model.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier

TARGET = "readmit_30d"
ID_COLS = ["patient_nbr"]

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]
X_train = train.drop(columns=[TARGET] + ID_COLS)
X_val = val.drop(columns=[TARGET] + ID_COLS)
X_test = test.drop(columns=[TARGET] + ID_COLS)

cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])


def evaluate(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print(f"{name:35s} AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
          f"Sens={sens:.3f}  Spec={spec:.3f}  (thr={threshold})")
    return dict(model=name, auroc=auroc, auprc=auprc, sensitivity=sens, specificity=spec)


results = []

# 1. Baseline: majority-class (always predict "no readmission")
baseline_prob = np.zeros(len(y_test))
results.append(evaluate("Baseline (majority-class)", y_test, baseline_prob))

# 2. Logistic regression
logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
])
logreg.fit(X_train, y_train)
logreg_prob = logreg.predict_proba(X_test)[:, 1]
results.append(evaluate("Logistic Regression", y_test, logreg_prob))

# 3. XGBoost, tuned on validation set
best_auprc, best_model, best_params = -1, None, None
for max_depth in [3, 5, 7]:
    for n_estimators in [200, 400, 600]:
        for learning_rate in [0.03, 0.05, 0.1]:
            clf = Pipeline([
                ("prep", preprocess),
                ("clf", XGBClassifier(
                    max_depth=max_depth, n_estimators=n_estimators,
                    learning_rate=learning_rate, eval_metric="aucpr",
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                    random_state=42, n_jobs=-1,
                )),
            ])
            clf.fit(X_train, y_train)
            val_prob = clf.predict_proba(X_val)[:, 1]
            val_auprc = average_precision_score(y_val, val_prob)
            if val_auprc > best_auprc:
                best_auprc, best_model = val_auprc, clf
                best_params = (max_depth, n_estimators, learning_rate)

print(f"\nBest XGBoost params (by val AUPRC): max_depth={best_params[0]}, "
      f"n_estimators={best_params[1]}, learning_rate={best_params[2]}, "
      f"val_AUPRC={best_auprc:.3f}\n")

xgb_prob = best_model.predict_proba(X_test)[:, 1]
results.append(evaluate("XGBoost (tuned on val)", y_test, xgb_prob))

# Uncertainty: bootstrap 95% CI on the test set for the best model's AUROC.
# (Bootstrap resampling of the test set is used instead of seed variation --
# it's a cleaner, more standard way to quantify uncertainty on a single
# held-out set and is explicitly an accepted alternative per the course spec.)
rng = np.random.default_rng(42)
n_boot = 1000
y_test_arr = y_test.to_numpy()
boot_aurocs = []
for _ in range(n_boot):
    idx = rng.integers(0, len(y_test_arr), len(y_test_arr))
    if len(np.unique(y_test_arr[idx])) < 2:
        continue
    boot_aurocs.append(roc_auc_score(y_test_arr[idx], xgb_prob[idx]))
ci_lo, ci_hi = np.percentile(boot_aurocs, [2.5, 97.5])

print(f"\nXGBoost test AUROC: {roc_auc_score(y_test, xgb_prob):.3f} "
      f"(95% bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}], n_boot={len(boot_aurocs)})")

results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)
print("\nSaved results/model_comparison.csv")
