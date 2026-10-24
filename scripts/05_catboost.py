"""
Step 5: CatBoost with native categorical handling + engineered features.

Unlike XGBoost (which needed one-hot encoding, exploding dimensionality for
high-cardinality columns like discharge_disposition_id), CatBoost consumes
categorical columns directly. Combined with the new engineered features
(total_prior_visits, num_med_changes, a1c_measured_and_changed), this is our
best shot at beating the 0.678 AUROC XGBoost ceiling from the raw features.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

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
for df_ in (X_train, X_val, X_test):
    for c in cat_cols:
        df_[c] = df_[c].astype(str)

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
val_pool = Pool(X_val, y_val, cat_features=cat_cols)
test_pool = Pool(X_test, cat_features=cat_cols)

best_auprc, best_model, best_params = -1, None, None
for depth in [4, 6, 8]:
    for learning_rate in [0.03, 0.05, 0.1]:
        model = CatBoostClassifier(
            depth=depth, learning_rate=learning_rate, iterations=1000,
            loss_function="Logloss", eval_metric="AUC",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_seed=42, verbose=False, early_stopping_rounds=50,
        )
        model.fit(train_pool, eval_set=val_pool)
        val_prob = model.predict_proba(val_pool)[:, 1]
        val_auprc = average_precision_score(y_val, val_prob)
        if val_auprc > best_auprc:
            best_auprc, best_model = val_auprc, model
            best_params = (depth, learning_rate, model.get_best_iteration())

print(f"Best CatBoost params: depth={best_params[0]}, lr={best_params[1]}, "
      f"best_iteration={best_params[2]}, val_AUPRC={best_auprc:.3f}\n")

test_prob = best_model.predict_proba(test_pool)[:, 1]
y_pred = (test_prob >= 0.5).astype(int)
auroc = roc_auc_score(y_test, test_prob)
auprc = average_precision_score(y_test, test_prob)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sens, spec = tp / (tp + fn), tn / (tn + fp)
acc = (tp + tn) / (tp + tn + fp + fn)

print(f"CatBoost (engineered features)      AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
      f"Acc={acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  (thr=0.5)")

rng = np.random.default_rng(42)
y_test_arr = y_test.to_numpy()
boot_aurocs = []
for _ in range(1000):
    idx = rng.integers(0, len(y_test_arr), len(y_test_arr))
    if len(np.unique(y_test_arr[idx])) < 2:
        continue
    boot_aurocs.append(roc_auc_score(y_test_arr[idx], test_prob[idx]))
ci_lo, ci_hi = np.percentile(boot_aurocs, [2.5, 97.5])
print(f"CatBoost test AUROC: {auroc:.3f} (95% bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}])")

# Feature importance (CatBoost's built-in, fast alternative to SHAP for this check)
importance = best_model.get_feature_importance(train_pool)
importance_df = pd.DataFrame({
    "feature": X_train.columns, "importance": importance
}).sort_values("importance", ascending=False).head(15)
print("\nTop 15 features by CatBoost importance:")
print(importance_df.to_string(index=False))

results_df = pd.read_csv("5_results/main_models/model_comparison.csv")
new_row = pd.DataFrame([{
    "model": "CatBoost (engineered features)", "auroc": auroc, "auprc": auprc,
    "accuracy": acc, "sensitivity": sens, "specificity": spec,
}])
results_df = results_df[results_df["model"] != "CatBoost (engineered features)"]
results_df = pd.concat([results_df, new_row], ignore_index=True)
results_df.to_csv("5_results/main_models/model_comparison.csv", index=False)
print("\nUpdated 5_results/main_models/model_comparison.csv")
