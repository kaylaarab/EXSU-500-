"""
Step 3: Interpretability artifact for the best model (XGBoost).

Produces a SHAP summary plot showing which features push the model's
readmission-risk prediction up or down, and for which patients. This is
the required calibration/interpretability artifact for the manuscript.
"""
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save to file

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TARGET = "readmit_30d"
ID_COLS = ["patient_nbr"]

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

y_train = train[TARGET]
X_train = train.drop(columns=[TARGET] + ID_COLS)
X_test = test.drop(columns=[TARGET] + ID_COLS)

cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# Best params found in 02_train_evaluate.py
clf = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(
        max_depth=5, n_estimators=200, learning_rate=0.05, eval_metric="aucpr",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42, n_jobs=-1,
    )),
])
clf.fit(X_train, y_train)

# Transform a sample of the test set for SHAP (full test set would be slow
# and unnecessary -- 2000 patients is plenty for a stable summary plot)
X_test_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
X_test_transformed = clf.named_steps["prep"].transform(X_test_sample)
feature_names = clf.named_steps["prep"].get_feature_names_out()

explainer = shap.TreeExplainer(clf.named_steps["clf"])
shap_values = explainer.shap_values(X_test_transformed)

# Summary plot: which features matter most, and in which direction
plt.figure()
shap.summary_plot(
    shap_values, X_test_transformed, feature_names=feature_names,
    show=False, max_display=15,
)
plt.tight_layout()
plt.savefig("results/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved results/shap_summary.png")

# Mean |SHAP value| per feature -> simple ranked table, easier to read than the plot
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False).head(15)
importance_df.to_csv("results/shap_feature_importance.csv", index=False)
print("Saved results/shap_feature_importance.csv")
print("\nTop 15 features by mean |SHAP value|:")
print(importance_df.to_string(index=False))
