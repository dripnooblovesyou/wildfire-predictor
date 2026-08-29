import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("data/processed/training_features.csv")
print("Loaded:", df.shape)

# drop rows with any missing values (points where NDVI or weather rasters had no coverage)
df = df.dropna()
print("After dropping missing values:", df.shape)

# feature columns for training - excludes x/y/lon/lat (location, not predictive on their own),
# year (would let the model just memorize fire years), aspect (superseded by northness/eastness),
# and label (the target). wind_u/wind_v/dewpoint are raw components already summarized by
# windspeed/humidity, so those are dropped too.
FEATURES = [
    "elevation", "slope", "northness", "eastness",
    "ndvi_06", "ndvi_07", "ndvi_08", "ndvi_09",
    "temp_06", "temp_07", "temp_08", "temp_09",
    "precip_06", "precip_07", "precip_08", "precip_09",
    "windspeed_06", "windspeed_07", "windspeed_08", "windspeed_09",
    "humidity_06", "humidity_07", "humidity_08", "humidity_09",
    "ndvi_anom_06", "ndvi_anom_07", "ndvi_anom_08", "ndvi_anom_09",
]

FEATURES_PRESEASON = [
    "elevation", "slope", "northness", "eastness",
    "ndvi_06", "ndvi_anom_06",
    "temp_06", "precip_06", "windspeed_06", "humidity_06",
]

FEATURES_CLEAN = [
    "elevation", "slope", "northness", "eastness",
    "ndvi_06", "ndvi_07",
    "ndvi_anom_06", "ndvi_anom_07",
    "vpd_06", "vpd_07",
    "precip_06", "precip_07",
    "windspeed_06", "windspeed_07",
    "dist_to_road",
    "spring_precip",
]

# X = df[FEATURES]
ACTIVE_FEATURES = FEATURES_CLEAN
X = df[ACTIVE_FEATURES]
y = df["label"]

print("Feature count:", len(ACTIVE_FEATURES))
print("X shape:", X.shape)
print("y distribution:")
print(y.value_counts())

BLOCK_SIZE = 100_000  # 100km in meters

# add block coordinates to the dataframe for spatial cross-validation
df["block_x"] = (df["x"] // BLOCK_SIZE).astype(int)
df["block_y"] = (df["y"] // BLOCK_SIZE).astype(int)
df["block_id"] = df["block_x"].astype(str) + "_" + df["block_y"].astype(str)

print("Number of unique blocks:", df["block_id"].nunique())
print("Points per block:")
print(df["block_id"].value_counts())

# set up GroupKFold cross-validation by block_id, so that all points in the same spatial block are either in the training or test set, but not both
# this prevents data leakage from nearby points and gives a more realistic estimate of model performance on unseen areas.
groups = df["block_id"]
gkf = GroupKFold(n_splits=5)

# preview the folds
for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    train_fires = y.iloc[train_idx].sum()
    test_fires = y.iloc[test_idx].sum()
    print(f"Fold {i+1}: train={len(train_idx)} ({train_fires} fires), "
          f"test={len(test_idx)} ({test_fires} fires)")

auc_roc_scores = []
auc_pr_scores = []

for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)

    auc_roc_scores.append(roc)
    auc_pr_scores.append(pr)
    print(f"Fold {i+1}: AUC-ROC={roc:.3f}, AUC-PR={pr:.3f}")

print()
print(f"Mean AUC-ROC: {np.mean(auc_roc_scores):.3f} ± {np.std(auc_roc_scores):.3f}")
print(f"Mean AUC-PR:  {np.mean(auc_pr_scores):.3f} ± {np.std(auc_pr_scores):.3f}")

# ---- Temporal validation: train on past, test on recent ----
print("\n=== TEMPORAL VALIDATION ===")

train_mask = df["year"] <= 2018
test_mask = df["year"] >= 2020

X_train_t = X[train_mask]
y_train_t = y[train_mask]
X_test_t = X[test_mask]
y_test_t = y[test_mask]

print(f"Train (≤2018): {len(X_train_t)} points, {y_train_t.sum()} fires")
print(f"Test (≥2020):  {len(X_test_t)} points, {y_test_t.sum()} fires")

model_t = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5,
    random_state=42, eval_metric="logloss",
)
model_t.fit(X_train_t, y_train_t)

probs_t = model_t.predict_proba(X_test_t)[:, 1]
print(f"Temporal AUC-ROC: {roc_auc_score(y_test_t, probs_t):.3f}")
print(f"Temporal AUC-PR:  {average_precision_score(y_test_t, probs_t):.3f}")

# train one final model on all the data (same hyperparameters as the CV folds)
final_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,
    random_state=42,
    eval_metric="logloss",
)
final_model.fit(X, y)

# pair feature names with their importances and sort, most important first
importances = sorted(
    zip(ACTIVE_FEATURES, final_model.feature_importances_),
    key=lambda pair: pair[1],
    reverse=True,
)

print()
print("Feature importances:")
for name, importance in importances:
    print(f"  {name}: {importance:.4f}")

# use a sample for speed
X_sample = X.sample(2000, random_state=42)

# TreeExplainer is optimized for tree models like XGBoost
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_sample)

print("SHAP values shape:", shap_values.shape)

# mean absolute SHAP value per feature = overall importance
import numpy as np
mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance = sorted(
    zip(ACTIVE_FEATURES, mean_abs_shap),
    key=lambda pair: pair[1],
    reverse=True,
)

print("\nSHAP feature importance (mean |SHAP|):")
for name, val in shap_importance:
    print(f"  {name}: {val:.4f}")

shap.summary_plot(shap_values, X_sample, feature_names=ACTIVE_FEATURES, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=120, bbox_inches="tight")
print("Saved shap_summary.png")

import joblib
joblib.dump(final_model, "data/processed/fire_model.pkl")
print("Saved model")