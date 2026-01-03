import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve
)

from preprocess import preprocess

DATA_PATH = "data/train.csv"
MODEL_PATH = "model/fraud_lgbm.pkl"
PLOT_DIR = "static/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)

print("Preprocessing...")
X, y = preprocess(df, training=True)

print("Train-validation split...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training LightGBM model...")
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.log_evaluation(period=0)]
)

print("Evaluating...")
y_prob = model.predict_proba(X_val)[:, 1]
y_pred = (y_prob > 0.9).astype(int)

auc_score = roc_auc_score(y_val, y_prob)
print(f"Validation AUC: {auc_score:.4f}")

fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"{PLOT_DIR}/roc_curve.png")
plt.close()

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format="d")
plt.title("Confusion Matrix (Threshold = 0.9)")
plt.savefig(f"{PLOT_DIR}/confusion_matrix.png")
plt.close()


precision, recall, _ = precision_recall_curve(y_val, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig(f"{PLOT_DIR}/precision_recall_curve.png")
plt.close()

plt.figure()
plt.hist(y_prob[y_val == 0], bins=50, alpha=0.6, label="Legit")
plt.hist(y_prob[y_val == 1], bins=50, alpha=0.6, label="Fraud")
plt.xlabel("Predicted Fraud Probability")
plt.ylabel("Count")
plt.legend()
plt.title("Model Confidence Distribution")
plt.savefig(f"{PLOT_DIR}/confidence_distribution.png")
plt.close()

evals_result = model.evals_result_

plt.figure()
plt.plot(evals_result["valid_0"]["auc"])
plt.xlabel("Boosting Iterations")
plt.ylabel("AUC")
plt.title("Learning Curve (Validation AUC)")
plt.savefig(f"{PLOT_DIR}/learning_curve_auc.png")
plt.close()

print("Saving model...")
joblib.dump(model, MODEL_PATH)

print("Training complete.")
print(f"All evaluation plots saved to {PLOT_DIR}/")