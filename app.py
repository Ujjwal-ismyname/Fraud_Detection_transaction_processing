import matplotlib
matplotlib.use("Agg")  # non-GUI backend (MANDATORY)

from flask import Flask, render_template, request
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os

from preprocess import preprocess
from sklearn.metrics import precision_score, recall_score, f1_score


app = Flask(__name__)

MODEL_PATH = "model/fraud_lgbm.pkl"
FRAUD_THRESHOLD = 0.7 

assert os.path.exists(MODEL_PATH), "Model file not found. Train model first."
model = joblib.load(MODEL_PATH)


def plot_prob_distribution(probs):
    plt.figure()
    plt.hist(probs, bins=30)
    plt.title("Fraud Probability Distribution (Uploaded Data)")
    plt.xlabel("Fraud Probability")
    plt.ylabel("Count")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    table = None
    stats = None
    graph = None
    eval_metrics = None
    plots = True

    if request.method == "POST":
        start = time.time()

        file = request.files["file"]
        df = pd.read_csv(file)

        has_label = "isFraud" in df.columns

        # ---------- PREPROCESS ----------
        X, y_true = preprocess(df, training=has_label)

        # ---------- PREDICTION ----------
        probs = model.predict_proba(X)[:, 1]
        df["FraudProbability"] = probs
        df["Prediction"] = (probs >= FRAUD_THRESHOLD).astype(int)

        latency = round(time.time() - start, 3)

        # ---------- STATISTICS ----------
        stats = {
            "Total Transactions": len(df),
            "High Risk Transactions": int((df["FraudProbability"] >= FRAUD_THRESHOLD).sum()),
            "Average Fraud Risk": round(df["FraudProbability"].mean(), 4),
            "Latency (sec)": latency
        }

        # ---------- EVALUATION (ONLY IF LABELED) ----------
        if has_label:
            eval_metrics = {
                "Precision": round(precision_score(y_true, df["Prediction"]), 3),
                "Recall": round(recall_score(y_true, df["Prediction"]), 3),
                "F1-Score": round(f1_score(y_true, df["Prediction"]), 3)
            }

        # ---------- HIGH-RISK SAMPLE ----------
        high_risk = df[df["FraudProbability"] >= FRAUD_THRESHOLD]
        table = (
            high_risk
            .sort_values("FraudProbability", ascending=False)
            .head(20)
            .to_html(index=False)
        )

        graph = plot_prob_distribution(probs)

    return render_template(
        "index.html",
        table=table,
        stats=stats,
        graph=graph,
        eval_metrics=eval_metrics,
        plots=plots
    )

if __name__ == "__main__":
    app.run(debug=True)