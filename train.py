"""Train and compare defensive intrusion-detection classifiers."""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["duration", "src_bytes", "dst_bytes", "packet_count", "failed_logins", "connection_rate", "syn_ratio"]

def load_data(path):
    df = pd.read_csv(path)
    missing = set(FEATURES + ["label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df

def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

def train_models(dataset="data/network_traffic.csv", model_path="models/best_model.joblib", metrics_path="models/metrics.json"):
    df = load_data(dataset)
    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    models = {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=7, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(y_test, model.predict(X_test))
        fitted[name] = model
    best_name = max(results, key=lambda name: results[name]["f1"])
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[best_name], "features": FEATURES, "name": best_name}, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)
    return best_name, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/network_traffic.csv")
    args = parser.parse_args()
    best, results = train_models(args.dataset)
    print("MODEL COMPARISON")
    for name, m in results.items():
        print(f"{name:20} accuracy={m['accuracy']:.3f} precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
    print(f"Best model by F1: {best}")
    print("Metrics are based only on the bundled synthetic teaching dataset.")

if __name__ == "__main__":
    main()
