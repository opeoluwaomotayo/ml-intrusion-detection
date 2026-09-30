"""Classify a user-supplied network-flow feature record using the saved model."""
import argparse
import joblib
import pandas as pd

def predict(record, model_path="models/best_model.joblib"):
    bundle = joblib.load(model_path)
    frame = pd.DataFrame([[record[f] for f in bundle["features"]]], columns=bundle["features"])
    pred = int(bundle["model"].predict(frame)[0])
    prob = float(bundle["model"].predict_proba(frame)[0][1])
    return {"classification": "INTRUSION-LIKE" if pred else "NORMAL-LIKE", "intrusion_probability": prob, "model": bundle["name"]}

def main():
    p = argparse.ArgumentParser(description="Classify an authorized/synthetic network-flow record")
    for name in ["duration", "src_bytes", "dst_bytes", "packet_count", "failed_logins", "connection_rate", "syn_ratio"]:
        p.add_argument(f"--{name.replace('_','-')}", type=float, required=True)
    a = p.parse_args()
    record = {k: getattr(a, k) for k in ["duration", "src_bytes", "dst_bytes", "packet_count", "failed_logins", "connection_rate", "syn_ratio"]}
    result = predict(record)
    print(f"Classification: {result['classification']}")
    print(f"Model probability: {result['intrusion_probability']:.1%}")
    print(f"Model: {result['model']}")
    print("Educational result only; do not use as a sole production security control.")

if __name__ == "__main__":
    main()
