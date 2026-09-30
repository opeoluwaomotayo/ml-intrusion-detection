from pathlib import Path
from generate_dataset import generate
from train import FEATURES, load_data, train_models
from predict import predict

def test_generation(tmp_path):
    path = tmp_path / "traffic.csv"
    assert generate(str(path), n=100) == 100
    df = load_data(path)
    assert set(FEATURES + ["label"]).issubset(df.columns)
    assert set(df["label"]) == {0, 1}

def test_training_and_prediction(tmp_path):
    data = tmp_path / "traffic.csv"
    model = tmp_path / "model.joblib"
    metrics = tmp_path / "metrics.json"
    generate(str(data), n=160)
    best, results = train_models(str(data), str(model), str(metrics))
    assert best in results
    assert model.exists() and metrics.exists()
    for values in results.values():
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert 0 <= values[key] <= 1
    result = predict({"duration": 2, "src_bytes": 500, "dst_bytes": 900, "packet_count": 20, "failed_logins": 0, "connection_rate": 2, "syn_ratio": 0.2}, str(model))
    assert result["classification"] in {"NORMAL-LIKE", "INTRUSION-LIKE"}
    assert 0 <= result["intrusion_probability"] <= 1
