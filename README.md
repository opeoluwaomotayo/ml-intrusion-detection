# Machine Learning Network Intrusion Detection

An educational defensive cybersecurity project that compares three machine-learning classifiers on a reproducible **synthetic network-flow dataset** and saves the model with the best F1 score.

## Models

- Logistic Regression
- Random Forest
- Gradient Boosting

## Evaluation

Each model is evaluated using accuracy, precision, recall, F1 score, and a confusion matrix. Results from the bundled data are teaching results only and are not claims of real-world IDS performance.

## Structure

```text
ml-intrusion-detection/
├── .github/workflows/python-tests.yml
├── data/network_traffic.csv
├── models/.gitkeep
├── tests/test_ids.py
├── generate_dataset.py
├── train.py
├── predict.py
├── pytest.ini
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Recreate the Dataset

```bash
python generate_dataset.py
```

## Train and Compare Models

```bash
python train.py
```

This writes the reproducible local artifacts `models/best_model.joblib` and `models/metrics.json`; both are intentionally ignored by Git.

## Example Prediction

```bash
python predict.py --duration 2 --src-bytes 500 --dst-bytes 900 --packet-count 20 --failed-logins 0 --connection-rate 2 --syn-ratio 0.2
```

The classifier accepts feature values only. It does not capture traffic, scan hosts, send packets, or connect to networks.

## Tests / GitHub Actions

```bash
python -m pytest -v
```

The GitHub Actions workflow uses this same command.

## Dataset Notice

`data/network_traffic.csv` is generated locally using a fixed random seed. It contains artificial examples of normal-like and intrusion-like traffic. It is provided for software demonstration and CI testing, not scientific benchmarking.

## Research Upgrade Path

A research version should use an appropriately licensed, documented intrusion-detection dataset; preserve a truly unseen test set; address class imbalance and data leakage; compare models statistically; report precision, recall, F1, ROC-AUC and confusion matrices; and document reproducibility and limitations.

## Ethical Scope

Use only authorized datasets and environments. This repository is designed for defensive detection research and teaching.

## Author

**Dr. Opeoluwa Omotayo Ajilore**  
Lecturer, Computer Science

## License

MIT License. See `LICENSE`.
