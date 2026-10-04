import pandas as pd
import requests
import time
from drift import detect_drift

# Load baseline
baseline = pd.read_csv("models/train_baseline.csv")
features = baseline.columns.tolist()

# Simulate drift: shift ages upward
new_data = baseline.copy()
new_data["age"] = new_data["age"] + 10  # Simulate 10-year age shift
new_data["hours-per-week"] = new_data["hours-per-week"] * 0.9  # Reduce work hours

# Detect drift
drift_results = detect_drift(baseline, new_data, features)

print("\n=== Drift Detection Results ===")
drifted_features = []
for feat, result in drift_results.items():
    print(f"\n{feat}:")
    print(f"  PSI: {result['psi']} {'[DRIFTED]' if result['drifted'] else ''}")
    print(f"  KS test p-value: {result['ks_pvalue']}")
    if result['drifted']:
        drifted_features.append(feat)

print(f"\n=== Summary ===")
print(f"Features drifted: {drifted_features}")
print(f"Drift detection lag: monitoring every 100 predictions")

# Send 5 predictions through the API to measure latency
api_url = "http://localhost:8000/predict"
latencies = []
print(f"\n=== API Latency Test (5 predictions) ===")
for i in range(5):
    sample = new_data.iloc[i].to_dict()
    payload = {
        "age": float(sample["age"]),
        "fnlwgt": float(sample["fnlwgt"]),
        "education_num": float(sample["education-num"]),
        "capital_gain": float(sample["capital-gain"]),
        "capital_loss": float(sample["capital-loss"]),
        "hours_per_week": float(sample["hours-per-week"])
    }
    start = time.time()
    resp = requests.post(api_url, json=payload)
    latency = (time.time() - start) * 1000
    latencies.append(latency)
    print(f"Request {i+1}: {latency:.2f} ms, prediction: {resp.json()['prediction']}")

print(f"\nAverage latency: {sum(latencies) / len(latencies):.2f} ms")