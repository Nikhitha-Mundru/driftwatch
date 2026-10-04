# DriftWatch: ML Model Monitoring Dashboard

A drift detection system that monitors data distribution shifts in production. Trains a Random Forest classifier to predict income (>$50K), serves it via FastAPI, and detects when incoming data no longer matches the training distribution.

## What it does

1. **Train:** Random Forest on Adult Income dataset (6 features, binary classification)
2. **Serve:** FastAPI endpoint that returns predictions and probabilities
3. **Monitor:** Detects drift using PSI (Population Stability Index) and KS test
4. **Alert:** Flags features that have shifted significantly

## Results

| Metric | Value |
| --- | --- |
| Model accuracy (test set) | 0.809 |
| Features monitored | 6 |
| Features that drifted (simulated test) | age, hours-per-week |
| Max PSI (hours-per-week) | 10.74 |
| KS test p-value | 0.0 |
| API latency (avg over 5 calls) | 2063 ms |

## How to use

**1. Install and train:**
```bash
python -m pip install scikit-learn pandas joblib scipy fastapi uvicorn
python src/train.py