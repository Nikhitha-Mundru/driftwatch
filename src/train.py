import joblib
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

df = fetch_openml("adult", version=2, as_frame=True).frame
X = df[FEATURES].astype(float)
y = (df["class"] == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
print("Accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.joblib")
X_train.to_csv("models/train_baseline.csv", index=False)