"""Train and evaluate the document-category classifier."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "document_samples.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA).dropna()
X = df["text"].astype(str)
y = df["label"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

base = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=12000,
                             lowercase=True, strip_accents="unicode")),
    ("svm", LinearSVC(class_weight="balanced"))
])
model = CalibratedClassifierCV(base, cv=3)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
report = classification_report(y_test, pred, output_dict=True, zero_division=0)
matrix = confusion_matrix(y_test, pred, labels=sorted(y.unique()))

joblib.dump(model, MODEL_DIR / "document_classifier.joblib")
metrics = {
    "accuracy": accuracy,
    "classification_report": report,
    "labels": sorted(y.unique()),
    "confusion_matrix": matrix.tolist(),
    "test_samples": int(len(y_test)),
    "note": "Metrics are from the included small teaching dataset. They are not evidence of real-world performance."
}
(MODEL_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Model saved to {MODEL_DIR / 'document_classifier.joblib'}")
print(f"Test accuracy: {accuracy:.3f}")
print(classification_report(y_test, pred, zero_division=0))
print("Evaluation metrics saved to models/evaluation_metrics.json")
