from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "employee_attrition_cleaned.csv"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

features = ["OverTime", "JobSatisfaction", "MonthlyIncome", "YearsAtCompany", "WorkLifeBalance"]
target = "Attrition"
df = pd.read_csv(DATA)

X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = {
    name: float(cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean())
    for name, model in models.items()
}
best_name = max(cv_f1, key=cv_f1.get)
best_model = models[best_name]
best_model.fit(X_train, y_train)

pred = best_model.predict(X_test)
report = {
    "selected_model": best_name,
    "cross_validation_f1_train": cv_f1,
    "test_accuracy": float(accuracy_score(y_test, pred)),
    "test_precision_attrition": float(precision_score(y_test, pred, zero_division=0)),
    "test_recall_attrition": float(recall_score(y_test, pred, zero_division=0)),
    "test_f1_attrition": float(f1_score(y_test, pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "classification_report": classification_report(
        y_test, pred, target_names=["Stayed", "Left"], zero_division=0, output_dict=True
    )
}
joblib.dump(best_model, MODEL_DIR / "attrition_model.joblib")
(REPORT_DIR / "model_metrics.json").write_text(json.dumps(report, indent=2))
print(f"Selected model: {best_name}")
print(f"Test accuracy: {report['test_accuracy']:.3f}")
print(f"Attrition F1-score: {report['test_f1_attrition']:.3f}")
print("Saved model to models/attrition_model.joblib")