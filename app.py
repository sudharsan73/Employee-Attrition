from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "attrition_model.joblib"
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

FEATURES = [
    "OverTime", "JobSatisfaction", "MonthlyIncome",
    "YearsAtCompany", "WorkLifeBalance"
]

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    form = {
        "overtime": "0",
        "satisfaction": "3",
        "income": "5000",
        "years": "5",
        "balance": "3"
    }

    if request.method == "POST":
        try:
            form = {
                "overtime": request.form.get("overtime", "0"),
                "satisfaction": request.form.get("satisfaction", "3"),
                "income": request.form.get("income", ""),
                "years": request.form.get("years", ""),
                "balance": request.form.get("balance", "3")
            }

            row = pd.DataFrame([{
                "OverTime": int(form["overtime"]),
                "JobSatisfaction": int(form["satisfaction"]),
                "MonthlyIncome": float(form["income"]),
                "YearsAtCompany": float(form["years"]),
                "WorkLifeBalance": int(form["balance"])
            }], columns=FEATURES)

            if row.loc[0, "OverTime"] not in [0, 1]:
                raise ValueError("Choose a valid overtime option.")
            if row.loc[0, "JobSatisfaction"] not in [1, 2, 3, 4]:
                raise ValueError("Job satisfaction must be between 1 and 4.")
            if row.loc[0, "MonthlyIncome"] <= 0:
                raise ValueError("Monthly income must be greater than zero.")
            if row.loc[0, "YearsAtCompany"] < 0:
                raise ValueError("Years at company cannot be negative.")
            if row.loc[0, "WorkLifeBalance"] not in [1, 2, 3, 4]:
                raise ValueError("Work-life balance must be between 1 and 4.")

            prediction = int(model.predict(row)[0])
            probability = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else None
            result = {
                "label": "Likely to Leave" if prediction == 1 else "Likely to Stay",
                "left": prediction == 1,
                "probability": probability
            }
        except (ValueError, TypeError) as exc:
            error = str(exc) or "Please enter valid values."

    return render_template("index.html", result=result, error=error, form=form)

if __name__ == "__main__":
    app.run(debug=True)