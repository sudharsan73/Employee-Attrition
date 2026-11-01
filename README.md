# Employee Attrition Prediction – Mini Project

A beginner-friendly machine-learning mini project that predicts whether an employee may leave or stay, using five selected HR features.

## Included files
- `data/employee_attrition_cleaned.csv` – cleaned dataset (1,470 rows, 6 columns)
- `train_model.py` – compares Logistic Regression and Random Forest using 5-fold stratified cross-validation, then evaluates the selected model on a held-out test set
- `models/attrition_model.joblib` – trained model used by the Flask application
- `app.py` – Flask web application
- `templates/index.html` and `static/css/style.css` – website interface
- `static/charts/` – four EDA charts
- `reports/model_metrics.json` – model evaluation results

## Dataset columns
Input features:
1. `OverTime` – 0 = No, 1 = Yes
2. `JobSatisfaction` – scale from 1 to 4
3. `MonthlyIncome` – monthly income value in the source dataset
4. `YearsAtCompany` – years at the company
5. `WorkLifeBalance` – scale from 1 to 4

Target:
- `Attrition` – 0 = Stayed, 1 = Left

## Setup (Windows)
1. Extract the ZIP file.
2. Open the extracted project folder in VS Code.
3. Open Terminal in VS Code.
4. Create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

5. Install packages:

```bash
pip install -r requirements.txt
```

6. Run the web application:

```bash
python app.py
```

7. Open the local address shown in the terminal, usually `http://127.0.0.1:5000`.

## Retrain the model
If you change the CSV or want to reproduce training:

```bash
python train_model.py
```

Then restart the Flask app.

## EDA charts
The four charts are in `static/charts/`:
- Overall attrition distribution (pie chart)
- Attrition by overtime (bar chart)
- Attrition by job satisfaction (bar chart)
- Attrition by years at company (bar chart)

## Current evaluation
The included model was selected using cross-validation on the training data, then evaluated on a separate 20% test set.
- Selected model: **Random Forest**
- Test accuracy: **79.3%**
- Attrition precision: **39.4%**
- Attrition recall: **55.3%**
- Attrition F1-score: **46.0%**

Accuracy alone is not enough because the dataset has more employees who stayed than employees who left. Discuss precision, recall, and F1-score in your report.

## Important limitations
- This is an educational mini project, not a validated HR decision system.
- The dataset is a historical sample; results may not generalize to other companies or current employees.
- Only five input features are used. Other factors may matter.
- A predicted probability is a model estimate, not a certainty.
- Do not use this model as the sole basis for hiring, firing, promotion, or other employment decisions.


## Deploy publicly on Render
1. Push this project folder to a GitHub repository.
2. On Render, choose **New > Web Service** and connect that repository.
3. Select Python 3.
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`
6. Choose a plan and click **Create Web Service**.
7. Render provides a public `onrender.com` URL after a successful deploy.

The Flask app object is named `app` in `app.py`, so `gunicorn app:app` is the correct start command.
