# ✈️ Airline Passenger Satisfaction Predictor

A machine learning project that predicts whether an airline passenger will be
**satisfied** or **neutral/dissatisfied** with their flight, based on trip
details and in-flight service ratings.

Built end-to-end: data cleaning → outlier removal → training and comparing
5 classification models → picking the best one based on accuracy, overfitting
checks, and ROC-AUC → deploying it behind a Flask web app with a custom UI.

## Highlights
- Trained and compared **5 models** (Logistic Regression, Decision Tree,
  Random Forest, Gradient Boosting, XGBoost) — selected based on test
  accuracy *and* train/test gap, to avoid picking an overfit model
- **XGBoost** selected as the final model — ~95.7% test accuracy, ~0.99 ROC-AUC
- Outlier removal via the IQR method, missing-value handling, categorical encoding
- Full evaluation: confusion matrix, precision/recall/F1, ROC curve, feature importance
- Flask web app with a simplified prediction form (shows only the highest-impact
  features — Customer Type, Type of Travel, Class, Online boarding, Inflight wifi,
  Seat comfort, Inflight entertainment, Ease of Online booking — the rest are
  filled in automatically with neutral defaults)
- Custom "boarding pass" themed frontend (HTML/CSS/JS, no framework)

## Tech stack
Python, scikit-learn, XGBoost, pandas, Flask, HTML/CSS/JS

## Project structure
project/
├── model_training/ # Model training notebook (Google Colab)
├── flask_app/ # Flask backend + frontend + trained model
│ ├── app.py
│ ├── requirements.txt
│ ├── model/ # model.pkl + preprocessor.pkl
│ ├── static/ # CSS + JS
│ └── templates/ # HTML
└── README.md

## How to run the app
```bash
cd project/flask_app
pip install -r requirements.txt
python app.py
```
Then open **http://127.0.0.1:5000**

## Model training
The notebook inside `model_training/` was run in Google Colab. It covers:
- loading and cleaning the dataset
- removing outliers with the IQR method
- encoding categorical features
- training and comparing 5 models with train/test/cross-validation accuracy
- evaluating the best model (confusion matrix, ROC curve, feature importance)
- exporting `model.pkl` and `preprocessor.pkl` for the Flask app

## Why XGBoost was chosen
Among the 5 models compared, XGBoost had the highest test accuracy and
ROC-AUC while keeping a small gap between train and test accuracy — meaning
it generalizes well instead of overfitting to the training data. Logistic
Regression underperformed since it can only draw a linear decision boundary,
while ensemble methods (Random Forest, Gradient Boosting, XGBoost) captured
the non-linear relationships between service ratings and satisfaction more
effectively.