# Airline Passenger Satisfaction — Full Project

Predicts whether a passenger will be **satisfied** or **neutral/dissatisfied**
with their flight, based on a Kaggle-style airline survey dataset. The project
has two parts:

```
project/
├── model_training/
│   ├── Airline_Passenger_Satisfaction_Training.ipynb   ← run this in Google Colab
│   ├── train_pipeline.py                               ← same pipeline as a plain script
│   ├── train.csv                                        ← dataset (copy your own if updating)
│   └── model_comparison_results.csv                     ← metrics for every model tried
└── flask_app/
    ├── app.py
    ├── requirements.txt
    ├── model/
    │   ├── model.pkl            ← trained XGBoost classifier (already included)
    │   └── preprocessor.pkl     ← encoders + scaler + feature order (already included)
    ├── templates/index.html
    └── static/css/style.css, static/js/script.js
```

The Flask app already ships with a trained `model.pkl` / `preprocessor.pkl`, so
**you can run the web app immediately without retraining anything.** Retrain
only if you want to use a different/updated dataset.

---

## 1. Run the Flask app (no training needed)

```bash
cd flask_app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Fill in the passenger/flight
details and service ratings, click **RUN FORECAST**, and the result appears
as a boarding-pass style stub with the predicted outcome and confidence.

> If you see a `InconsistentVersionWarning` in the terminal, it's harmless —
> it just means your local scikit-learn/xgboost version differs slightly
> from the one used to train the bundled model. Predictions still work. If
> you want it to go away, retrain with step 2 below using your installed
> library versions.

---

## 2. (Optional) Retrain the model in Google Colab

1. Open [Google Colab](https://colab.research.google.com) and upload
   `model_training/Airline_Passenger_Satisfaction_Training.ipynb`.
2. Upload `train.csv` when the notebook asks for it (or mount Google Drive
   and point the `pd.read_csv(...)` call at your copy).
3. Run all cells top to bottom. The notebook:
   - drops the id/index columns
   - fills the small number of missing `Arrival Delay` values with the median
   - removes outliers from `Flight Distance` and the two delay columns using
     the IQR method (with before/after boxplots)
   - label-encodes the categorical columns
   - trains **5 models** (Logistic Regression, Decision Tree, Random Forest,
     Gradient Boosting, XGBoost) and compares train/test/cross-validation
     accuracy to catch over- and under-fitting
   - picks the best model (highest test accuracy among models with a
     train/test gap under 3%)
   - prints a full evaluation: classification report, confusion matrix,
     ROC curve, feature importances
   - saves `model.pkl` and `preprocessor.pkl`
4. Download both files from Colab's file panel (or use the commented-out
   `files.download(...)` cells at the end of the notebook) and drop them
   into `flask_app/model/`, replacing the existing ones.
5. Restart the Flask app.

`train_pipeline.py` in the same folder is the identical pipeline as a plain
`.py` script, in case you'd rather run it locally instead of in Colab.

---

## Why XGBoost was selected

Five models were trained and compared on the same 80/20 split plus 5-fold
cross-validation (see `model_comparison_results.csv` for the exact numbers
from this run):

| Model | Test accuracy | Train − test gap | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~86.8% | ~0.3% | ~0.92 |
| Decision Tree (depth-limited) | ~93.3% | ~0.4% | ~0.98 |
| Gradient Boosting | ~94.7% | ~0.3% | ~0.99 |
| Random Forest | ~95.0% | ~1.1% | ~0.99 |
| **XGBoost** | **~95.7%** | **~1.4%** | **~0.99** |

- **Logistic Regression underfits** — it can only draw a straight decision
  boundary, so it can't capture the non-linear way service ratings interact,
  leaving several points of accuracy on the table.
- **A single Decision Tree** does better but still underperforms the
  ensembles, since one tree can't average away noise the way many trees can.
- **Random Forest, Gradient Boosting, and XGBoost** all perform strongly with
  small train/test gaps (no overfitting), because they combine many weak
  learners rather than relying on one model to memorize the data.
- **XGBoost** came out on top on test accuracy and ROC-AUC while keeping the
  gap small, thanks to its built-in regularization (`subsample`,
  `colsample_bytree`, shrinkage via `learning_rate`), so it was selected as
  the final model.

Re-running the notebook on a different split or updated data may shift these
numbers slightly — the notebook always re-selects whichever model wins under
the same rule (best test accuracy among models with under a 3% overfit gap),
so the "best model" logic stays honest even if the exact winner changes.
