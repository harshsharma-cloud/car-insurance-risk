# Car Insurance Claim Prediction

A machine learning system that predicts the probability of a car insurance claim based on customer profile data. Built as an end-to-end project from raw data through to a deployed Streamlit application.



## What this does

Given a customer profile (age, driving experience, vehicle type, credit score, etc.), the model outputs a claim probability and classifies the customer as **LOW**, **MEDIUM**, or **HIGH** risk. The system was validated on a held-out test set of 2,000 records and achieves:

- **ROC-AUC:** 0.8858
- **F1 Score:** 0.7257
- **Accuracy:** 82.95%

A Logistic Regression model was selected after comparing it against Random Forest across cross-validated ROC-AUC, F1, and calibration metrics.



## Applications

1. Prediction Interface** (`app/streamlit_app.py`) — enter a customer profile and get an instant risk verdict.

2. Stakeholder Dashboard** (`app/dashboard.py`) — analytics view covering claim patterns by segment, feature importance, model performance curves, error analysis, and evidence-backed focus areas.
[https://car-insurance-risk-rkjd2pxnhhsxjngdy55nwj.streamlit.app/]


## Running locally

```bash
# Clone the repo and install dependencies
pip install -r requirements.txt

# Prediction app
streamlit run app/streamlit_app.py

# Stakeholder dashboard (separate port)
streamlit run app/dashboard.py --server.port 8502

# Run the test suite
python -m pytest tests/ -v
```



## Project structure

```
├── app/
│   ├── streamlit_app.py    # Customer risk prediction interface
│   └── dashboard.py        # Stakeholder analytics dashboard
├── src/
│   ├── data.py             # Data loading and schema validation
│   ├── preprocessing.py    # sklearn ColumnTransformer builder
│   ├── train.py            # Train/test split and model persistence
│   ├── evaluate.py         # Metrics, calibration, and error analysis
│   ├── interpret.py        # Feature importance and coefficient extraction
│   ├── predict.py          # Inference layer (loads pipeline, validates input)
│   └── risk.py             # Risk tier classification and focus area detection
├── models/
│   └── final_pipeline.joblib   # Trained sklearn Pipeline (required for deployment)
├── data/raw/
│   └── car_insurance.csv       # Source dataset (10,000 records)
├── notebooks/              # EDA and modeling scripts (01 through 07)
├── tests/                  # Pytest unit tests (51 tests)
└── requirements.txt
```

---

## Dataset

Public educational dataset sourced from DataCamp. 10,000 records, 15 predictive features, binary target (claim / no claim). Overall claim rate: 31.3%.

*Not real customer data.*
