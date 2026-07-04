# House Price Prediction - FAML Assignment 9

End-to-end house price prediction system using machine learning, with a FastAPI backend and Streamlit frontend.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Approach](#approach)
- [Models Compared](#models-compared)
- [Running the Application](#running-the-application)
- [Screenshots](#screenshots)
- [Requirements](#requirements)

## Project Structure

```
├── Housing.csv              # Dataset (545 records, 13 features)
├── model_try.ipynb          # EDA, feature engineering & model training
├── house_price_model.pkl    # Trained RandomForestRegressor
├── feature_columns.pkl      # Feature columns used in training
├── main.py                  # FastAPI backend API
├── app_fixed.py             # Streamlit frontend UI
└── objects/                 # Application screenshots
```

## Dataset

**Housing.csv** contains 545 records with the following features:

- **Numerical:** `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
- **Categorical:** `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`
- **Target:** `price`

## Approach

1. **EDA** — Explored data distribution, identified numerical vs categorical columns, checked for missing values
2. **Feature Engineering** — Binary encoding (yes/no → 1/0), one-hot encoding (`furnishingstatus`), created `price_per_sqft` feature
3. **Model Training** — Compared Linear Regression vs Random Forest Regressor
4. **Model Selection** — Random Forest performed better (MAE: ~287k vs ~504k)

## Models Compared

| Model | MAE | MSE |
|---|---|---|
| Linear Regression | 504,031.78 | 587,217,091,347.07 |
| Random Forest | 287,866.14 | 327,229,666,131.63 |

Random Forest was selected and saved as `house_price_model.pkl`.

## Running the Application

### 1. Start the FastAPI backend

```bash
uvicorn main:app --reload
```

API runs at `http://127.0.0.1:8000`. Endpoints:
- `GET /` — Health check
- `GET /data` — View dataset
- `POST /predict` — Predict house price (send JSON with house features)

### 2. Start the Streamlit frontend

```bash
streamlit run app_fixed.py
```

Opens a web UI to input house details and get predicted prices.

## Screenshots

### API Response

![API Response](objects/Screenshot%202026-07-04%20at%2011.01.57%E2%80%AFAM.png)

### Streamlit UI

![Streamlit UI](objects/Screenshot%202026-07-04%20at%2011.06.30%E2%80%AFAM.png)

## Requirements

- Python 3.10+
- fastapi, uvicorn, pydantic
- pandas, numpy, scikit-learn
- streamlit, requests
