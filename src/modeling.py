from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import DATA_PATH, load_dataset


MODEL_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "sale_year",
    "age",
    "is_renovated",
]


@st.cache_resource(show_spinner=False)
def train_price_model(path: str | Path = DATA_PATH) -> dict[str, object]:
    df = load_dataset(path)
    X = df[MODEL_FEATURES].copy()
    X["zipcode"] = X["zipcode"].astype(int)
    X["is_renovated"] = X["is_renovated"].astype(int)
    y = df["price"]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    random_state=42,
                    max_depth=8,
                    learning_rate=0.05,
                    max_iter=300,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = np.clip(model.predict(X_test), 0, None)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mape": float(mean_absolute_percentage_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    model.fit(X, y)

    return {
        "model": model,
        "metrics": metrics,
        "features": MODEL_FEATURES,
    }


def predict_price(model_bundle: dict[str, object], subject_frame: pd.DataFrame) -> float:
    X = subject_frame[MODEL_FEATURES].copy()
    X["zipcode"] = X["zipcode"].astype(int)
    X["is_renovated"] = X["is_renovated"].astype(int)
    model = model_bundle["model"]
    prediction = model.predict(X)[0]
    return float(max(prediction, 0.0))


@st.cache_data(show_spinner=False)
def score_transaction_universe(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = load_dataset(path).copy()
    bundle = train_price_model(path)

    X = df[MODEL_FEATURES].copy()
    X["zipcode"] = X["zipcode"].astype(int)
    X["is_renovated"] = X["is_renovated"].astype(int)

    predictions = np.clip(bundle["model"].predict(X), 0, None)
    df["predicted_price"] = predictions
    df["pricing_gap"] = df["predicted_price"] - df["price"]
    df["pricing_gap_pct"] = (
        df["pricing_gap"] / df["price"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    df["pricing_gap_pct"] = df["pricing_gap_pct"].fillna(0.0)

    return df
