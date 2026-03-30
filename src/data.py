from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).resolve().parents[1] / "kc_house_data.csv"

NUMERIC_COLUMNS = [
    "price",
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
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]


@st.cache_data(show_spinner=False)
def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["id"] = df["id"].astype(str)
    df["zipcode"] = df["zipcode"].astype("Int64").astype(str)

    sale_year = df["date"].dt.year.fillna(pd.Timestamp.today().year)
    last_update_year = np.where(df["yr_renovated"].fillna(0) > 0, df["yr_renovated"], df["yr_built"])

    df["sale_year"] = sale_year.astype(int)
    df["sale_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["price_per_sqft"] = df["price"] / df["sqft_living"].replace(0, np.nan)
    df["age"] = (sale_year - df["yr_built"]).clip(lower=0)
    df["is_renovated"] = df["yr_renovated"].fillna(0).gt(0)
    df["has_basement"] = df["sqft_basement"].fillna(0).gt(0)

    # Backward-compatible aliases while the rest of the app migrates to the step-0 naming.
    df["property_age"] = df["age"]
    df["renovated"] = df["is_renovated"]

    df["last_update_year"] = last_update_year.astype(int)
    df["basement_share"] = (
        df["sqft_basement"] / df["sqft_living"].replace(0, np.nan)
    ).fillna(0)
    df["transaction_label"] = (
        df["id"]
        + " | ZIP "
        + df["zipcode"]
        + " | $"
        + df["price"].round(0).map(lambda value: f"{value:,.0f}")
    )

    return df.sort_values("date").reset_index(drop=True)


def apply_market_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    start_date = pd.to_datetime(filters["date_range"][0])
    end_date = pd.to_datetime(filters["date_range"][1])
    mask &= df["date"].between(start_date, end_date)

    price_min, price_max = filters["price_range"]
    mask &= df["price"].between(price_min, price_max)

    living_min, living_max = filters["living_range"]
    mask &= df["sqft_living"].between(living_min, living_max)

    bedrooms_min, bedrooms_max = filters["bedrooms_range"]
    mask &= df["bedrooms"].between(bedrooms_min, bedrooms_max)

    bathrooms_min, bathrooms_max = filters["bathrooms_range"]
    mask &= df["bathrooms"].between(bathrooms_min, bathrooms_max)

    grade_min, grade_max = filters["grade_range"]
    mask &= df["grade"].between(grade_min, grade_max)

    condition_min, condition_max = filters["condition_range"]
    mask &= df["condition"].between(condition_min, condition_max)

    if filters["zipcodes"]:
        mask &= df["zipcode"].isin(filters["zipcodes"])

    if filters["waterfront"] == "Oui":
        mask &= df["waterfront"].eq(1)
    elif filters["waterfront"] == "Non":
        mask &= df["waterfront"].eq(0)

    if filters["renovated"] == "Oui":
        mask &= df["is_renovated"]
    elif filters["renovated"] == "Non":
        mask &= ~df["is_renovated"]

    return df.loc[mask].copy()


def summarize_market(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "transactions": 0,
            "median_price": 0.0,
            "average_price": 0.0,
            "median_price_per_sqft": 0.0,
            "median_living": 0.0,
        }

    return {
        "transactions": float(len(df)),
        "median_price": float(df["price"].median()),
        "average_price": float(df["price"].mean()),
        "median_price_per_sqft": float(df["price_per_sqft"].median()),
        "median_living": float(df["sqft_living"].median()),
    }


def compute_market_insights(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["Aucune transaction ne correspond aux filtres selectionnes."]

    insights: list[str] = []

    zip_summary = (
        df.groupby("zipcode")
        .agg(median_price=("price", "median"), transactions=("id", "count"))
        .sort_values(["median_price", "transactions"], ascending=[False, False])
    )
    top_zip = zip_summary.index[0]
    insights.append(
        f"Le ZIP {top_zip} affiche la mediane la plus elevee dans l'echantillon filtre, a "
        f"${zip_summary.iloc[0]['median_price']:,.0f}."
    )

    renovated = df.groupby("is_renovated")["price_per_sqft"].median()
    if True in renovated.index and False in renovated.index and renovated.loc[False] > 0:
        premium = (renovated.loc[True] / renovated.loc[False] - 1) * 100
        insights.append(
            f"Les biens renoves se vendent environ {premium:.1f}% plus cher par sqft que les non-renoves."
        )

    grade_summary = df.groupby("grade")["price"].median().sort_values(ascending=False)
    best_grade = int(grade_summary.index[0])
    insights.append(
        f"Le grade {best_grade} domine la distribution recente avec une mediane de "
        f"${grade_summary.iloc[0]:,.0f}."
    )

    return insights


def zipcode_profile(df: pd.DataFrame, zipcode: str) -> pd.Series:
    zip_df = df[df["zipcode"] == str(zipcode)]
    if zip_df.empty:
        return df.median(numeric_only=True)
    return zip_df.median(numeric_only=True)


def build_subject_record(
    *,
    df: pd.DataFrame,
    zipcode: str,
    bedrooms: float,
    bathrooms: float,
    sqft_living: float,
    sqft_lot: float,
    floors: float,
    waterfront: int,
    view: int,
    condition: int,
    grade: int,
    yr_built: int,
    yr_renovated: int,
    lat: float | None,
    long: float | None,
    sqft_basement: float,
    sqft_living15: float | None,
    sqft_lot15: float | None,
) -> pd.DataFrame:
    profile = zipcode_profile(df, zipcode)
    today = pd.Timestamp.today().normalize()

    lat_value = float(profile["lat"]) if lat is None else float(lat)
    long_value = float(profile["long"]) if long is None else float(long)
    living15_value = (
        float(profile["sqft_living15"]) if sqft_living15 is None else float(sqft_living15)
    )
    lot15_value = float(profile["sqft_lot15"]) if sqft_lot15 is None else float(sqft_lot15)

    sqft_above = max(float(sqft_living) - float(sqft_basement), 0.0)
    last_update_year = yr_renovated if yr_renovated > 0 else yr_built

    subject = pd.DataFrame(
        [
            {
                "id": "manual-input",
                "date": today,
                "price": np.nan,
                "bedrooms": float(bedrooms),
                "bathrooms": float(bathrooms),
                "sqft_living": float(sqft_living),
                "sqft_lot": float(sqft_lot),
                "floors": float(floors),
                "waterfront": int(waterfront),
                "view": int(view),
                "condition": int(condition),
                "grade": int(grade),
                "sqft_above": float(sqft_above),
                "sqft_basement": float(sqft_basement),
                "yr_built": int(yr_built),
                "yr_renovated": int(yr_renovated),
                "zipcode": str(zipcode),
                "lat": lat_value,
                "long": long_value,
                "sqft_living15": living15_value,
                "sqft_lot15": lot15_value,
                "sale_year": int(today.year),
                "sale_month": today.to_period("M").to_timestamp(),
                "price_per_sqft": np.nan,
                "age": max(int(today.year) - int(yr_built), 0),
                "is_renovated": bool(yr_renovated > 0),
                "has_basement": bool(float(sqft_basement) > 0),
                "property_age": max(int(today.year) - int(yr_built), 0),
                "renovated": bool(yr_renovated > 0),
                "last_update_year": int(last_update_year),
                "basement_share": (
                    float(sqft_basement) / float(sqft_living) if float(sqft_living) > 0 else 0.0
                ),
                "transaction_label": f"manual-input | ZIP {zipcode}",
            }
        ]
    )

    return subject


def _candidate_masks(df: pd.DataFrame, subject: pd.Series) -> list[pd.Series]:
    zipcode = str(subject["zipcode"])
    sqft_living = max(float(subject["sqft_living"]), 1.0)
    bedrooms = float(subject["bedrooms"])
    bathrooms = float(subject["bathrooms"])
    grade = float(subject["grade"])
    condition = float(subject["condition"])
    lat = float(subject["lat"])
    long = float(subject["long"])

    same_zip = df["zipcode"].eq(zipcode)
    tight_size = df["sqft_living"].between(sqft_living * 0.7, sqft_living * 1.3)
    medium_size = df["sqft_living"].between(sqft_living * 0.55, sqft_living * 1.45)
    wide_size = df["sqft_living"].between(sqft_living * 0.4, sqft_living * 1.6)
    tight_rooms = df["bedrooms"].between(bedrooms - 1, bedrooms + 1) & df["bathrooms"].between(
        bathrooms - 1, bathrooms + 1
    )
    medium_rooms = df["bedrooms"].between(bedrooms - 2, bedrooms + 2) & df[
        "bathrooms"
    ].between(bathrooms - 1.5, bathrooms + 1.5)
    grade_match = df["grade"].between(grade - 1, grade + 1)
    broad_grade = df["grade"].between(grade - 2, grade + 2)
    condition_match = df["condition"].between(condition - 1, condition + 1)
    geo_window = df["lat"].between(lat - 0.03, lat + 0.03) & df["long"].between(
        long - 0.03, long + 0.03
    )

    return [
        same_zip & tight_size & tight_rooms & grade_match & condition_match,
        same_zip & medium_size & medium_rooms & broad_grade,
        geo_window & medium_size & medium_rooms,
        same_zip & wide_size,
        medium_size & medium_rooms & broad_grade,
        wide_size,
    ]


def find_comparables(df: pd.DataFrame, subject: pd.Series, max_results: int = 12) -> pd.DataFrame:
    candidates = df[df["id"] != str(subject["id"])].copy()
    chosen = candidates

    for mask in _candidate_masks(candidates, subject):
        subset = candidates[mask].copy()
        if len(subset) >= 5:
            chosen = subset
            break

    subject_age = max(float(subject["age"]), 0.0)
    chosen["geo_distance"] = np.sqrt(
        (chosen["lat"] - float(subject["lat"])) ** 2 + (chosen["long"] - float(subject["long"])) ** 2
    )
    chosen["size_gap"] = (
        (chosen["sqft_living"] - float(subject["sqft_living"])).abs()
        / max(float(subject["sqft_living"]), 1.0)
    )
    chosen["age_gap"] = (
        (chosen["age"] - subject_age).abs() / max(subject_age, 1.0)
    )
    chosen["similarity_score"] = (
        chosen["size_gap"] * 0.35
        + (chosen["bedrooms"] - float(subject["bedrooms"])).abs() * 0.12
        + (chosen["bathrooms"] - float(subject["bathrooms"])).abs() * 0.12
        + (chosen["grade"] - float(subject["grade"])).abs() * 0.15
        + (chosen["condition"] - float(subject["condition"])).abs() * 0.08
        + chosen["geo_distance"] * 6.0
        + chosen["age_gap"] * 0.08
    )

    return chosen.sort_values(["similarity_score", "date"], ascending=[True, False]).head(max_results)


def blend_price_estimate(comparables: pd.DataFrame, model_prediction: float) -> dict[str, float]:
    sources: list[float] = []

    if not comparables.empty:
        comp_median = float(comparables["price"].median())
        comp_q1 = float(comparables["price"].quantile(0.25))
        comp_q3 = float(comparables["price"].quantile(0.75))
        sources.append(comp_median)
    else:
        comp_median = float("nan")
        comp_q1 = float("nan")
        comp_q3 = float("nan")

    if np.isfinite(model_prediction):
        sources.append(float(model_prediction))

    central = float(np.mean(sources)) if sources else float("nan")

    if np.isfinite(comp_q1) and np.isfinite(comp_q3):
        band = max((comp_q3 - comp_q1) / 2, central * 0.08)
    else:
        band = central * 0.12

    return {
        "estimate": central,
        "low": max(central - band, 0.0),
        "high": central + band,
        "comp_median": comp_median,
        "comp_q1": comp_q1,
        "comp_q3": comp_q3,
    }
