from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STRATEGY_PROFILES: dict[str, dict[str, Any]] = {
    "Core": {
        "description": "Priorise la qualite, la liquidite et la stabilite locale.",
        "weights": {
            "discount_score": 0.15,
            "quality_score": 0.35,
            "stability_score": 0.40,
            "value_add_score": 0.10,
        },
    },
    "Core+": {
        "description": "Cherche un peu plus de rendement tout en gardant un profil defensif.",
        "weights": {
            "discount_score": 0.25,
            "quality_score": 0.30,
            "stability_score": 0.25,
            "value_add_score": 0.20,
        },
    },
    "Value-add": {
        "description": "Met l'accent sur le rerating potentiel, la renovation et la sous-valorisation.",
        "weights": {
            "discount_score": 0.35,
            "quality_score": 0.10,
            "stability_score": 0.15,
            "value_add_score": 0.40,
        },
    },
    "Balanced": {
        "description": "Equilibre upside, qualite et diversification.",
        "weights": {
            "discount_score": 0.30,
            "quality_score": 0.25,
            "stability_score": 0.25,
            "value_add_score": 0.20,
        },
    },
}


def _percentile_score(series: pd.Series) -> pd.Series:
    clean = series.fillna(series.median())
    return clean.rank(pct=True, method="average") * 100


def score_investment_candidates(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    scored = df.copy()
    zip_stats = (
        scored.groupby("zipcode")
        .agg(
            zip_transactions=("id", "count"),
            zip_median_price=("price", "median"),
            zip_median_ppsf=("price_per_sqft", "median"),
        )
        .reset_index()
    )
    scored = scored.merge(zip_stats, on="zipcode", how="left")

    quality_raw = (
        scored["grade"] / 13 * 0.45
        + scored["condition"] / 5 * 0.20
        + scored["view"] / 4 * 0.10
        + scored["waterfront"] * 0.10
        + scored["is_renovated"].astype(int) * 0.15
    )
    stability_raw = (
        _percentile_score(scored["zip_transactions"]) * 0.55
        + _percentile_score(scored["zip_median_ppsf"]) * 0.25
        + _percentile_score(scored["sqft_living15"]) * 0.20
    )
    value_add_raw = (
        (scored["age"].clip(0, 120) / 120) * 0.45
        + (1 - scored["condition"].clip(1, 5) / 5) * 0.20
        + (~scored["is_renovated"]).astype(int) * 0.25
        + (1 - scored["grade"].clip(1, 13) / 13) * 0.10
    )

    scored["discount_score"] = _percentile_score(scored["pricing_gap_pct"].clip(-0.25, 0.35))
    scored["quality_score"] = _percentile_score(quality_raw)
    scored["stability_score"] = _percentile_score(stability_raw)
    scored["value_add_score"] = _percentile_score(value_add_raw)

    weights = STRATEGY_PROFILES[strategy]["weights"]
    scored["investment_score"] = (
        scored["discount_score"] * weights["discount_score"]
        + scored["quality_score"] * weights["quality_score"]
        + scored["stability_score"] * weights["stability_score"]
        + scored["value_add_score"] * weights["value_add_score"]
    )

    scored["pricing_gap_positive"] = scored["pricing_gap"].clip(lower=0)
    scored["pricing_gap_pct_positive"] = scored["pricing_gap_pct"].clip(lower=0)
    scored["quality_bucket"] = pd.cut(
        scored["investment_score"],
        bins=[0, 55, 70, 85, 100],
        labels=["Watchlist", "Interesting", "Priority", "High conviction"],
        include_lowest=True,
    )

    return scored.sort_values(
        ["investment_score", "pricing_gap_pct_positive", "price"],
        ascending=[False, False, True],
    )


def build_portfolio(
    candidates: pd.DataFrame,
    *,
    total_budget: float,
    max_assets: int,
    max_assets_per_zip: int,
    min_score: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    selected_rows: list[dict[str, Any]] = []
    deployed = 0.0
    zip_counts: dict[str, int] = {}

    ranked = candidates[candidates["investment_score"] >= min_score].sort_values(
        ["investment_score", "pricing_gap_positive"],
        ascending=[False, False],
    )

    for _, row in ranked.iterrows():
        if len(selected_rows) >= max_assets:
            break

        zipcode = str(row["zipcode"])
        if zip_counts.get(zipcode, 0) >= max_assets_per_zip:
            continue

        if deployed + float(row["price"]) > total_budget:
            continue

        selected_rows.append(row.to_dict())
        deployed += float(row["price"])
        zip_counts[zipcode] = zip_counts.get(zipcode, 0) + 1

    if not selected_rows:
        return ranked.head(0).copy()

    portfolio = pd.DataFrame(selected_rows)
    portfolio["portfolio_weight"] = portfolio["price"] / portfolio["price"].sum()
    return portfolio


def summarize_portfolio(portfolio: pd.DataFrame, budget: float) -> dict[str, float]:
    if portfolio.empty:
        return {
            "assets": 0.0,
            "capital": 0.0,
            "budget_remaining": budget,
            "avg_score": 0.0,
            "upside_dollars": 0.0,
            "upside_pct": 0.0,
            "zipcodes": 0.0,
        }

    capital = float(portfolio["price"].sum())
    upside_dollars = float(portfolio["pricing_gap_positive"].sum())

    return {
        "assets": float(len(portfolio)),
        "capital": capital,
        "budget_remaining": max(float(budget - capital), 0.0),
        "avg_score": float(portfolio["investment_score"].mean()),
        "upside_dollars": upside_dollars,
        "upside_pct": float(upside_dollars / capital) if capital > 0 else 0.0,
        "zipcodes": float(portfolio["zipcode"].nunique()),
    }


def portfolio_insights(portfolio: pd.DataFrame, strategy: str, budget: float) -> list[str]:
    if portfolio.empty:
        return [
            "Aucun portefeuille n'a pu etre construit avec les contraintes actuelles. "
            "Augmentez le budget ou assouplissez les filtres."
        ]

    insights: list[str] = []
    summary = summarize_portfolio(portfolio, budget)
    top_zip = (
        portfolio.groupby("zipcode")["price"].sum().sort_values(ascending=False).index[0]
    )
    top_asset = portfolio.sort_values("investment_score", ascending=False).iloc[0]

    insights.append(
        f"Le portefeuille {strategy} deploie ${summary['capital']:,.0f} sur "
        f"{int(summary['assets'])} actifs et couvre {int(summary['zipcodes'])} ZIP codes."
    )
    insights.append(
        f"La plus forte exposition est concentree sur le ZIP {top_zip}, avec "
        f"${portfolio.groupby('zipcode')['price'].sum().sort_values(ascending=False).iloc[0]:,.0f} engages."
    )
    insights.append(
        f"L'opportunite la plus convaincante est l'actif {top_asset['id']} avec un score de "
        f"{top_asset['investment_score']:.1f} et un upside implicite de ${max(top_asset['pricing_gap'], 0):,.0f}."
    )

    return insights
