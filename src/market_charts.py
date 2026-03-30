from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import pandas as pd


def _empty_figure(message: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


def price_histogram(df: pd.DataFrame):
    if df.empty:
        return _empty_figure("Aucune transaction a afficher.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["price"], bins=35, color="#9C6644", edgecolor="white", alpha=0.9)
    ax.set_title("Distribution des prix")
    ax.set_xlabel("Prix de vente ($)")
    ax.set_ylabel("Nombre de proprietes")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def price_vs_sqft_scatter(df: pd.DataFrame):
    if df.empty:
        return _empty_figure("Aucune transaction a afficher.")

    sample = df.sample(min(len(df), 3000), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scatter = ax.scatter(
        sample["sqft_living"],
        sample["price"],
        c=sample["grade"],
        cmap="copper",
        alpha=0.72,
        s=26,
        edgecolors="none",
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Grade")
    ax.set_title("Prix vs superficie habitable")
    ax.set_xlabel("Superficie habitable (sqft_living)")
    ax.set_ylabel("Prix de vente ($)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame):
    numeric_columns = [
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
        "sqft_basement",
        "yr_built",
        "price_per_sqft",
        "age",
    ]
    available_columns = [column for column in numeric_columns if column in df.columns]
    if df.empty or len(available_columns) < 2:
        return _empty_figure("Pas assez de variables pour la heatmap.")

    corr = df[available_columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    image = ax.imshow(corr, cmap="BrBG", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Matrice de correlation des variables numeriques")

    for row in range(len(corr.index)):
        for col in range(len(corr.columns)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def average_price_by_bedrooms(df: pd.DataFrame):
    if df.empty:
        return _empty_figure("Aucune transaction a afficher.")

    grouped = (
        df.groupby("bedrooms")
        .agg(avg_price=("price", "mean"), properties=("id", "count"))
        .reset_index()
        .sort_values("bedrooms")
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(grouped["bedrooms"].astype(str), grouped["avg_price"], color="#355070", alpha=0.9)
    ax.set_title("Prix moyen par nombre de chambres")
    ax.set_xlabel("Nombre de chambres")
    ax.set_ylabel("Prix moyen ($)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig
