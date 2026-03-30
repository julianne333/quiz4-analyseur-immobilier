from __future__ import annotations

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import pandas as pd


def _empty_figure(message: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


def comparable_bar_chart(subject: pd.Series, comparables: pd.DataFrame):
    if comparables.empty:
        return _empty_figure("Aucun comparable disponible pour le graphique.")

    chart_df = comparables.copy().head(8)
    chart_df = chart_df.assign(
        label=chart_df["id"].astype(str),
        kind="Comparable",
    )

    subject_row = pd.DataFrame(
        [
            {
                "id": subject["id"],
                "label": f"{subject['id']} (selection)",
                "price": subject["price"],
                "price_per_sqft": subject["price_per_sqft"],
                "kind": "Selection",
            }
        ]
    )

    combined = pd.concat(
        [subject_row[["label", "price", "price_per_sqft", "kind"]], chart_df[["label", "price", "price_per_sqft", "kind"]]],
        ignore_index=True,
    )

    colors = ["#9C6644" if value == "Selection" else "#355070" for value in combined["kind"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(combined["label"], combined["price"], color=colors, alpha=0.9)
    axes[0].set_title("Comparaison des prix")
    axes[0].set_xlabel("Proprietes")
    axes[0].set_ylabel("Prix ($)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(combined["label"], combined["price_per_sqft"], color=colors, alpha=0.9)
    axes[1].set_title("Comparaison du prix par pi2")
    axes[1].set_xlabel("Proprietes")
    axes[1].set_ylabel("Prix / pi2 ($)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Propriete selectionnee vs comparables", fontsize=13)
    fig.tight_layout()
    return fig
