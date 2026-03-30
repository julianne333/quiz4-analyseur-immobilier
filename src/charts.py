from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk


PALETTE = ["#D8BFAA", "#9C6644", "#355070", "#1F2A30"]


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(
        template="plotly_white",
        margin=dict(l=16, r=16, t=24, b=16),
        height=360,
    )
    return figure


def price_distribution_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("Aucune transaction a afficher.")

    figure = px.histogram(
        df,
        x="price",
        nbins=40,
        template="plotly_white",
        color_discrete_sequence=[PALETTE[1]],
    )
    figure.update_layout(
        title="Distribution des prix",
        xaxis_title="Prix de vente ($)",
        yaxis_title="Transactions",
        margin=dict(l=16, r=16, t=48, b=16),
        height=360,
    )
    return figure


def monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("Aucune tendance disponible.")

    monthly = (
        df.groupby("sale_month")
        .agg(median_price=("price", "median"), transactions=("id", "count"))
        .reset_index()
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=monthly["sale_month"],
            y=monthly["median_price"],
            name="Prix median",
            mode="lines+markers",
            line=dict(color=PALETTE[2], width=3),
        )
    )
    figure.add_trace(
        go.Bar(
            x=monthly["sale_month"],
            y=monthly["transactions"],
            name="Transactions",
            marker_color=PALETTE[0],
            opacity=0.45,
            yaxis="y2",
        )
    )
    figure.update_layout(
        title="Evolution mensuelle",
        template="plotly_white",
        margin=dict(l=16, r=16, t=48, b=16),
        height=360,
        yaxis=dict(title="Prix median ($)"),
        yaxis2=dict(title="Transactions", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
    )
    return figure


def price_vs_living_chart(df: pd.DataFrame, sample_size: int = 2500) -> go.Figure:
    if df.empty:
        return _empty_figure("Pas assez de donnees pour le nuage de points.")

    sample = df.sample(min(len(df), sample_size), random_state=42)
    figure = px.scatter(
        sample,
        x="sqft_living",
        y="price",
        color="grade",
        size="bedrooms",
        hover_data=["zipcode", "bathrooms", "condition", "price_per_sqft"],
        template="plotly_white",
        color_continuous_scale=PALETTE,
    )
    figure.update_layout(
        title="Prix vs surface habitable",
        xaxis_title="Surface habitable (sqft)",
        yaxis_title="Prix ($)",
        margin=dict(l=16, r=16, t=48, b=16),
        height=360,
    )
    return figure


def zipcode_boxplot(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if df.empty:
        return _empty_figure("Pas de comparaison par ZIP.")

    top_zipcodes = df["zipcode"].value_counts().head(top_n).index
    subset = df[df["zipcode"].isin(top_zipcodes)]
    figure = px.box(
        subset,
        x="zipcode",
        y="price",
        color="zipcode",
        template="plotly_white",
        color_discrete_sequence=PALETTE,
    )
    figure.update_layout(
        title="Dispersion des prix sur les ZIP les plus actifs",
        xaxis_title="ZIP code",
        yaxis_title="Prix ($)",
        showlegend=False,
        margin=dict(l=16, r=16, t=48, b=16),
        height=360,
    )
    return figure


def transaction_map(df: pd.DataFrame, max_points: int = 2000) -> pdk.Deck:
    sample = df.sample(min(len(df), max_points), random_state=42).copy()
    sample["radius"] = np.clip(sample["price"] / 4500, 90, 350)
    sample["fill_color"] = sample["waterfront"].apply(
        lambda value: [53, 80, 112, 200] if value == 1 else [156, 102, 68, 140]
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=sample,
        get_position="[long, lat]",
        get_radius="radius",
        get_fill_color="fill_color",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=float(sample["lat"].median()),
        longitude=float(sample["long"].median()),
        zoom=9.1,
        pitch=35,
    )

    return pdk.Deck(
        map_style="light",
        initial_view_state=view_state,
        layers=[layer],
        tooltip={
            "html": "<b>ZIP:</b> {zipcode}<br/><b>Prix:</b> ${price}<br/><b>Surface:</b> {sqft_living} sqft",
            "style": {"backgroundColor": "#21313C", "color": "white"},
        },
    )


def comparables_scatter(
    comparables: pd.DataFrame,
    *,
    subject_sqft: float,
    estimate_price: float,
    actual_price: float | None = None,
) -> go.Figure:
    if comparables.empty:
        return _empty_figure("Aucun comparable n'a pu etre trouve.")

    figure = px.scatter(
        comparables,
        x="sqft_living",
        y="price",
        color="similarity_score",
        hover_data=["zipcode", "bedrooms", "bathrooms", "grade"],
        template="plotly_white",
        color_continuous_scale=[PALETTE[2], PALETTE[1], PALETTE[0]],
    )
    figure.add_trace(
        go.Scatter(
            x=[subject_sqft],
            y=[estimate_price],
            mode="markers",
            marker=dict(color=PALETTE[3], size=16, symbol="diamond"),
            name="Estimation",
        )
    )

    if actual_price is not None and np.isfinite(actual_price):
        figure.add_trace(
            go.Scatter(
                x=[subject_sqft],
                y=[actual_price],
                mode="markers",
                marker=dict(color=PALETTE[1], size=14, symbol="x"),
                name="Prix reel",
            )
        )

    figure.update_layout(
        title="Comparables retenus",
        xaxis_title="Surface habitable (sqft)",
        yaxis_title="Prix ($)",
        margin=dict(l=16, r=16, t=48, b=16),
        height=380,
    )
    return figure


def portfolio_allocation_chart(portfolio: pd.DataFrame) -> go.Figure:
    if portfolio.empty:
        return _empty_figure("Aucune allocation a afficher.")

    allocation = (
        portfolio.groupby("zipcode")
        .agg(capital=("price", "sum"), assets=("id", "count"))
        .reset_index()
        .sort_values("capital", ascending=False)
    )

    figure = px.bar(
        allocation,
        x="zipcode",
        y="capital",
        color="assets",
        template="plotly_white",
        color_continuous_scale=[PALETTE[0], PALETTE[2]],
    )
    figure.update_layout(
        title="Allocation du portefeuille par ZIP",
        xaxis_title="ZIP code",
        yaxis_title="Capital deploye ($)",
        margin=dict(l=16, r=16, t=48, b=16),
        height=360,
    )
    return figure


def portfolio_opportunity_chart(portfolio: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if portfolio.empty:
        return _empty_figure("Aucun actif selectionne.")

    subset = portfolio.sort_values("pricing_gap_positive", ascending=False).head(top_n).copy()
    subset["asset_label"] = subset["id"] + " | " + subset["zipcode"]

    figure = px.bar(
        subset,
        x="asset_label",
        y="pricing_gap_positive",
        color="investment_score",
        template="plotly_white",
        color_continuous_scale=[PALETTE[0], PALETTE[1], PALETTE[2]],
    )
    figure.update_layout(
        title="Upside implicite des meilleurs picks",
        xaxis_title="Actif",
        yaxis_title="Upside implicite ($)",
        margin=dict(l=16, r=16, t=48, b=16),
        height=360,
    )
    return figure
