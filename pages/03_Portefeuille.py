from __future__ import annotations

import streamlit as st

from src.ai_narration import ai_is_configured, generate_investment_narrative
from src.charts import portfolio_allocation_chart, portfolio_opportunity_chart, transaction_map
from src.data import apply_market_filters, summarize_market
from src.modeling import score_transaction_universe
from src.portfolio import (
    STRATEGY_PROFILES,
    build_portfolio,
    portfolio_insights,
    score_investment_candidates,
    summarize_portfolio,
)
from src.ui import configure_page, inject_app_css, render_ai_note, render_hero, render_market_filters


configure_page("03 | Portefeuille")
inject_app_css()

full_df = score_transaction_universe()
filters = render_market_filters(full_df, key_prefix="portfolio")
filtered_df = apply_market_filters(full_df, filters)

render_hero(
    "Portfolio Construction",
    "Construire une poche cible",
    "Selectionnez un univers d'acquisition, choisissez une strategie, allouez un budget et "
    "laissez l'app prioriser les dossiers les plus convaincants.",
)

if filtered_df.empty:
    st.warning("Le filtre actuel ne laisse aucun candidat dans l'univers investi.")
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    strategy = st.selectbox("Strategie", options=list(STRATEGY_PROFILES.keys()))
    st.caption(STRATEGY_PROFILES[strategy]["description"])
with right:
    total_budget = st.number_input(
        "Budget d'acquisition ($)",
        min_value=500000,
        max_value=25000000,
        value=5000000,
        step=250000,
    )

control_1, control_2, control_3 = st.columns(3)
with control_1:
    max_assets = st.slider("Nombre max d'actifs", min_value=2, max_value=20, value=8)
with control_2:
    max_assets_per_zip = st.slider("Max d'actifs par ZIP", min_value=1, max_value=5, value=2)
with control_3:
    min_score = st.slider("Score minimum", min_value=40, max_value=90, value=65)

scored_candidates = score_investment_candidates(filtered_df, strategy)
portfolio = build_portfolio(
    scored_candidates,
    total_budget=total_budget,
    max_assets=max_assets,
    max_assets_per_zip=max_assets_per_zip,
    min_score=min_score,
)
portfolio_summary = summarize_portfolio(portfolio, total_budget)
market_summary = summarize_market(filtered_df)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Actifs retenus", f"{int(portfolio_summary['assets'])}")
metric_2.metric("Capital deploye", f"${portfolio_summary['capital']:,.0f}")
metric_3.metric("Upside implicite", f"${portfolio_summary['upside_dollars']:,.0f}", f"{portfolio_summary['upside_pct'] * 100:+.1f}%")
metric_4.metric("Score moyen", f"{portfolio_summary['avg_score']:.1f}")

if portfolio.empty:
    st.warning(
        "Aucun portefeuille n'a pu etre construit avec les contraintes actuelles. "
        "Essayez un budget plus eleve, un score minimum plus bas ou un univers plus large."
    )
    st.stop()

left, right = st.columns(2, gap="large")
with left:
    st.plotly_chart(portfolio_allocation_chart(portfolio), use_container_width=True)
with right:
    st.plotly_chart(portfolio_opportunity_chart(portfolio), use_container_width=True)

st.markdown('<div class="section-label">Carte des picks</div>', unsafe_allow_html=True)
st.pydeck_chart(transaction_map(portfolio, max_points=min(len(portfolio), filters["map_points"])), use_container_width=True)

st.markdown('<div class="section-label">Lectures portefeuille</div>', unsafe_allow_html=True)
portfolio_notes = portfolio_insights(portfolio, strategy, total_budget)
for note in portfolio_notes:
    st.markdown(f'<div class="insight-box">{note}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Narration IA</div>', unsafe_allow_html=True)
top_zip_exposure = (
    portfolio.groupby("zipcode")["price"].sum().sort_values(ascending=False).head(5).to_dict()
)
fallback_portfolio_note = (
    f"Le portefeuille propose {int(portfolio_summary['assets'])} acquisitions pour un capital deploye de "
    f"${portfolio_summary['capital']:,.0f} sur un budget de ${total_budget:,.0f}. "
    f"La strategie {strategy} ressort avec un score moyen de {portfolio_summary['avg_score']:.1f} et un upside implicite "
    f"agrege de ${portfolio_summary['upside_dollars']:,.0f}. L'opportunite principale est la combinaison d'un filtre marche "
    "cible et d'une discipline de diversification par ZIP. Le risque principal reste la nature retrospective du dataset "
    "et le fait que l'upside provient d'un modele de prix, pas d'un cash-flow reel. La prochaine action recommandee est "
    "de soumettre les meilleurs picks a une revue humaine active sur le terrain et sur les loyers comparables."
)
auto_ai = st.toggle(
    "Activer la note IA automatique pour ce portefeuille",
    value=ai_is_configured(),
    key="portfolio_ai_toggle",
)
if auto_ai:
    narrative = generate_investment_narrative(
        context_type="portfolio_construction",
        payload={
            "strategy": strategy,
            "budget": total_budget,
            "candidate_universe_transactions": int(market_summary["transactions"]),
            "candidate_universe_median_price": round(market_summary["median_price"], 2),
            "selected_assets": int(portfolio_summary["assets"]),
            "deployed_capital": round(portfolio_summary["capital"], 2),
            "budget_remaining": round(portfolio_summary["budget_remaining"], 2),
            "average_score": round(portfolio_summary["avg_score"], 2),
            "upside_dollars": round(portfolio_summary["upside_dollars"], 2),
            "upside_pct": round(portfolio_summary["upside_pct"] * 100, 2),
            "zip_exposure": {key: round(value, 2) for key, value in top_zip_exposure.items()},
            "top_assets": portfolio[
                ["id", "zipcode", "price", "predicted_price", "investment_score"]
            ]
            .head(5)
            .to_dict(orient="records"),
            "portfolio_insights": portfolio_notes[:3],
        },
        fallback_text=fallback_portfolio_note,
    )
    badge = "Gemini live" if narrative["source"] == "gemini" else "Mode hors ligne"
    render_ai_note(badge, narrative["text"])
else:
    render_ai_note("Resume local", fallback_portfolio_note)

st.markdown('<div class="section-label">Selection finale</div>', unsafe_allow_html=True)
portfolio_table = portfolio[
    [
        "id",
        "date",
        "zipcode",
        "price",
        "predicted_price",
        "pricing_gap_positive",
        "pricing_gap_pct_positive",
        "investment_score",
        "quality_bucket",
        "grade",
        "condition",
        "sqft_living",
    ]
].copy()
portfolio_table["date"] = portfolio_table["date"].dt.date
portfolio_table["price"] = portfolio_table["price"].map(lambda value: f"${value:,.0f}")
portfolio_table["predicted_price"] = portfolio_table["predicted_price"].map(lambda value: f"${value:,.0f}")
portfolio_table["pricing_gap_positive"] = portfolio_table["pricing_gap_positive"].map(lambda value: f"${value:,.0f}")
portfolio_table["pricing_gap_pct_positive"] = portfolio_table["pricing_gap_pct_positive"].map(lambda value: f"{value * 100:.1f}%")
portfolio_table["investment_score"] = portfolio_table["investment_score"].map(lambda value: f"{value:.1f}")
st.dataframe(portfolio_table, use_container_width=True, hide_index=True)
