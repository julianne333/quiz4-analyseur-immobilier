from __future__ import annotations

import os

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from src.data import compute_market_insights, load_dataset, summarize_market
from src.ui import configure_page, inject_app_css, render_hero


load_dotenv()

# Local .env support for LLM integrations. Downstream modules read the same env vars.
LLM_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

configure_page("King County Market Lab")
inject_app_css()

df = load_dataset()
summary = summarize_market(df)
insights = compute_market_insights(df)

render_hero(
    "King County | Seattle Metro",
    "Market Lab Immobilier",
    "Une application Streamlit pour explorer 21 613 transactions du comte de King, "
    "suivre les dynamiques locales et evaluer rapidement un bien individuel.",
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", f"{int(summary['transactions']):,}")
col2.metric("Prix median", f"${summary['median_price']:,.0f}")
col3.metric("Prix moyen", f"${summary['average_price']:,.0f}")
col4.metric("Prix median / sqft", f"${summary['median_price_per_sqft']:,.0f}")

st.markdown('<div class="section-label">Parcours</div>', unsafe_allow_html=True)

left, right = st.columns([1.2, 1], gap="large")
with left:
    st.markdown(
        """
        <div class="info-card">
            <h3>1. Explorer le marche</h3>
            <p>
                Utilisez la page <strong>01_Marche</strong> pour filtrer le pipeline de ventes,
                comparer les ZIP codes, visualiser la dynamique des prix et localiser les
                transactions sur la carte.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h3>2. Evaluer un bien</h3>
            <p>
                Utilisez la page <strong>02_Evaluer_un_bien</strong> pour charger un bien existant
                ou saisir une cible manuellement, puis obtenir des comparables et une estimation de prix.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h3>3. Construire un portefeuille</h3>
            <p>
                Utilisez la page <strong>03_Portefeuille</strong> pour composer une poche cible,
                allouer un budget et prioriser les meilleurs dossiers selon une strategie d'investissement.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="info-card">
            <h3>Ce que l'app livre</h3>
            <p>
                KPI de marche, evolution mensuelle, carte geographique, dispersion par ZIP,
                selection de comparables, estimation de prix, construction de portefeuille
                et narration IA pour accelerer les notes d'investissement.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-label">Lectures rapides</div>', unsafe_allow_html=True)
for insight in insights:
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Volume par ZIP</div>', unsafe_allow_html=True)
zip_snapshot = (
    df.groupby("zipcode")
    .agg(
        transactions=("id", "count"),
        median_price=("price", "median"),
        median_ppsf=("price_per_sqft", "median"),
    )
    .sort_values("transactions", ascending=False)
    .head(12)
    .reset_index()
)
zip_snapshot["median_price"] = zip_snapshot["median_price"].map(lambda value: f"${value:,.0f}")
zip_snapshot["median_ppsf"] = zip_snapshot["median_ppsf"].map(lambda value: f"${value:,.0f}")
st.dataframe(zip_snapshot, use_container_width=True, hide_index=True)

st.caption(
    "Astuce: lancez l'application puis naviguez avec le menu de pages Streamlit dans la barre laterale."
)
