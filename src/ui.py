from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st


def configure_page(title: str) -> None:
    st.set_page_config(
        page_title=title,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_app_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(234,219,200,0.7), transparent 28%),
                    linear-gradient(180deg, #fbf7f1 0%, #f4ece2 100%);
            }
            .hero-panel {
                padding: 1.5rem 1.7rem;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(156,102,68,0.96), rgba(53,80,112,0.92));
                color: #fffdf9;
                box-shadow: 0 20px 55px rgba(33, 49, 60, 0.18);
                margin-bottom: 1rem;
            }
            .hero-panel h1 {
                font-size: 2.3rem;
                margin: 0.2rem 0 0.6rem 0;
                line-height: 1.1;
            }
            .hero-panel p {
                font-size: 1rem;
                margin: 0;
                max-width: 56rem;
            }
            .eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.12rem;
                font-size: 0.75rem;
                opacity: 0.88;
            }
            .info-card {
                padding: 1rem 1.1rem;
                border-radius: 18px;
                background: rgba(255, 251, 245, 0.88);
                border: 1px solid rgba(156, 102, 68, 0.12);
                box-shadow: 0 14px 35px rgba(33, 49, 60, 0.08);
                height: 100%;
            }
            .info-card h3 {
                font-size: 0.95rem;
                text-transform: uppercase;
                letter-spacing: 0.08rem;
                margin-bottom: 0.25rem;
                color: #5c4736;
            }
            .info-card p {
                margin: 0;
                color: #21313c;
            }
            .section-label {
                font-size: 0.85rem;
                letter-spacing: 0.08rem;
                text-transform: uppercase;
                color: #6b5644;
                margin-top: 1rem;
                margin-bottom: 0.2rem;
            }
            .insight-box {
                padding: 1rem 1.2rem;
                border-left: 4px solid #355070;
                border-radius: 14px;
                background: rgba(255, 251, 245, 0.9);
                margin-bottom: 0.8rem;
            }
            .ai-note {
                padding: 1.05rem 1.2rem;
                border-radius: 18px;
                background: rgba(255, 251, 245, 0.94);
                border: 1px solid rgba(53, 80, 112, 0.16);
                box-shadow: 0 14px 35px rgba(33, 49, 60, 0.08);
                margin-bottom: 0.8rem;
            }
            .ai-badge {
                display: inline-block;
                font-size: 0.74rem;
                text-transform: uppercase;
                letter-spacing: 0.08rem;
                padding: 0.24rem 0.55rem;
                border-radius: 999px;
                background: rgba(53, 80, 112, 0.12);
                color: #355070;
                margin-bottom: 0.6rem;
            }
            div[data-testid="stMetric"] {
                background: rgba(255, 251, 245, 0.92);
                border: 1px solid rgba(156, 102, 68, 0.14);
                padding: 0.85rem 1rem;
                border-radius: 18px;
                box-shadow: 0 12px 28px rgba(33, 49, 60, 0.06);
            }
            .block-container {
                padding-top: 1.8rem;
                padding-bottom: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <section class="hero-panel">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_market_filters(df: pd.DataFrame, key_prefix: str = "market") -> dict[str, Any]:
    with st.sidebar:
        st.markdown("## Filtres")
        date_min = df["date"].min().date()
        date_max = df["date"].max().date()
        date_selection = st.date_input(
            "Periode",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
            key=f"{key_prefix}_date",
        )
        if isinstance(date_selection, tuple) and len(date_selection) == 2:
            date_range = date_selection
        else:
            date_range = (date_min, date_max)

        price_min = int(df["price"].min())
        price_max = int(df["price"].quantile(0.99))
        living_min = int(df["sqft_living"].min())
        living_max = int(df["sqft_living"].quantile(0.99))

        filters = {
            "date_range": date_range,
            "price_range": st.slider(
                "Prix ($)",
                min_value=price_min,
                max_value=price_max,
                value=(price_min, price_max),
                step=25000,
                key=f"{key_prefix}_price",
            ),
            "living_range": st.slider(
                "Surface habitable (sqft)",
                min_value=living_min,
                max_value=living_max,
                value=(living_min, living_max),
                step=50,
                key=f"{key_prefix}_living",
            ),
            "bedrooms_range": st.slider(
                "Chambres",
                min_value=int(df["bedrooms"].min()),
                max_value=int(df["bedrooms"].max()),
                value=(int(df["bedrooms"].min()), int(df["bedrooms"].max())),
                key=f"{key_prefix}_bedrooms",
            ),
            "bathrooms_range": st.slider(
                "Salles de bain",
                min_value=float(df["bathrooms"].min()),
                max_value=float(df["bathrooms"].max()),
                value=(float(df["bathrooms"].min()), float(df["bathrooms"].max())),
                step=0.25,
                key=f"{key_prefix}_bathrooms",
            ),
            "grade_range": st.slider(
                "Grade",
                min_value=int(df["grade"].min()),
                max_value=int(df["grade"].max()),
                value=(int(df["grade"].min()), int(df["grade"].max())),
                key=f"{key_prefix}_grade",
            ),
            "condition_range": st.slider(
                "Condition",
                min_value=int(df["condition"].min()),
                max_value=int(df["condition"].max()),
                value=(int(df["condition"].min()), int(df["condition"].max())),
                key=f"{key_prefix}_condition",
            ),
            "zipcodes": st.multiselect(
                "ZIP codes",
                options=sorted(df["zipcode"].unique()),
                default=[],
                key=f"{key_prefix}_zipcodes",
            ),
            "waterfront": st.radio(
                "Front de mer",
                options=["Tous", "Oui", "Non"],
                horizontal=True,
                key=f"{key_prefix}_waterfront",
            ),
            "renovated": st.radio(
                "Renove",
                options=["Tous", "Oui", "Non"],
                horizontal=True,
                key=f"{key_prefix}_renovated",
            ),
            "map_points": st.slider(
                "Points sur la carte",
                min_value=200,
                max_value=4000,
                value=1500,
                step=100,
                key=f"{key_prefix}_map_points",
            ),
        }

        st.caption("Les visuels et tableaux se recalculent a partir de ces filtres.")

    return filters


def metric_delta(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (current / baseline - 1) * 100


def render_ai_note(label: str, text: str) -> None:
    safe_label = html.escape(label)
    safe_text = html.escape(text.replace("**", "")).replace("\n", "<br><br>")
    st.markdown(
        f"""
        <div class="ai-note">
            <div class="ai-badge">{safe_label}</div>
            <div>{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
