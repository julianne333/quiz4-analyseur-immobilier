from __future__ import annotations

import streamlit as st

from src.ai_narration import ai_is_configured, generate_summary_from_prompt
from src.data import load_dataset
from src.market_charts import (
    average_price_by_bedrooms,
    correlation_heatmap,
    price_histogram,
    price_vs_sqft_scatter,
)
from src.ui import configure_page, inject_app_css, render_ai_note, render_hero


def build_market_filters(df):
    with st.sidebar:
        st.markdown("## Filtres du marche")

        price_range = st.slider(
            "Fourchette de prix ($)",
            min_value=int(df["price"].min()),
            max_value=int(df["price"].max()),
            value=(int(df["price"].min()), int(df["price"].max())),
            step=25000,
        )

        bedrooms_range = st.slider(
            "Nombre de chambres",
            min_value=int(df["bedrooms"].min()),
            max_value=int(df["bedrooms"].max()),
            value=(int(df["bedrooms"].min()), int(df["bedrooms"].max())),
        )

        zipcode_selection = st.multiselect(
            "Code postal / zipcode",
            options=sorted(df["zipcode"].unique()),
            default=[],
        )

        grade_range = st.slider(
            "Grade de construction",
            min_value=int(df["grade"].min()),
            max_value=int(df["grade"].max()),
            value=(int(df["grade"].min()), int(df["grade"].max())),
        )

        waterfront_only = st.checkbox("Front de mer uniquement", value=False)

        yr_built_range = st.slider(
            "Annee de construction",
            min_value=int(df["yr_built"].min()),
            max_value=int(df["yr_built"].max()),
            value=(int(df["yr_built"].min()), int(df["yr_built"].max())),
        )

    return {
        "price_range": price_range,
        "bedrooms_range": bedrooms_range,
        "zipcodes": zipcode_selection,
        "grade_range": grade_range,
        "waterfront_only": waterfront_only,
        "yr_built_range": yr_built_range,
    }


def apply_market_filters(df, filters):
    filtered = df[
        df["price"].between(*filters["price_range"])
        & df["bedrooms"].between(*filters["bedrooms_range"])
        & df["grade"].between(*filters["grade_range"])
        & df["yr_built"].between(*filters["yr_built_range"])
    ].copy()

    if filters["zipcodes"]:
        filtered = filtered[filtered["zipcode"].isin(filters["zipcodes"])]

    if filters["waterfront_only"]:
        filtered = filtered[filtered["waterfront"].eq(1)]

    return filtered


def build_market_prompt(filtered_df):
    grade_distribution = (
        filtered_df["grade"].value_counts(normalize=True).sort_index().mul(100).round(1).to_dict()
    )
    bedroom_distribution = (
        filtered_df["bedrooms"].value_counts(normalize=True).sort_index().mul(100).round(1).to_dict()
    )
    top_zipcodes = (
        filtered_df.groupby("zipcode")["price"]
        .agg(["count", "mean", "median"])
        .sort_values("count", ascending=False)
        .head(5)
        .round(0)
        .to_dict(orient="index")
    )
    pct_waterfront = float(filtered_df["waterfront"].mean() * 100) if not filtered_df.empty else 0.0

    prompt = f"""
Tu es un analyste immobilier senior. Voici les statistiques d'un segment du marche immobilier du comte de King (Seattle) :

- Nombre de proprietes : {len(filtered_df)}
- Prix moyen : {filtered_df['price'].mean():,.0f} $
- Prix median : {filtered_df['price'].median():,.0f} $
- Prix min / max : {filtered_df['price'].min():,.0f} $ / {filtered_df['price'].max():,.0f} $
- Prix moyen par pi2 : {filtered_df['price_per_sqft'].mean():,.0f} $
- Repartition par grade : {grade_distribution}
- Repartition par chambres : {bedroom_distribution}
- % front de mer : {pct_waterfront:.1f}%
- Top ZIP codes du segment : {top_zipcodes}

Redige un resume executif de ce segment en 3-4 paragraphes.
Identifie les tendances cles, les risques d'interpretation et les opportunites d'investissement.
N'utilise pas de markdown ni de puces dans la reponse.
""".strip()

    fallback_text = (
        f"Le segment filtre contient {len(filtered_df):,} proprietes, avec un prix moyen de "
        f"${filtered_df['price'].mean():,.0f} et un prix median de ${filtered_df['price'].median():,.0f}. "
        f"La fourchette observee s'etend de ${filtered_df['price'].min():,.0f} a ${filtered_df['price'].max():,.0f}, "
        f"pour un prix moyen de ${filtered_df['price_per_sqft'].mean():,.0f} par pied carre.\n\n"
        f"Du point de vue qualitatif, les grades dominants sont {grade_distribution}. "
        f"La part de biens en front de mer atteint {pct_waterfront:.1f}% du segment, ce qui peut tirer la moyenne "
        "vers le haut lorsque l'echantillon devient plus restreint.\n\n"
        "L'opportunite principale consiste a reperer les sous-segments ou le prix moyen reste modere malgre un niveau "
        "de grade solide ou une localisation recurrente. Le risque principal est de surinterpreter un segment tres filtre, "
        "surtout si certains ZIP codes ou quelques transactions exceptionnelles concentrent la valeur."
    )

    return prompt, fallback_text


configure_page("01 | Marche")
inject_app_css()

full_df = load_dataset()
filters = build_market_filters(full_df)
filtered_df = apply_market_filters(full_df, filters)

render_hero(
    "Onglet 1",
    "Exploration du marche",
    "Explore le marche immobilier du comte de King de maniere interactive a l'aide de filtres, "
    "de KPI et de visualisations analytiques.",
)

if filtered_df.empty:
    st.warning("Aucune propriete ne correspond aux filtres actuels. Elargis les criteres.")
    st.stop()

mean_price_per_sqft = float(filtered_df["price_per_sqft"].mean())

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("N proprietes", f"{len(filtered_df):,}")
kpi_2.metric("Prix moyen", f"${filtered_df['price'].mean():,.0f}")
kpi_3.metric("Prix median", f"${filtered_df['price'].median():,.0f}")
kpi_4.metric("Prix moyen / pi2", f"${mean_price_per_sqft:,.0f}")

st.markdown('<div class="section-label">Visualisations matplotlib</div>', unsafe_allow_html=True)

left, right = st.columns(2, gap="large")
with left:
    st.pyplot(price_histogram(filtered_df), clear_figure=True, use_container_width=True)
with right:
    st.pyplot(price_vs_sqft_scatter(filtered_df), clear_figure=True, use_container_width=True)

left, right = st.columns(2, gap="large")
with left:
    st.pyplot(correlation_heatmap(filtered_df), clear_figure=True, use_container_width=True)
with right:
    st.pyplot(average_price_by_bedrooms(filtered_df), clear_figure=True, use_container_width=True)

st.markdown('<div class="section-label">Resume genere par LLM</div>', unsafe_allow_html=True)
st.caption(
    "Le bouton ci-dessous construit un prompt a partir des statistiques du segment filtre puis appelle Gemini."
)

prompt, fallback_text = build_market_prompt(filtered_df)

if st.session_state.get("market_summary_prompt") != prompt:
    st.session_state.pop("market_summary_result", None)
    st.session_state["market_summary_prompt"] = prompt

if st.button("Generer un resume du marche", use_container_width=True):
    with st.spinner("Generation du resume en cours..."):
        st.session_state["market_summary_result"] = generate_summary_from_prompt(
            prompt=prompt,
            fallback_text=fallback_text,
        )

if "market_summary_result" in st.session_state:
    result = st.session_state["market_summary_result"]
    badge = "Gemini live" if result["source"] == "gemini" else "Mode hors ligne"
    render_ai_note(badge, result["text"])
else:
    if ai_is_configured():
        st.info("Clique sur le bouton pour generer une note d'analyste sur le segment filtre.")
    else:
        st.info("Gemini n'est pas disponible pour le moment. Le bouton utilisera un resume local de secours.")

st.markdown('<div class="section-label">Table des proprietes filtrees</div>', unsafe_allow_html=True)
table = filtered_df[
    [
        "id",
        "date",
        "zipcode",
        "price",
        "bedrooms",
        "grade",
        "yr_built",
        "waterfront",
        "price_per_sqft",
    ]
].copy()
table["date"] = table["date"].dt.date
table["price"] = table["price"].map(lambda value: f"${value:,.0f}")
table["price_per_sqft"] = table["price_per_sqft"].map(lambda value: f"${value:,.0f}")
st.dataframe(table, use_container_width=True, hide_index=True)
