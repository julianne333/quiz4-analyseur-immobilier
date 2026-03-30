from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ai_narration import ai_is_configured, generate_summary_from_prompt
from src.data import load_dataset
from src.property_charts import comparable_bar_chart
from src.ui import configure_page, inject_app_css, render_ai_note, render_hero


def format_property_label(row: pd.Series) -> str:
    return (
        f"{row['id']} | ${row['price']:,.0f} | {row['sqft_living']:,.0f} pi2 | "
        f"grade {int(row['grade'])}"
    )


def find_comparables(df: pd.DataFrame, subject: pd.Series) -> pd.DataFrame:
    min_sqft = float(subject["sqft_living"]) * 0.8
    max_sqft = float(subject["sqft_living"]) * 1.2

    comparables = df[
        (df["id"] != subject["id"])
        & (df["zipcode"] == subject["zipcode"])
        & (df["bedrooms"] == subject["bedrooms"])
        & (df["sqft_living"].between(min_sqft, max_sqft))
    ].copy()

    comparables["sqft_gap"] = (comparables["sqft_living"] - float(subject["sqft_living"])).abs()
    comparables["price_gap"] = (comparables["price"] - float(subject["price"])).abs()

    return comparables.sort_values(["sqft_gap", "price_gap", "date"]).head(10)


def comparable_summary(subject: pd.Series, comparables: pd.DataFrame) -> dict[str, float | str]:
    if comparables.empty:
        return {
            "count": 0,
            "mean_price": 0.0,
            "price_gap": 0.0,
            "price_gap_pct": 0.0,
            "status": "Aucun comparable",
        }

    mean_price = float(comparables["price"].mean())
    gap = float(subject["price"] - mean_price)
    gap_pct = float(gap / mean_price * 100) if mean_price else 0.0

    if gap > 0:
        status = "Surcote"
    elif gap < 0:
        status = "Decote"
    else:
        status = "Prix aligne"

    return {
        "count": int(len(comparables)),
        "mean_price": mean_price,
        "price_gap": gap,
        "price_gap_pct": gap_pct,
        "status": status,
    }


def build_recommendation_prompt(subject: pd.Series, comparables: pd.DataFrame, summary: dict[str, float | str]):
    prompt = f"""
Tu es un analyste immobilier senior. Evalue cette propriete pour un investisseur :

PROPRIETE ANALYSEE :
- Prix : {float(subject['price']):,.0f} $
- Chambres : {int(subject['bedrooms'])} | Salles de bain : {float(subject['bathrooms'])}
- Superficie : {float(subject['sqft_living']):,.0f} pi2 | Terrain : {float(subject['sqft_lot']):,.0f} pi2
- Grade : {int(subject['grade'])}/13 | Condition : {int(subject['condition'])}/5
- Annee de construction : {int(subject['yr_built'])} | Renovee : {"Oui" if bool(subject['is_renovated']) else "Non"}
- Front de mer : {"Oui" if int(subject['waterfront']) == 1 else "Non"} | Vue : {int(subject['view'])}/4

ANALYSE COMPARATIVE :
- Nombre de comparables trouves : {summary['count']}
- Prix moyen des comparables : {float(summary['mean_price']):,.0f} $
- Ecart vs comparables : {float(summary['price_gap']):+,.0f} $ ({float(summary['price_gap_pct']):+.1f}%)
- Statut : {summary['status']}

Redige une recommandation d'investissement en 3-4 paragraphes.
Inclus : evaluation du prix, forces et faiblesses, verdict final
(Acheter / A surveiller / Eviter) avec justification.
N'utilise pas de markdown ni de puces dans la reponse.
""".strip()

    if comparables.empty:
        fallback = (
            f"La propriete {subject['id']} en ZIP {subject['zipcode']} ne dispose pas de comparables suffisants selon "
            "les criteres definis. Il faut donc rester prudent avant toute conclusion definitive. Les points forts "
            f"visibles sont un grade de {int(subject['grade'])}/13, une surface de {float(subject['sqft_living']):,.0f} pi2 "
            f"et un prix affiche de ${float(subject['price']):,.0f}. Le principal risque est l'absence d'un benchmark local "
            "fiable sur des biens vraiment similaires. Verdict provisoire : A surveiller, en attendant d'elargir la recherche "
            "de comparables ou de completer l'analyse par une verification terrain."
        )
        return prompt, fallback

    fallback = (
        f"La propriete analysee affiche un prix de ${float(subject['price']):,.0f}, contre une moyenne de "
        f"${float(summary['mean_price']):,.0f} pour {summary['count']} comparables locaux. "
        f"L'ecart ressort a {float(summary['price_gap']):+,.0f} $ ({float(summary['price_gap_pct']):+.1f}%), ce qui indique "
        f"une situation de {summary['status'].lower()} par rapport au marche local. Les forces principales tiennent au profil "
        f"du bien, notamment son grade de {int(subject['grade'])}/13, sa condition de {int(subject['condition'])}/5 et son "
        f"positionnement dans le ZIP {subject['zipcode']}. Les faiblesses eventuelles concernent surtout le niveau de prix "
        "relatif face aux comparables. Verdict initial : "
        f"{'Acheter' if summary['status'] == 'Decote' else 'A surveiller' if summary['status'] == 'Prix aligne' else 'Eviter'}, "
        "a confirmer par une revue detaillee du bien et des elements qualitatifs non visibles dans le dataset."
    )
    return prompt, fallback


configure_page("02 | Analyse d'une propriete")
inject_app_css()

df = load_dataset()

render_hero(
    "Onglet 2",
    "Analyse d'une propriete",
    "Selectionne une maison, compare-la a des biens similaires et genere une recommandation "
    "d'investissement basee sur ses comparables.",
)

st.markdown('<div class="section-label">Selection de la propriete</div>', unsafe_allow_html=True)
zipcodes = sorted(df["zipcode"].unique())
selected_zipcode = st.selectbox("ZIP code", options=zipcodes)

zip_df = df[df["zipcode"] == selected_zipcode].copy()
bedroom_options = sorted(zip_df["bedrooms"].dropna().unique())
selected_bedrooms = st.selectbox("Nombre de chambres", options=bedroom_options)

candidate_df = zip_df[zip_df["bedrooms"] == selected_bedrooms].sort_values("price", ascending=False)
selected_property_id = st.selectbox(
    "Selectionner une maison",
    options=candidate_df["id"].tolist(),
    format_func=lambda property_id: format_property_label(
        candidate_df.loc[candidate_df["id"] == property_id].iloc[0]
    ),
)

subject_frame = candidate_df[candidate_df["id"] == selected_property_id].head(1).copy()
subject = subject_frame.iloc[0]
comparables = find_comparables(df, subject)
summary = comparable_summary(subject, comparables)

st.markdown('<div class="section-label">Fiche descriptive</div>', unsafe_allow_html=True)
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Prix", f"${float(subject['price']):,.0f}")
metric_2.metric("Prix / pi2", f"${float(subject['price_per_sqft']):,.0f}")
metric_3.metric("Grade", f"{int(subject['grade'])}/13")
metric_4.metric("Condition", f"{int(subject['condition'])}/5")

detail_left, detail_right = st.columns(2, gap="large")
with detail_left:
    st.dataframe(
        pd.DataFrame(
            [
                {"Attribut": "ID", "Valeur": subject["id"]},
                {"Attribut": "ZIP code", "Valeur": subject["zipcode"]},
                {"Attribut": "Chambres", "Valeur": int(subject["bedrooms"])},
                {"Attribut": "Salles de bain", "Valeur": float(subject["bathrooms"])},
                {"Attribut": "Superficie habitable", "Valeur": f"{float(subject['sqft_living']):,.0f} pi2"},
                {"Attribut": "Terrain", "Valeur": f"{float(subject['sqft_lot']):,.0f} pi2"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
with detail_right:
    st.dataframe(
        pd.DataFrame(
            [
                {"Attribut": "Annee de construction", "Valeur": int(subject["yr_built"])},
                {"Attribut": "Renovee", "Valeur": "Oui" if bool(subject["is_renovated"]) else "Non"},
                {"Attribut": "Front de mer", "Valeur": "Oui" if int(subject["waterfront"]) == 1 else "Non"},
                {"Attribut": "Vue", "Valeur": f"{int(subject['view'])}/4"},
                {"Attribut": "Sous-sol", "Valeur": "Oui" if bool(subject["has_basement"]) else "Non"},
                {"Attribut": "Date de vente", "Valeur": str(subject["date"].date())},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

st.markdown('<div class="section-label">Comparables</div>', unsafe_allow_html=True)
comp_1, comp_2, comp_3, comp_4 = st.columns(4)
comp_1.metric("Comparables trouves", f"{summary['count']}")
comp_2.metric("Prix moyen comps", f"${float(summary['mean_price']):,.0f}")
comp_3.metric("Ecart vs comps", f"{float(summary['price_gap']):+,.0f} $", f"{float(summary['price_gap_pct']):+.1f}%")
comp_4.metric("Statut", str(summary["status"]))

if summary["status"] == "Surcote":
    st.warning("La propriete semble en surcote par rapport au marche local des comparables.")
elif summary["status"] == "Decote":
    st.success("La propriete semble en decote par rapport au marche local des comparables.")
elif summary["status"] == "Prix aligne":
    st.info("La propriete parait globalement alignee avec ses comparables.")
else:
    st.info("Pas assez de comparables stricts pour conclure proprement.")

if comparables.empty:
    st.warning(
        "Aucun comparable ne respecte simultanement les criteres: meme zipcode, meme nombre de chambres et superficie habitable a plus ou moins 20%."
    )
else:
    comparable_table = comparables[
        [
            "id",
            "date",
            "zipcode",
            "price",
            "price_per_sqft",
            "sqft_living",
            "grade",
            "condition",
        ]
    ].copy()
    comparable_table["date"] = comparable_table["date"].dt.date
    comparable_table["price"] = comparable_table["price"].map(lambda value: f"${value:,.0f}")
    comparable_table["price_per_sqft"] = comparable_table["price_per_sqft"].map(lambda value: f"${value:,.0f}")
    comparable_table["sqft_living"] = comparable_table["sqft_living"].map(lambda value: f"{value:,.0f}")
    st.dataframe(comparable_table, use_container_width=True, hide_index=True)

st.markdown('<div class="section-label">Visualisation comparative</div>', unsafe_allow_html=True)
st.pyplot(comparable_bar_chart(subject, comparables), clear_figure=True, use_container_width=True)

st.markdown('<div class="section-label">Recommandation generee par LLM</div>', unsafe_allow_html=True)
prompt, fallback_text = build_recommendation_prompt(subject, comparables, summary)

if st.session_state.get("property_recommendation_prompt") != prompt:
    st.session_state.pop("property_recommendation_result", None)
    st.session_state["property_recommendation_prompt"] = prompt

if st.button("Generer une recommandation", use_container_width=True):
    with st.spinner("Generation de la recommandation en cours..."):
        st.session_state["property_recommendation_result"] = generate_summary_from_prompt(
            prompt=prompt,
            fallback_text=fallback_text,
        )

if "property_recommendation_result" in st.session_state:
    result = st.session_state["property_recommendation_result"]
    badge = "Gemini live" if result["source"] == "gemini" else "Mode hors ligne"
    render_ai_note(badge, result["text"])
else:
    if ai_is_configured():
        st.info("Clique sur le bouton pour generer une recommandation d'investissement.")
    else:
        st.info("Gemini n'est pas disponible pour le moment. Le bouton utilisera une recommandation locale de secours.")
