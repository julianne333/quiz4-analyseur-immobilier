# King County Real Estate Analyzer

Application Streamlit pour explorer le marche immobilier du comte de King, analyser une propriete et generer des notes d'analyste avec Gemini.

## Fonctionnalites

- Onglet 1: exploration du marche avec filtres interactifs, KPI et graphiques matplotlib.
- Onglet 2: analyse d'une propriete avec comparables stricts et recommandation LLM.
- Resume du marche et recommandation propriete via Gemini, avec mode de secours hors ligne.

## Lancer en local

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Secrets locaux

Copiez `.streamlit/secrets.toml.example` vers `.streamlit/secrets.toml` puis renseignez votre cle:

```toml
GEMINI_API_KEY = "your_key"
GEMINI_MODEL = "gemini-2.5-flash-lite"
```

L'application accepte aussi `GEMINI_API_KEY` ou `GOOGLE_API_KEY` depuis `.env`.

## Preparation GitHub

Si le dossier n'est pas encore un depot Git:

```bash
git init
git add .
git commit -m "Initial Streamlit real estate analyzer"
```

Puis creez un depot GitHub vide et poussez le projet:

```bash
git branch -M main
git remote add origin <URL_DU_REPO_GITHUB>
git push -u origin main
```

## Deploiement Streamlit Community Cloud

1. Connectez votre compte GitHub a Streamlit Community Cloud.
2. Cliquez sur "New app".
3. Selectionnez votre depot GitHub.
4. Choisissez la branche `main`.
5. Definissez le fichier principal sur `app.py`.
6. Ouvrez "Advanced settings".
7. Collez le contenu de votre `secrets.toml` dans la zone Secrets, par exemple:

```toml
GEMINI_API_KEY = "your_key"
GEMINI_MODEL = "gemini-2.5-flash-lite"
```

8. Lancez le deploiement.

## Structure utile

- `app.py`: page d'accueil
- `pages/01_Marche.py`: exploration du marche
- `pages/02_Evaluer_un_bien.py`: analyse d'une propriete
- `src/data.py`: chargement et preparation des donnees
- `src/market_charts.py`: graphiques matplotlib pour l'onglet marche
- `src/property_charts.py`: graphique comparatif de la propriete

## Notes de deploiement

- `requirements.txt` est a la racine pour que Streamlit Community Cloud installe les dependances.
- `.streamlit/config.toml` est deja en place pour la configuration Streamlit.
- `.streamlit/secrets.toml` ne doit pas etre versionne.
