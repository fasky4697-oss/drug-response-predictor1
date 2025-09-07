# Drug Response Predictor — Minimal Streamlit Starter

This is a minimal, tested skeleton for a **Drug Response Predictor** focusing on cancer drug sensitivity using multi-omics ideas.
The goal: get a working Streamlit app (MVP) that accepts simple input (expression + drug response), runs a light integration (PCA),
trains a RandomForest, and shows basic metrics + an interactive UMAP/PCA plot.

## Contents
- `app/streamlit_app.py` — Streamlit frontend (entrypoint)
- `src/` — helper modules:
  - `data_connectors.py` — sample data loader & upload helpers
  - `preprocessing.py` — simple normalization & harmonization helpers
  - `integration.py` — PCA-based latent factor extraction (quick alternative to MOFA)
  - `modeling.py` — training and prediction wrappers
  - `explain.py` — optional SHAP integration (gracefully degrades if SHAP absent)
- `example_data/` — tiny toy CSVs to test the app
- `requirements.txt` — minimal packages to run the app

## Quick start (locally)
1. Clone repo.
2. Create virtualenv and install requirements:
   ```
   python -m venv .venv
   source .venv/bin/activate    # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Run app:
   ```
   streamlit run app/streamlit_app.py
   ```
4. Upload CSVs via the app or choose the included example data.

## Design notes
- This starter intentionally avoids heavy dependencies (e.g., MOFA) to keep the app runnable on Streamlit Community Cloud.
- Places to extend: replace PCA with MOFA/mofapy2, add connectors for DepMap/GDSC/GDC, add drug branch (SMILES/graph), add more advanced CV and model registry.

## License
MIT
