import streamlit as st
from src.data_connectors import load_example_data, read_uploaded_csvs
from src.preprocessing import harmonize_expression, prepare_target
from src.integration import run_pca, umap_from_latent
from src.modeling import train_rf, evaluate_regression
from src.explain import shap_summary_or_message
import pandas as pd

st.set_page_config(page_title='Drug Response Predictor (MVP)', layout='wide')
st.title('Drug Response Predictor — MVP (PCA + RF)')

st.sidebar.header('Data')
use_example = st.sidebar.checkbox('Use example data', value=True)

if use_example:
    expr_df, target_df = load_example_data()
    st.sidebar.write('Example data loaded: small toy dataset')
else:
    uploaded = st.sidebar.file_uploader('Upload CSVs: expression (genes x samples) and drug response (sample, response)', accept_multiple_files=True)
    expr_df, target_df = read_uploaded_csvs(uploaded)

if expr_df is None or target_df is None:
    st.info('Upload both expression and drug response CSVs or choose example data.')
    st.stop()

st.subheader('Data preview')
c1, c2 = st.columns(2)
with c1:
    st.write('Expression (top rows)')
    st.dataframe(expr_df.head())
with c2:
    st.write('Drug response (top rows)')
    st.dataframe(target_df.head())

# Preprocess
expr_h = harmonize_expression(expr_df)
y, sample_index = prepare_target(target_df, expr_h.columns.tolist())
if y is None:
    st.error('Could not align samples between expression and target. Check sample IDs.')
    st.stop()

st.subheader('Integration: PCA latent factors')
n_components = st.slider('Number of PCA components (latent factors)', min_value=2, max_value=10, value=4)
latent_df = run_pca(expr_h, n_components=n_components)
st.write('Latent factors (samples x components)')
st.dataframe(latent_df.head())

st.plotly_chart(umap_from_latent(latent_df), use_container_width=True)

# Modeling
st.subheader('Train model (RandomForest)')
test_size = st.slider('Test set fraction', min_value=0.1, max_value=0.5, value=0.2)
if st.button('Train & Evaluate'):
    model, scores, X_test, y_test, y_pred = train_rf(latent_df, y, test_size=test_size)
    st.success(f'Cross-validated R^2 (5-fold on training): mean={scores.mean():.3f} ± {scores.std():.3f}')
    st.write('Hold-out test metrics:')
    st.write(evaluate_regression(y_test, y_pred))
    # Explain
    st.subheader('Explainability (SHAP)')
    shap_msg = shap_summary_or_message(model, X_test, feature_names=latent_df.columns.tolist())
    st.write(shap_msg)

st.sidebar.markdown('---')
st.sidebar.markdown('This is a minimal starter. Extend connectors, add MOFA/VAE, or add drug encoders (SMILES).')
