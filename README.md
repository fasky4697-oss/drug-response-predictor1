# Multi-Omics Drug Response Predictor

A comprehensive web application for personalized medicine using multi-omics data integration and machine learning.

## Features

- **Multi-Omics Data Integration**: Expression, methylation, CNV, and mutation data
- **Multiple ML Models**: Random Forest, SVR, and Elastic Net
- **Molecular Subtype Analysis**: K-means clustering for patient stratification
- **Personalized Drug Recommendations**: Patient-specific drug response predictions
- **Interactive Visualizations**: UMAP, PCA, and comprehensive plotting

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run drug-response-predictor/app/streamlit_app.py
```

### Streamlit Cloud Deployment
1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy with:
   - **Repository**: `your-username/your-repo-name`
   - **Branch**: `main`
   - **Main file path**: `drug-response-predictor/app/streamlit_app.py`

## Project Structure
```
├── drug-response-predictor/
│   ├── app/
│   │   └── streamlit_app.py          # Main application
│   ├── src/
│   │   ├── data_connectors.py        # Data loading and processing
│   │   ├── integration.py            # Multi-omics integration methods
│   │   ├── modeling.py               # Machine learning models
│   │   ├── preprocessing.py          # Data preprocessing
│   │   └── explain.py                # Model interpretability
│   └── example_data/
│       ├── sample_expression.csv     # Example gene expression data
│       └── sample_drug_response.csv  # Example drug response data
├── requirements.txt                  # Python dependencies
└── .streamlit/
    └── config.toml                   # Streamlit configuration
```

## Usage

1. **Data Input**: Use example data or upload your own multi-omics files
2. **Integration**: Choose from concatenation, PCA-per-omic, or weighted PCA
3. **Modeling**: Train multiple ML models and compare performance
4. **Subtyping**: Identify molecular subtypes using clustering
5. **Recommendations**: Get personalized drug response predictions

## Dependencies

- streamlit>=1.49.0
- pandas>=2.0.0
- numpy>=2.0.0,<2.3.0
- scikit-learn>=1.3.0
- plotly>=5.15.0
- umap-learn>=0.5.0
- joblib>=1.3.0
- shap>=0.42.0