# Overview

This is a Multi-Omics Drug Response Predictor application built with Streamlit. The project provides a web-based interface for analyzing cancer drug sensitivity using integrated genomic data from multiple sources (expression, methylation, copy number variation, and mutations). The application implements machine learning workflows to predict drug responses based on multi-omics patient profiles, featuring data integration techniques, dimensionality reduction, and interactive visualizations.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit-based web application** with a multi-tab interface for different analysis stages
- **Interactive data visualization** using Plotly for charts, plots, and UMAP/PCA visualizations
- **File upload system** supporting CSV format for multi-omics data input
- **Sidebar navigation** for data input options and parameter configuration

## Backend Architecture
- **Modular Python architecture** with separation of concerns across distinct modules:
  - `data_connectors.py`: Handles data loading and file processing
  - `preprocessing.py`: Data harmonization and normalization
  - `integration.py`: Multi-omics data integration using PCA and concatenation methods
  - `modeling.py`: Machine learning model training and evaluation
  - `explain.py`: Model interpretability using SHAP (when available)

## Data Processing Pipeline
- **Multi-omics data integration** supporting concatenation, individual PCA per omic type, and weighted PCA approaches
- **Standardization and normalization** of different data types (expression, methylation, CNV, mutations)
- **Dimensionality reduction** using PCA and UMAP for latent factor extraction
- **Sample alignment** across different omics datasets

## Machine Learning Framework
- **Multiple model support** including Random Forest, Support Vector Regression, and Elastic Net
- **Cross-validation** for model evaluation and hyperparameter tuning
- **Regression metrics** for drug response prediction assessment
- **Model interpretability** through SHAP integration (graceful degradation if unavailable)

## Data Management
- **Example data generation** for demonstration purposes using synthetic multi-omics datasets
- **CSV file processing** with automatic data type detection and harmonization
- **Memory-efficient processing** suitable for Streamlit Community Cloud deployment

# External Dependencies

## Core Libraries
- **Streamlit** (≥1.18): Web application framework and user interface
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and preprocessing utilities
- **Plotly**: Interactive data visualization and charting

## Specialized Libraries
- **UMAP-learn**: Uniform Manifold Approximation and Projection for dimensionality reduction
- **SHAP**: Model interpretation and feature importance analysis (optional dependency)
- **Joblib**: Model serialization and parallel processing support

## Development Considerations
- **Lightweight dependency strategy** to ensure compatibility with Streamlit Community Cloud
- **Graceful degradation** for optional features (SHAP) when dependencies are unavailable
- **Extensibility points** identified for future integration with DepMap, GDSC, GDC databases
- **MOFA integration pathway** planned as alternative to current PCA-based approach