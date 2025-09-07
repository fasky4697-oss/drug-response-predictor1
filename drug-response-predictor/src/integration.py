import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def integrate_multi_omics(data_dict, n_components=10, integration_method='concatenation'):
    """Integrate multi-omics data using various methods"""
    if integration_method == 'concatenation':
        return concatenate_omics(data_dict, n_components)
    elif integration_method == 'pca_each':
        return pca_each_omic(data_dict, n_components)
    elif integration_method == 'weighted_pca':
        return weighted_pca_integration(data_dict, n_components)
    
def concatenate_omics(data_dict, n_components=10):
    """Simple concatenation after standardization"""
    omics_data = []
    feature_origins = []
    sample_names = None
    
    for omic_type, df in data_dict.items():
        if omic_type == 'drug_response':
            continue
        
        # Standardize each omic type
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.T)  # samples x features
        
        omics_data.append(scaled_data)
        feature_origins.extend([omic_type] * df.shape[0])
        
        if sample_names is None:
            sample_names = df.columns.tolist()
    
    # Concatenate all omics
    integrated_matrix = np.hstack(omics_data)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=min(n_components, integrated_matrix.shape[1]))
    latent_factors = pca.fit_transform(integrated_matrix)
    
    cols = [f'LF{i+1}' for i in range(latent_factors.shape[1])]
    result_df = pd.DataFrame(latent_factors, index=sample_names, columns=cols)
    
    return result_df, pca, feature_origins

def pca_each_omic(data_dict, n_components_per_omic=3):
    """Apply PCA to each omic type separately, then concatenate"""
    latent_factors_list = []
    sample_names = None
    
    for omic_type, df in data_dict.items():
        if omic_type == 'drug_response':
            continue
            
        # Apply PCA to each omic type
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.T)  # samples x features
        
        n_comp = min(n_components_per_omic, scaled_data.shape[1], scaled_data.shape[0])
        pca = PCA(n_components=n_comp)
        latent = pca.fit_transform(scaled_data)
        
        # Create DataFrame with informative column names
        cols = [f'{omic_type}_PC{i+1}' for i in range(latent.shape[1])]
        latent_df = pd.DataFrame(latent, index=df.columns, columns=cols)
        latent_factors_list.append(latent_df)
        
        if sample_names is None:
            sample_names = df.columns.tolist()
    
    # Concatenate all latent factors
    integrated_df = pd.concat(latent_factors_list, axis=1)
    return integrated_df

def run_pca(expr_df, n_components=4):
    """Legacy function for backward compatibility"""
    # expr_df: genes x samples -> we want samples x genes
    mat = expr_df.T.values
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(mat)
    cols = [f'PC{i+1}' for i in range(Z.shape[1])]
    return pd.DataFrame(Z, index=expr_df.columns, columns=cols)

def umap_from_latent(latent_df):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(latent_df.values)
    df = pd.DataFrame(embedding, index=latent_df.index, columns=['UMAP1','UMAP2'])
    df_plot = df.reset_index().rename(columns={'index':'sample'})
    fig = px.scatter(df_plot, x='UMAP1', y='UMAP2', hover_data=['sample'])
    return fig
