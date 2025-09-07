import pandas as pd
from sklearn.decomposition import PCA
import umap
import plotly.express as px

def run_pca(expr_df, n_components=4):
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
