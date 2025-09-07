import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_connectors import load_example_data, read_uploaded_csvs
from src.preprocessing import harmonize_expression, prepare_target
from src.integration import integrate_multi_omics, pca_each_omic, umap_from_latent, run_pca
from src.modeling import train_multiple_models, identify_molecular_subtypes, predict_drug_ranking, train_rf, evaluate_regression
from src.explain import shap_summary_or_message

# Page configuration
st.set_page_config(
    page_title='Multi-Omics Drug Response Predictor',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title('🧬 Multi-Omics Drug Response Predictor')
st.markdown('### Personalized Medicine through Integrated Genomic Analysis')

# Sidebar for data input
st.sidebar.header('📊 Data Input')
use_example = st.sidebar.checkbox('Use example multi-omics data', value=True)

if use_example:
    data_dict = load_example_data()
    st.sidebar.success('✅ Example multi-omics data loaded')
    st.sidebar.write(f"- Expression: {data_dict['expression'].shape}")
    st.sidebar.write(f"- Methylation: {data_dict['methylation'].shape}")
    st.sidebar.write(f"- CNV: {data_dict['cnv'].shape}")
    st.sidebar.write(f"- Mutations: {data_dict['mutations'].shape}")
else:
    uploaded = st.sidebar.file_uploader(
        'Upload multi-omics CSV files',
        accept_multiple_files=True,
        help='Upload files in order: expression, methylation, CNV, mutations, drug_response'
    )
    data_dict = read_uploaded_csvs(uploaded)

if data_dict is None:
    st.info('📁 Please upload multi-omics data files or use example data to begin analysis.')
    st.stop()

# Main interface tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview", 
    "🔬 Multi-Omics Integration", 
    "🤖 ML Models", 
    "🎯 Molecular Subtypes", 
    "💊 Drug Recommendations"
])

with tab1:
    st.header('Multi-Omics Data Overview')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('📊 Data Summary')
        for omic_type, df in data_dict.items():
            if omic_type != 'drug_response':
                st.metric(
                    f"{omic_type.title()} Features", 
                    f"{df.shape[0]:,}",
                    help=f"Shape: {df.shape[0]} features × {df.shape[1]} samples"
                )
    
    with col2:
        st.subheader('🎯 Drug Response Data')
        if 'drug_response' in data_dict:
            drug_df = data_dict['drug_response']
            st.dataframe(drug_df.head())
            
            # Drug response distribution
            if len(drug_df.columns) > 1:
                response_col = drug_df.columns[1] if 'response' in drug_df.columns[1].lower() else drug_df.columns[1]
                fig = px.histogram(
                    drug_df, 
                    x=response_col, 
                    title="Drug Response Distribution",
                    color_discrete_sequence=['#FF6B6B']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Show sample data previews
    st.subheader('🔍 Data Previews')
    for omic_type, df in data_dict.items():
        if omic_type != 'drug_response':
            with st.expander(f"{omic_type.title()} Data Preview"):
                st.dataframe(df.head())

with tab2:
    st.header('🔬 Multi-Omics Integration')
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader('Integration Parameters')
        integration_method = st.selectbox(
            'Integration Method',
            ['concatenation', 'pca_each', 'weighted_pca'],
            help='Method for combining multi-omics data'
        )
        
        n_components = st.slider(
            'Number of Latent Factors',
            min_value=5, max_value=20, value=10,
            help='Dimensionality of integrated representation'
        )
        
        if st.button('🔄 Run Integration', type='primary'):
            with st.spinner('Integrating multi-omics data...'):
                if integration_method == 'pca_each':
                    st.session_state['latent_df'] = pca_each_omic(data_dict, n_components_per_omic=3)
                else:
                    integrated_result = integrate_multi_omics(data_dict, n_components, integration_method)
                    if isinstance(integrated_result, tuple):
                        st.session_state['latent_df'] = integrated_result[0]
                    else:
                        st.session_state['latent_df'] = integrated_result
                
                st.success('✅ Integration completed!')
    
    with col2:
        if 'latent_df' in st.session_state:
            st.subheader('📈 Integrated Latent Factors')
            latent_df = st.session_state['latent_df']
            st.dataframe(latent_df.head())
            
            # UMAP visualization
            st.subheader('🗺️ UMAP Visualization')
            umap_fig = umap_from_latent(latent_df)
            st.plotly_chart(umap_fig, use_container_width=True)

with tab3:
    st.header('🤖 Machine Learning Models')
    
    if 'latent_df' not in st.session_state:
        st.warning('⚠️ Please run multi-omics integration first in the Integration tab.')
    else:
        latent_df = st.session_state['latent_df']
        
        # Prepare target data
        drug_df = data_dict['drug_response']
        expr_h = harmonize_expression(data_dict['expression'])
        y, sample_index = prepare_target(drug_df, latent_df.index.tolist())
        
        if y is None:
            st.error('❌ Could not align samples between latent factors and drug response data.')
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader('Model Parameters')
                test_size = st.slider('Test Set Fraction', 0.1, 0.5, 0.2)
                
                if st.button('🚀 Train Models', type='primary'):
                    with st.spinner('Training multiple ML models...'):
                        models, results, X_test, y_test, common_samples = train_multiple_models(
                            latent_df, y, test_size=test_size
                        )
                        
                        st.session_state['models'] = models
                        st.session_state['results'] = results
                        st.session_state['X_test'] = X_test
                        st.session_state['y_test'] = y_test
                        
                    st.success('✅ Models trained successfully!')
            
            with col2:
                if 'results' in st.session_state:
                    st.subheader('📊 Model Performance Comparison')
                    results = st.session_state['results']
                    
                    # Create performance comparison
                    performance_data = []
                    for model_name, result in results.items():
                        performance_data.append({
                            'Model': model_name,
                            'CV R²': f"{result['cv_scores'].mean():.3f} ± {result['cv_scores'].std():.3f}",
                            'Test R²': f"{result['test_metrics']['R²']:.3f}",
                            'Test RMSE': f"{result['test_metrics']['RMSE']:.3f}",
                            'Test MAE': f"{result['test_metrics']['MAE']:.3f}"
                        })
                    
                    performance_df = pd.DataFrame(performance_data)
                    st.dataframe(performance_df, use_container_width=True)
                    
                    # Visualization of predictions vs actual
                    if 'y_test' in st.session_state:
                        st.subheader('🎯 Predictions vs Actual')
                        y_test = st.session_state['y_test']
                        
                        fig = make_subplots(
                            rows=1, cols=len(results),
                            subplot_titles=list(results.keys())
                        )
                        
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                        for i, (model_name, result) in enumerate(results.items()):
                            fig.add_trace(
                                go.Scatter(
                                    x=y_test,
                                    y=result['predictions'],
                                    mode='markers',
                                    name=model_name,
                                    marker=dict(color=colors[i % len(colors)]),
                                    showlegend=False
                                ),
                                row=1, col=i+1
                            )
                            
                            # Add diagonal line
                            min_val, max_val = min(y_test.min(), result['predictions'].min()), max(y_test.max(), result['predictions'].max())
                            fig.add_trace(
                                go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    line=dict(dash='dash', color='gray'),
                                    showlegend=False
                                ),
                                row=1, col=i+1
                            )
                        
                        fig.update_layout(height=400, title_text="Model Predictions vs Actual Values")
                        fig.update_xaxes(title_text="Actual Values")
                        fig.update_yaxes(title_text="Predicted Values")
                        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header('🎯 Molecular Subtype Analysis')
    
    if 'latent_df' not in st.session_state:
        st.warning('⚠️ Please run multi-omics integration first.')
    else:
        latent_df = st.session_state['latent_df']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader('Clustering Parameters')
            n_clusters = st.slider('Number of Molecular Subtypes', 2, 6, 3)
            
            if st.button('🔍 Identify Subtypes', type='primary'):
                with st.spinner('Identifying molecular subtypes...'):
                    subtype_df, kmeans_model = identify_molecular_subtypes(latent_df, n_clusters)
                    st.session_state['subtypes'] = subtype_df
                    st.session_state['kmeans_model'] = kmeans_model
                
                st.success('✅ Molecular subtypes identified!')
        
        with col2:
            if 'subtypes' in st.session_state:
                subtype_df = st.session_state['subtypes']
                
                st.subheader('📊 Molecular Subtype Distribution')
                subtype_counts = subtype_df['Molecular_Subtype'].value_counts()
                
                fig = px.pie(
                    values=subtype_counts.values,
                    names=subtype_counts.index,
                    title="Distribution of Molecular Subtypes",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Subtype visualization in latent space
                st.subheader('🗺️ Subtypes in Latent Space')
                
                # Use first two components for visualization
                viz_df = latent_df.iloc[:, :2].copy()
                viz_df = viz_df.merge(subtype_df.set_index('Sample'), left_index=True, right_index=True)
                
                fig = px.scatter(
                    viz_df.reset_index(),
                    x=viz_df.columns[0],
                    y=viz_df.columns[1],
                    color='Molecular_Subtype',
                    hover_data=['index'],
                    title="Molecular Subtypes in Latent Factor Space",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header('💊 Personalized Drug Recommendations')
    
    if 'models' not in st.session_state:
        st.warning('⚠️ Please train ML models first.')
    elif 'latent_df' not in st.session_state:
        st.warning('⚠️ Please run multi-omics integration first.')
    else:
        models = st.session_state['models']
        latent_df = st.session_state['latent_df']
        subtypes = st.session_state.get('subtypes', None)
        
        st.subheader('🎯 Drug Response Predictions')
        
        # Generate predictions for all samples
        pred_df = predict_drug_ranking(models, latent_df, subtypes)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader('Sample Selection')
            selected_sample = st.selectbox(
                'Choose sample for personalized recommendations:',
                options=pred_df.index.tolist()
            )
            
            if subtypes is not None:
                sample_subtype = pred_df.loc[selected_sample, 'Molecular_Subtype']
                st.info(f"🧬 Molecular Subtype: **{sample_subtype}**")
        
        with col2:
            if selected_sample:
                st.subheader(f'📊 Predictions for {selected_sample}')
                
                sample_predictions = pred_df.loc[selected_sample]
                model_predictions = {
                    col: sample_predictions[col] 
                    for col in pred_df.columns 
                    if col not in ['Molecular_Subtype']
                }
                
                # Create bar chart of model predictions
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(model_predictions.keys()),
                        y=list(model_predictions.values()),
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                ])
                
                fig.update_layout(
                    title=f"Drug Response Predictions for {selected_sample}",
                    xaxis_title="ML Models",
                    yaxis_title="Predicted Drug Response",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed predictions
                pred_table = pd.DataFrame({
                    'Model': list(model_predictions.keys()),
                    'Predicted Response': [f"{val:.3f}" for val in model_predictions.values()]
                })
                st.dataframe(pred_table, use_container_width=True)
        
        # Subtype-based analysis
        if subtypes is not None:
            st.subheader('🧬 Subtype-Based Drug Response Analysis')
            
            # Calculate average response by subtype
            subtype_analysis = pred_df.groupby('Molecular_Subtype')[['Random Forest', 'SVR', 'Elastic Net']].mean()
            
            fig = px.bar(
                subtype_analysis.reset_index(),
                x='Molecular_Subtype',
                y=['Random Forest', 'SVR', 'Elastic Net'],
                title="Average Drug Response by Molecular Subtype",
                barmode='group',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show subtype recommendations
            st.subheader('🎯 Subtype-Specific Recommendations')
            best_subtype = subtype_analysis.mean(axis=1).idxmax()
            worst_subtype = subtype_analysis.mean(axis=1).idxmin()
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **Best Responding Subtype:** {best_subtype}")
                st.write(f"Average predicted response: {subtype_analysis.mean(axis=1)[best_subtype]:.3f}")
            
            with col2:
                st.warning(f"⚠️ **Lowest Responding Subtype:** {worst_subtype}")
                st.write(f"Average predicted response: {subtype_analysis.mean(axis=1)[worst_subtype]:.3f}")

# Sidebar additional information
st.sidebar.markdown('---')
st.sidebar.header('ℹ️ About')
st.sidebar.markdown("""
**Multi-Omics Drug Response Predictor**

This application integrates multiple genomic data types to predict drug responses and enable personalized medicine:

🧬 **Multi-Omics Integration**
- Gene expression
- DNA methylation
- Copy number variations
- Mutation data

🤖 **Machine Learning Models**
- Random Forest
- Support Vector Regression
- Elastic Net

🎯 **Personalized Medicine**
- Molecular subtype identification
- Drug response prediction
- Patient-specific recommendations
""")