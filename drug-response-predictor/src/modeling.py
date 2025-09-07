import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def train_multiple_models(latent_df, y_series, test_size=0.2, random_state=42):
    """Train multiple ML models for drug response prediction"""
    # Align data
    common_samples = list(set(latent_df.index) & set(y_series.index))
    X = latent_df.loc[common_samples].values
    y = y_series.loc[common_samples].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    models = {}
    results = {}
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'cv_scores': rf_scores,
        'test_metrics': evaluate_regression(y_test, rf_pred),
        'predictions': rf_pred
    }
    
    # Support Vector Regression
    svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
    svr_scores = cross_val_score(svr_model, X_train, y_train, cv=5, scoring='r2')
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(X_test)
    
    models['SVR'] = svr_model
    results['SVR'] = {
        'cv_scores': svr_scores,
        'test_metrics': evaluate_regression(y_test, svr_pred),
        'predictions': svr_pred
    }
    
    # Elastic Net
    elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state)
    elastic_scores = cross_val_score(elastic_model, X_train, y_train, cv=5, scoring='r2')
    elastic_model.fit(X_train, y_train)
    elastic_pred = elastic_model.predict(X_test)
    
    models['Elastic Net'] = elastic_model
    results['Elastic Net'] = {
        'cv_scores': elastic_scores,
        'test_metrics': evaluate_regression(y_test, elastic_pred),
        'predictions': elastic_pred
    }
    
    return models, results, X_test, y_test, common_samples

def identify_molecular_subtypes(latent_df, n_clusters=3):
    """Identify molecular subtypes using clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_df.values)
    
    subtype_df = pd.DataFrame({
        'Sample': latent_df.index,
        'Molecular_Subtype': [f'Subtype_{i+1}' for i in cluster_labels]
    })
    
    return subtype_df, kmeans

def predict_drug_ranking(models, latent_df, molecular_subtypes=None):
    """Rank drugs for personalized medicine"""
    predictions = {}
    
    for model_name, model in models.items():
        pred = model.predict(latent_df.values)
        predictions[model_name] = pred
    
    # Create comprehensive prediction DataFrame
    pred_df = pd.DataFrame(predictions, index=latent_df.index)
    
    if molecular_subtypes is not None:
        pred_df = pred_df.merge(
            molecular_subtypes.set_index('Sample'), 
            left_index=True, right_index=True, how='left'
        )
    
    return pred_df

def train_rf(latent_df, y_series, test_size=0.2, random_state=42):
    """Legacy function for backward compatibility"""
    common_samples = list(set(latent_df.index) & set(y_series.index))
    X = latent_df.loc[common_samples].values
    y = y_series.loc[common_samples].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, scores, X_test, y_test, y_pred

def evaluate_regression(y_true, y_pred):
    return {
        'R²': float(r2_score(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(np.mean(np.abs(y_true - y_pred)))
    }
