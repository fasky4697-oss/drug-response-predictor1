import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

def train_rf(latent_df, y_series, test_size=0.2, random_state=42):
    # align
    X = latent_df.loc[y_series.index].values
    y = y_series.loc[latent_df.index].values if set(latent_df.index) <= set(y_series.index) else y_series.loc[latent_df.index].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, scores, X_test, y_test, y_pred

def evaluate_regression(y_true, y_pred):
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(mean_squared_error(y_true, y_pred, squared=False))
    }
