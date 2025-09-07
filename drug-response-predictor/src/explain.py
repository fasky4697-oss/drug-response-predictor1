def shap_summary_or_message(model, X_test, feature_names=None):
    try:
        import shap
        import pandas as pd
        explainer = shap.Explainer(model)
        shap_vals = explainer(X_test)
        # show a short textual summary (Streamlit will render)
        mean_abs = pd.DataFrame(abs(shap_vals.values).mean(axis=0).reshape(1,-1), columns=feature_names)
        top = mean_abs.T.sort_values(by=0, ascending=False).head(10)
        return {'shap_top_features': top.index.tolist(), 'shap_values': top[0].tolist()}
    except Exception as e:
        return f'SHAP not available or failed: {e}. Install "shap" to enable explanations.' 
