import pandas as pd
import numpy as np
import io

def load_example_data():
    """Load multi-omics example data for drug response prediction"""
    expr_csv = pkg_data('example_data/sample_expression.csv')
    target_csv = pkg_data('example_data/sample_drug_response.csv')
    expr = pd.read_csv(expr_csv, index_col=0)
    target = pd.read_csv(target_csv)
    
    # Generate synthetic multi-omics data for demo
    n_samples = expr.shape[1]
    
    # Synthetic methylation data
    methylation = pd.DataFrame(
        np.random.beta(2, 5, size=(200, n_samples)),
        index=[f'CpG_{i}' for i in range(200)],
        columns=expr.columns
    )
    
    # Synthetic copy number variation data
    cnv = pd.DataFrame(
        np.random.normal(0, 0.3, size=(150, n_samples)),
        index=[f'CNV_{i}' for i in range(150)],
        columns=expr.columns
    )
    
    # Synthetic mutation data (binary)
    mutations = pd.DataFrame(
        np.random.binomial(1, 0.1, size=(100, n_samples)),
        index=[f'MUT_{i}' for i in range(100)],
        columns=expr.columns
    )
    
    return {
        'expression': expr,
        'methylation': methylation,
        'cnv': cnv,
        'mutations': mutations,
        'drug_response': target
    }

def read_uploaded_csvs(uploaded_files):
    """Read multiple CSV files for multi-omics data.
    Expected files: expression, methylation, cnv, mutations, drug_response
    Return dict of DataFrames or None on failure."""
    if not uploaded_files or len(uploaded_files) < 1:
        return None
    
    data_dict = {}
    file_types = ['expression', 'methylation', 'cnv', 'mutations', 'drug_response']
    
    try:
        for i, file in enumerate(uploaded_files):
            df = pd.read_csv(file)
            # if first column looks like identifiers, set as index
            if df.columns[0].lower() in ('gene','gene_id','genes','id','feature','cpg','mutation'):
                df = df.set_index(df.columns[0])
            
            # Assign to data type based on file order or name
            if i < len(file_types):
                data_dict[file_types[i]] = df
            else:
                data_dict[f'data_{i}'] = df
        
        return data_dict if data_dict else None
    except Exception as e:
        print('read_uploaded_csvs error', e)
        return None

# helper to return packaged example file path
def pkg_data(relpath):
    import os
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), relpath)
