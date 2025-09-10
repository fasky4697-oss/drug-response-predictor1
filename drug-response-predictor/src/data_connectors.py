import pandas as pd
import numpy as np
import io

def load_example_data():
    """Load multi-omics example data for drug response prediction"""
    expr_csv = pkg_data('example_data/sample_expression.csv')
    target_csv = pkg_data('example_data/sample_drug_response.csv')
    expr = pd.read_csv(expr_csv, index_col=0)
    target = pd.read_csv(target_csv)
    
    # Extend samples to have more data points
    np.random.seed(42)  # For reproducible results
    n_original_samples = expr.shape[1]
    n_target_samples = 50  # Increase to 50 samples
    
    # Create additional samples by adding noise to existing ones
    additional_samples = n_target_samples - n_original_samples
    if additional_samples > 0:
        # Extend expression data
        new_expr_data = []
        new_sample_names = []
        for i in range(additional_samples):
            base_sample = expr.iloc[:, i % n_original_samples]
            noise = np.random.normal(0, 0.1, size=base_sample.shape)
            new_sample = base_sample + noise
            new_expr_data.append(new_sample)
            new_sample_names.append(f'Sample_{n_original_samples + i + 1}')
        
        new_expr_df = pd.DataFrame(new_expr_data, columns=expr.index).T
        new_expr_df.columns = new_sample_names
        expr = pd.concat([expr, new_expr_df], axis=1)
        
        # Extend drug response data
        for i in range(additional_samples):
            base_response = target.iloc[i % len(target)].copy()
            base_response.iloc[0] = f'Sample_{n_original_samples + i + 1}'
            base_response.iloc[1] = base_response.iloc[1] + np.random.normal(0, 0.3)
            target = pd.concat([target, base_response.to_frame().T], ignore_index=True)
    
    n_samples = expr.shape[1]
    all_sample_names = expr.columns.tolist()
    
    # Generate synthetic multi-omics data for demo
    # Synthetic methylation data
    methylation = pd.DataFrame(
        np.random.beta(2, 5, size=(200, n_samples)),
        index=[f'CpG_{i}' for i in range(200)],
        columns=all_sample_names
    )
    
    # Synthetic copy number variation data
    cnv = pd.DataFrame(
        np.random.normal(0, 0.3, size=(150, n_samples)),
        index=[f'CNV_{i}' for i in range(150)],
        columns=all_sample_names
    )
    
    # Synthetic mutation data (binary)
    mutations = pd.DataFrame(
        np.random.binomial(1, 0.1, size=(100, n_samples)),
        index=[f'MUT_{i}' for i in range(100)],
        columns=all_sample_names
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
    Auto-detect file types from filename or use upload order as fallback.
    Expected files: expression, methylation, cnv, mutations, drug_response
    Return dict of DataFrames or None on failure."""
    if not uploaded_files or len(uploaded_files) < 1:
        return None
    
    data_dict = {}
    file_types = ['expression', 'methylation', 'cnv', 'mutations', 'drug_response']
    used_types = set()
    
    def detect_file_type(filename):
        """Detect file type from filename"""
        filename_lower = filename.lower()
        if 'expression' in filename_lower or 'expr' in filename_lower or 'gene' in filename_lower:
            return 'expression'
        elif 'methylation' in filename_lower or 'methyl' in filename_lower or 'cpg' in filename_lower:
            return 'methylation'
        elif 'cnv' in filename_lower or 'copy' in filename_lower:
            return 'cnv'
        elif 'mutation' in filename_lower or 'mut' in filename_lower:
            return 'mutations'
        elif 'drug' in filename_lower or 'response' in filename_lower or 'target' in filename_lower:
            return 'drug_response'
        return None
    
    try:
        # First pass: try to detect from filename
        for file in uploaded_files:
            df = pd.read_csv(file)
            # if first column looks like identifiers, set as index
            if df.columns[0].lower() in ('gene','gene_id','genes','id','feature','cpg','mutation'):
                df = df.set_index(df.columns[0])
            
            file_type = detect_file_type(file.name)
            if file_type and file_type not in used_types:
                data_dict[file_type] = df
                used_types.add(file_type)
        
        # Second pass: assign remaining files by order for files that couldn't be detected
        remaining_types = [ft for ft in file_types if ft not in used_types]
        file_idx = 0
        
        for file in uploaded_files:
            file_type = detect_file_type(file.name)
            # If file type not detected or file wasn't processed in first pass
            if file_type is None and file_idx < len(remaining_types):
                df = pd.read_csv(file)
                if df.columns[0].lower() in ('gene','gene_id','genes','id','feature','cpg','mutation'):
                    df = df.set_index(df.columns[0])
                data_dict[remaining_types[file_idx]] = df
                file_idx += 1
        
        return data_dict if data_dict else None
    except Exception as e:
        print('read_uploaded_csvs error', e)
        return None

# helper to return packaged example file path
def pkg_data(relpath):
    import os
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), relpath)
