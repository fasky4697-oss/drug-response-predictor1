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
    Auto-detect file types from filename and content.
    Expected files: expression, methylation, cnv, mutations, drug_response
    Return dict of DataFrames or None on failure."""
    if not uploaded_files or len(uploaded_files) < 1:
        return None
    
    data_dict = {}
    file_types = ['expression', 'methylation', 'cnv', 'mutations', 'drug_response']
    
    def detect_file_type(filename, df):
        """Detect file type from filename and content"""
        filename_lower = filename.lower()
        
        # Check filename first
        if 'expression' in filename_lower or 'expr' in filename_lower:
            return 'expression'
        elif 'methylation' in filename_lower or 'methyl' in filename_lower:
            return 'methylation'
        elif 'cnv' in filename_lower:
            return 'cnv'
        elif 'mutation' in filename_lower or 'mut' in filename_lower:
            return 'mutations'
        elif 'drug' in filename_lower or 'response' in filename_lower:
            return 'drug_response'
        
        # Check content patterns if filename detection fails
        first_col = df.columns[0].lower() if len(df.columns) > 0 else ''
        if len(df) > 0:
            first_row_index = str(df.index[0]).lower() if not df.index.empty else ''
        else:
            first_row_index = ''
            
        # Check for drug response (has 'sample' and numeric response)
        if df.shape[1] == 2 and ('sample' in first_col or df.iloc[:, 0].dtype == 'object'):
            return 'drug_response'
        
        # Check for methylation (CpG sites)
        if 'cpg' in first_col or 'cg' in first_row_index:
            return 'methylation'
            
        # Check for mutations (mutation names or binary data)
        if 'mut' in first_col or 'mutation' in first_col:
            return 'mutations'
        elif df.select_dtypes(include=['number']).max().max() <= 1 and df.select_dtypes(include=['number']).min().min() >= 0:
            # Binary data might be mutations
            return 'mutations'
            
        # Check for CNV (genomic coordinates)
        if 'chr' in first_row_index or 'feature' in first_col:
            return 'cnv'
            
        # Check for gene expression (gene names)
        if 'gene' in first_col or any(gene in first_row_index for gene in ['brca', 'tp53', 'egfr']):
            return 'expression'
            
        return None
    
    try:
        # Process each file
        file_assignments = {}
        
        for file in uploaded_files:
            df = pd.read_csv(file)
            
            # Set index if first column looks like identifiers
            if df.columns[0].lower() in ('gene','gene_id','genes','id','feature','cpg','mutation'):
                df = df.set_index(df.columns[0])
            
            file_type = detect_file_type(file.name, df)
            if file_type:
                file_assignments[file.name] = (file_type, df)
        
        # Assign detected files to data_dict
        used_types = set()
        for filename, (file_type, df) in file_assignments.items():
            if file_type not in used_types:
                data_dict[file_type] = df
                used_types.add(file_type)
        
        # For remaining files, assign by order
        remaining_types = [ft for ft in file_types if ft not in used_types]
        remaining_files = [f for f in uploaded_files if f.name not in [fn for fn, _ in file_assignments.items()]]
        
        for i, file in enumerate(remaining_files):
            if i < len(remaining_types):
                df = pd.read_csv(file)
                if df.columns[0].lower() in ('gene','gene_id','genes','id','feature','cpg','mutation'):
                    df = df.set_index(df.columns[0])
                data_dict[remaining_types[i]] = df
        
        return data_dict if data_dict else None
    except Exception as e:
        print('read_uploaded_csvs error', e)
        return None

# helper to return packaged example file path
def pkg_data(relpath):
    import os
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), relpath)
