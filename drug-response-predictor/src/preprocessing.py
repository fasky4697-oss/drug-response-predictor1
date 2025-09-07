import pandas as pd
import numpy as np

def harmonize_expression(expr_df):
    """Simple harmonization:
    - If genes on rows and samples on columns: keep as is
    - If samples on rows: transpose
    - Log2(TPM+1)-like transform if necessary
    """
    df = expr_df.copy()
    # if samples appear to be rows (many columns labeled like gene1,gene2) use as-is
    # Heuristic: if index looks numeric (1..n) and columns contain 'TCGA' or 'Sample' strings, transpose
    if df.shape[0] < df.shape[1] and df.index.dtype == object and df.columns.dtype == object and df.columns.str.contains('Sample|TCGA|CCLE', case=False).sum() == 0:
        # assume genes x samples already (genes in index)
        pass
    # ensure genes are rows and samples are columns
    if not all(isinstance(i, str) for i in df.index):
        df = df.set_index(df.columns[0])
    # convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    # fill NaN
    df = df.fillna(df.median(axis=1), axis=0).fillna(0)
    # log2 transform if values high
    if (df.values.max() > 100).any():
        df = np.log2(df + 1)
    return df

def prepare_target(target_df, sample_list):
    """Expect target_df with columns ['sample','response'] or first column sample and second response.
    Align target to sample_list (expression columns). Returns y (pd.Series) and sample_index (list).
    """
    if target_df is None:
        return None, None
    df = target_df.copy()
    if df.shape[1] >= 2:
        sample_col = df.columns[0]
        resp_col = df.columns[1]
    else:
        return None, None
    df = df.rename(columns={sample_col: 'sample', resp_col: 'response'})
    # basic alignment: intersection
    samples = [s for s in sample_list if s in df['sample'].values]
    if not samples:
        # maybe sample ids are in index
        df2 = df.set_index('sample')
        common = [s for s in sample_list if s in df2.index]
        if not common:
            return None, None
        y = df2.loc[common, 'response']
        return y, common
    df2 = df.set_index('sample')
    y = df2.loc[samples, 'response']
    return y, samples
