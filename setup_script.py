# This script sets up necessary utility functions and packages. Run it before any of the other scripts!
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests
import random

########################## Utility functions

# ! setup_script.py ==================
# todo: Random Forest + Bootstrap Variable Selection
# ! ==================================


def compute_area(x_values, y_values):
    """Calculate area under curve"""
    a = 0
    for i in range(len(x_values) - 1):

        # ? calculates area under each two consecutive points
        # ? and sums everything up
        x1 = x_values[i]
        x2 = x_values[i + 1]
        y1 = y_values[i]
        y2 = y_values[i + 1]
        a += (y1 + y2) * (x2 - x1) / 2
    
    return a

def make_transparent(color, alpha=0.4):
    """Make a color transparent by adding alpha channel"""
    import matplotlib.colors as mcolors
    
    # Convert color to RGB if it's a named color
    if isinstance(color, str):
        rgba = mcolors.to_rgba(color)
        return mcolors.to_rgba(rgba, alpha=alpha)
    else:
        # Assuming color is already RGB or RGBA
        return (*color[:3], alpha)

def rf_var_selection(X, y, q, B=100):
    """Random forest variable selection with bootstrap"""
    random.seed(42)
    np.random.seed(42)
    
    freqs_b = {col: 0 for col in X.columns}
    n_samples = X.shape[0]
    
    for b in range(B):
        # Bootstrap sampling
        idx_b = np.random.choice(range(n_samples), size=n_samples, replace=True)
        X_b = X.iloc[idx_b]
        y_b = y.iloc[idx_b]
        
        # Train random forest
        rf_b = RandomForestClassifier(random_state=b)
        rf_b.fit(X_b, y_b)
        
        # Get feature importances
        imps_b = rf_b.feature_importances_
        
        # Get top q features
        sorted_indices = np.argsort(imps_b)[-q:]
        for idx in sorted_indices:
            freqs_b[X.columns[idx]] += 1
    
    # Normalize frequencies
    for key in freqs_b:
        freqs_b[key] /= B
        
    return freqs_b

def summary_stats(data, y, adjust_method="hommel", seed=42):
    """Calculate statistical tests between groups"""
    
    # ? Returns p-values which is a measure of how 2 groups
    # ? are different. The lower the p-value, the more significant
    np.random.seed(seed)
    
    pvals = {col : 0 for col in data.columns}
    stats = {col : 0 for col in data.columns}
    
    # Convert y to numeric if it's categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        y_numeric = pd.factorize(y)[0]
    else:
        y_numeric = y
    
    for col in data.columns:
        x1 = data.loc[y_numeric == 0, col].dropna()
        x2 = data.loc[y_numeric == 1, col].dropna()
        if pd.api.types.is_numeric_dtype(data[col]):
            # Wilcoxon test for numeric data
            stat, p_value = mannwhitneyu(x1, x2)
        else:
            # Chi-square test for categorical data
            contingency_table = pd.crosstab(data[col], y)
            stat, p_value, _, _ = chi2_contingency(contingency_table)
        pvals[col] = p_value
        stats[col] = stat
    
    # Adjust p-values for multiple testing
    pvals_array = np.array(list(pvals.values()))
    adjusted_pvals = multipletests(pvals_array, method=adjust_method)[1]
    
    pvals = {col: adjusted_pvals[i] for i, col in enumerate(pvals.keys())}
    
    return {'pvals': pvals, 'stats': stats}