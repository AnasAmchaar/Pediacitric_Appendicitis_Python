# This script sets up necessary utility functions and packages. Run it before any of the other scripts!
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, chi2_contingency
from statsmodels.stats.multitest import multipletests
import subprocess
import importlib
import sys

########################## Utility functions

def compute_area(x_values, y_values):
    """Calculate area under curve using trapezoidal rule"""
    a = 0
    for i in range(len(x_values) - 1):
        x1 = x_values[i]
        x2 = x_values[i + 1]
        y1 = y_values[i]
        y2 = y_values[i + 1]
        a += (y1 + y2) * (x2 - x1) / 2
    
    return a

def use_package(package_name):
    """Install and import a package if not already installed"""
    try:
        importlib.import_module(package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    finally:
        return importlib.import_module(package_name)

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
    import random
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
    np.random.seed(seed)
    
    pvals = {}
    stats = {}
    
    # Convert y to numeric if it's categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        y_numeric = pd.factorize(y)[0] + 1  # +1 to match R's 1-indexing
    else:
        y_numeric = y
    
    for col in data.columns:
        x1 = data.loc[y_numeric == 1, col].dropna()
        x2 = data.loc[y_numeric == 2, col].dropna()
        
        if pd.api.types.is_numeric_dtype(data[col]):
            # Wilcoxon test for numeric data (equivalent to R's wilcox.test)
            try:
                stat, p_value = wilcoxon(x1, x2, nan_policy='omit')
                pvals[col] = p_value
                stats[col] = stat
            except ValueError:  # If samples have different lengths or other issues
                pvals[col] = np.nan
                stats[col] = np.nan
        else:
            # Chi-square test for categorical data
            try:
                contingency_table = pd.crosstab(data[col], y)
                stat, p_value, _, _ = chi2_contingency(contingency_table)
                pvals[col] = p_value
                stats[col] = stat
            except (ValueError, np.linalg.LinAlgError):
                pvals[col] = np.nan
                stats[col] = np.nan
    
    # Adjust p-values for multiple testing
    pvals_array = np.array(list(pvals.values()))
    adjusted_pvals = multipletests(pvals_array, method=adjust_method)[1]
    
    # Update p-values with adjusted values
    for i, col in enumerate(pvals.keys()):
        pvals[col] = adjusted_pvals[i]
    
    return {'pvals': pvals, 'stats': stats}

########################## Package equivalents
# In Python, we use import statements instead of usePackage function calls

# Import equivalent packages
use_package("pandas")  # Base data manipulation - similar to readxl in R
# use_package("scikit-learn")  # Machine learning package - similar to caret, randomForest in R
use_package("matplotlib")  # Plotting - built into R 
use_package("seaborn")  # Enhanced plotting - similar to ggplot2 in R
use_package("scipy")  # Scientific computations - similar to stats in R
use_package("numpy")  # Numerical computations - built into R
use_package("statsmodels")  # Statistical models - similar to stats in R

# Optional specialized packages (uncomment if needed)
# use_package("gower")  # For Gower distance calculation, equivalent to cluster::daisy
# use_package("impyute")  # For imputation, alternative to mice in R

# Import libraries after ensuring they're installed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from scipy import stats
import gower  # For Gower distance calculation
