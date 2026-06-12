import os
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from econml.dml import CausalForestDML

def train_causal_forest(X_train, T_train, Y_train):
    """
    Trains a CausalForestDML model using EconML.
    
    Args:
        X_train (np.ndarray): Training features.
        T_train (np.ndarray): Training treatment assignments.
        Y_train (np.ndarray): Training outcomes.
        
    Returns:
        CausalForestDML: The trained Causal Forest model.
    """
    print("Training Causal Forest...")
    
    # Base models for Y and T
    lgbm_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42
    }
    
    cf_model = CausalForestDML(
        model_y=LGBMClassifier(**lgbm_params),
        model_t=LGBMClassifier(**lgbm_params),
        n_estimators=500,
        min_samples_leaf=50,
        max_depth=None,
        discrete_treatment=True,
        discrete_outcome=True,
        honest=True,
        inference=True,
        n_jobs=-1,
        random_state=42
    )
    
    # Note: CausalForestDML fit signature requires Y first, then T
    cf_model.fit(Y_train, T_train, X=X_train)
    print("Causal Forest training complete")
    
    return cf_model

def get_cf_cate_with_intervals(model, X, alpha=0.05):
    """
    Computes CATE estimates and confidence intervals for a Causal Forest model.
    
    Args:
        model (CausalForestDML): The fitted Causal Forest model.
        X (np.ndarray): Feature matrix.
        alpha (float): Significance level for confidence intervals.
        
    Returns:
        tuple: cate_estimates, lower_bound, upper_bound
    """
    print(f"Generating CATE estimates and {1-alpha:.0%} confidence intervals...")
    
    # Point estimates
    cate_estimates = model.effect(X)
    
    # Confidence intervals
    lower_bound, upper_bound = model.effect_interval(X, alpha=alpha)
    
    # Summary statistics
    mean_cate = np.mean(cate_estimates)
    std_cate = np.std(cate_estimates)
    
    ci_widths = upper_bound - lower_bound
    mean_ci_width = np.mean(ci_widths)
    
    # Fraction of intervals that contain zero
    contains_zero = (lower_bound <= 0) & (upper_bound >= 0)
    frac_contains_zero = np.mean(contains_zero)
    
    print("\n--- Causal Forest CATE Summary ---")
    print(f"Mean CATE:                {mean_cate:.6f}")
    print(f"Std CATE:                 {std_cate:.6f}")
    print(f"Mean CI Width:            {mean_ci_width:.6f}")
    print(f"Fraction containing zero: {frac_contains_zero:.2%}")
    
    return cate_estimates, lower_bound, upper_bound

def get_cf_feature_importance(model, feature_names):
    """
    Extracts and prints feature importances from the Causal Forest model.
    
    Args:
        model (CausalForestDML): The fitted model.
        feature_names (list): List of feature names.
        
    Returns:
        pd.Series: Feature importances sorted descending.
    """
    print("\nExtracting feature importances...")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if callable(importances):
            importances = importances()
    elif hasattr(model, 'feature_importances'):
        importances = model.feature_importances()
        if callable(importances):
            importances = importances()
    else:
        print("Feature importances not directly available on this model.")
        importances = np.zeros(len(feature_names))
        
    # Ensure importances is flat and length matches feature_names
    importances = np.array(importances).flatten()
    
    # Create pandas series and sort
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    print("--- Top 5 Features by Importance ---")
    print(imp_series.head(5))
    
    return imp_series

if __name__ == '__main__':
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    import src.preprocessing as prep
    
    print("Starting Causal Forest smoke test...")
    
    data_path = os.path.join(project_root, 'data', 'criteo-research-uplift-v2.1.csv.gz')
    save_dir = os.path.join(project_root, 'results')
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Load data
        print("Loading and splitting data...")
        df = prep.load_data(data_path, sample_frac=0.1)
        df_train, df_cal, df_test = prep.split_data(df)
        
        # Get X, T, Y
        X_train, T_train, Y_train = prep.get_XTY(df_train, outcome='conversion')
        X_test, T_test, Y_test = prep.get_XTY(df_test, outcome='conversion')
        
        # Feature names f0 through f11
        feature_names = [f'f{i}' for i in range(12)]
        
        # Standardize features
        print("Standardizing features...")
        # pass dummy cal set as we only need train and test
        X_train_scaled, _, X_test_scaled, scaler = prep.standardize_features(X_train, X_test, X_test)
        
        # Train Causal Forest
        cf_model = train_causal_forest(X_train_scaled, T_train, Y_train)
        
        # Get CATE with intervals
        print("\nEvaluating Causal Forest on test set...")
        cate_est, lb, ub = get_cf_cate_with_intervals(cf_model, X_test_scaled)
        
        # Get feature importance
        importances = get_cf_feature_importance(cf_model, feature_names)
        
        # Save model
        cf_path = os.path.join(save_dir, 'causal_forest.pkl')
        print(f"\nSaving model to {cf_path}...")
        joblib.dump(cf_model, cf_path)
        print("Causal Forest smoke test completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file exists or adjust the path before running the script.")
