import sys
import os
sys.path.insert(0, '.')

import numpy as np
import joblib
import src.preprocessing as prep
from src.causal_forest import train_causal_forest, get_cf_cate_with_intervals, get_cf_feature_importance

def main():
    print("="*50)
    print(" EXPERIMENT 04: TRAIN CAUSAL FOREST ")
    print("="*50)
    
    data_path = 'data/criteo-research-uplift-v2.1.csv.gz'
    print(f"Loading data from {data_path} (sample_frac=0.1)...")
    df = prep.load_data(data_path, sample_frac=0.1, random_state=42)
    
    print("Splitting and standardizing data...")
    df_train, df_cal, df_test = prep.split_data(df, random_state=42)
    X_train, T_train, Y_train = prep.get_XTY(df_train, outcome='conversion')
    X_test, T_test, Y_test = prep.get_XTY(df_test, outcome='conversion')
    X_train_s, X_cal_s, X_test_s, _ = prep.standardize_features(X_train, X_train, X_test)
    
    print("Training CausalForestDML...")
    cf_model = train_causal_forest(X_train_s, T_train, Y_train)
    
    print("Getting CATE estimates and confidence intervals...")
    cate_est, lower, upper = get_cf_cate_with_intervals(cf_model, X_test_s)
    cate_est = cate_est.flatten()
    lower = lower.flatten()
    upper = upper.flatten()
    ci_widths = upper - lower
    ci_contains_zero = (lower <= 0) & (upper >= 0)
    
    print("Getting feature importances...")
    importances = get_cf_feature_importance(cf_model, [f"f{i}" for i in range(12)])
    
    os.makedirs('results', exist_ok=True)
    save_path = 'results/causal_forest.pkl'
    print(f"Saving model to {save_path}...")
    joblib.dump(cf_model, save_path)
    
    print("\n--- Causal Forest CATE Summary ---")
    print(f"Mean CATE: {np.mean(cate_est):.6f}")
    print(f"Std CATE:  {np.std(cate_est):.6f}")
    print(f"Mean 95% CI Width: {np.mean(ci_widths):.6f}")
    print(f"Fraction of CIs containing zero: {np.mean(ci_contains_zero)*100:.2f}%")
    
    print("\n--- Top 5 Features by Importance ---")
    if importances is not None:
        for feat, imp in list(importances.items())[:5]:
            print(f"{feat:<10} | {imp:.6f}")
    else:
        print("Feature importances not available.")
        
    print("\nComplete.")

if __name__ == '__main__':
    main()
