import sys
import os
sys.path.insert(0, '.')

import numpy as np
import joblib
import src.preprocessing as prep
import src.propensity as prop
from src.metalearners import train_s_learner, train_t_learner, train_x_learner, train_r_learner, get_cate_estimates

def main():
    print("="*50)
    print(" EXPERIMENT 03: TRAIN METALEARNERS ")
    print("="*50)
    
    data_path = 'data/criteo-research-uplift-v2.1.csv.gz'
    print(f"Loading data from {data_path} (sample_frac=0.1)...")
    df = prep.load_data(data_path, sample_frac=0.1, random_state=42)
    
    print("Splitting and standardizing data...")
    df_train, df_cal, df_test = prep.split_data(df, random_state=42)
    X_train, T_train, Y_train = prep.get_XTY(df_train, outcome='conversion')
    X_test, T_test, Y_test = prep.get_XTY(df_test, outcome='conversion')
    X_train_s, X_cal_s, X_test_s, _ = prep.standardize_features(X_train, X_train, X_test)
    
    print("Training propensity model...")
    prop_model = prop.train_propensity_model(X_train_s, T_train)
    
    models = {}
    cates = {}
    
    print("\n--- Training S-Learner ---")
    models['S-Learner'] = train_s_learner(X_train_s, T_train, Y_train)
    cates['S-Learner'] = get_cate_estimates(models['S-Learner'], X_test_s)
    
    print("\n--- Training T-Learner ---")
    models['T-Learner'] = train_t_learner(X_train_s, T_train, Y_train)
    cates['T-Learner'] = get_cate_estimates(models['T-Learner'], X_test_s)
    
    print("\n--- Training X-Learner ---")
    models['X-Learner'] = train_x_learner(X_train_s, T_train, Y_train, prop_model)
    cates['X-Learner'] = get_cate_estimates(models['X-Learner'], X_test_s)
    
    print("\n--- Training R-Learner ---")
    models['R-Learner'] = train_r_learner(X_train_s, T_train, Y_train)
    r_cate = get_cate_estimates(models['R-Learner'], X_test_s)
    cates['R-Learner'] = r_cate[0] if isinstance(r_cate, tuple) else r_cate

    os.makedirs('results', exist_ok=True)
    print("\n--- Saving Models ---")
    for name, model in models.items():
        # Ensure consistent naming that evaluation script can pick up
        # We will use lower case with underscore
        file_name = name.lower().replace('-', '_') + '.joblib'
        path = os.path.join('results', file_name)
        joblib.dump(model, path)
        print(f"Saved {name} to {path}")

    print("\n--- Final CATE Summary ---")
    print(f"{'Model':<15} | {'Mean CATE':<12} | {'Std CATE':<12}")
    print("-" * 45)
    for name, cate in cates.items():
        if isinstance(cate, np.ndarray):
            cate = cate.flatten()
        print(f"{name:<15} | {np.mean(cate):<12.6f} | {np.std(cate):<12.6f}")

    print("\nComplete.")

if __name__ == '__main__':
    main()
