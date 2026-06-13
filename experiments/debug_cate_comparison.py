import sys
import os
import numpy as np
sys.path.insert(0, '.')

import src.preprocessing as prep
from src.metalearners import train_r_learner, get_cate_estimates
from src.causal_forest import train_causal_forest
from src.evaluation import compute_auuc

def main():
    print("Loading data...")
    data_path = 'data/criteo-research-uplift-v2.1.csv.gz'
    df = prep.load_data(data_path, sample_frac=0.1, random_state=42)
    df_train, df_cal, df_test = prep.split_data(df, random_state=42)
    
    X_train, T_train, Y_train = prep.get_XTY(df_train, outcome='conversion')
    X_test, T_test, Y_test = prep.get_XTY(df_test, outcome='conversion')
    
    print("Standardizing features...")
    X_train_s, X_cal_s, X_test_s, _ = prep.standardize_features(X_train, X_train, X_test)
    
    print("Training R-Learner...")
    r_learner = train_r_learner(X_train_s, T_train, Y_train)
    r_cate = get_cate_estimates(r_learner, X_test_s)
    if isinstance(r_cate, tuple):
        r_cate = r_cate[0]
        
    print("Training Causal Forest...")
    cf_model = train_causal_forest(X_train_s, T_train, Y_train)
    cf_cate = cf_model.effect(X_test_s).flatten()
    r_cate = r_cate.flatten() if isinstance(r_cate, np.ndarray) else np.array(r_cate).flatten()

    print("R-Learner CATE - mean:", np.mean(r_cate), "std:", np.std(r_cate), "min:", np.min(r_cate), "max:", np.max(r_cate))
    print("Causal Forest CATE - mean:", np.mean(cf_cate), "std:", np.std(cf_cate), "min:", np.min(cf_cate), "max:", np.max(cf_cate))
    
    identical = np.allclose(r_cate, cf_cate)
    print("Are the two arrays identical?", identical)
    
    if np.std(r_cate) > 0 and np.std(cf_cate) > 0:
        correlation = np.corrcoef(r_cate, cf_cate)[0, 1]
    else:
        correlation = 0.0
    print("Correlation between the two arrays:", correlation)
    
    r_auuc = compute_auuc(r_cate, T_test, Y_test)
    cf_auuc = compute_auuc(cf_cate, T_test, Y_test)
    
    print("R-Learner AUUC:", r_auuc)
    print("Causal Forest AUUC:", cf_auuc)
    
    if identical:
        print("Diagnosis: BUG: same array used for both models")
    elif np.isclose(r_auuc, cf_auuc):
        print("Diagnosis: EVALUATION EDGE CASE: different arrays produce same AUUC — both models rank users identically in the negative direction")
    else:
        print("Diagnosis: OK: models are distinct")

if __name__ == '__main__':
    main()
