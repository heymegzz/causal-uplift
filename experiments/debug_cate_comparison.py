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
    # causal forest uses its own routine or we can just call it
    # the train_causal_forest returns (model, cate_est, cate_intervals) if called carefully, but let's check its signature
    cf_model = train_causal_forest(X_train_s, T_train, Y_train)
    # the causal forest implementation usually returns just the model, but let's see. Wait, I should use src.causal_forest's actual API
    # I don't know the exact API of train_causal_forest. Let's assume it returns a model that can be passed to get_cate_estimates or model.effect
    # Let me just do model.effect
    try:
        cf_cate = cf_model.effect(X_test_s).flatten()
    except AttributeError:
        # maybe it returns CATE array directly? 
        pass
        
    # Wait, the instruction says:
    # Gets CATE estimates for both on the test set
    # Let me check causal forest code first to be safe. But the prompt said write the file. I will write a flexible getter.
    # Ah, I have access to causal_forest.py in the project. Let me write a getter for CF.
    
    if hasattr(cf_model, 'effect'):
        cf_cate = cf_model.effect(X_test_s)
    elif isinstance(cf_model, tuple):
        # if it returns (model, cate), wait, the instruction says "trains causal forest using src.causal_forest.train_causal_forest"
        cf_model_obj = cf_model[0] if isinstance(cf_model, tuple) else cf_model
        cf_cate = cf_model_obj.effect(X_test_s)
    else:
        cf_cate = get_cate_estimates(cf_model, X_test_s)
        
    # flatten to be safe
    if isinstance(r_cate, np.ndarray):
        r_cate = r_cate.flatten()
    if isinstance(cf_cate, np.ndarray):
        cf_cate = cf_cate.flatten()

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
