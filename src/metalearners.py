import os
import numpy as np
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
from econml.metalearners import SLearner, TLearner, XLearner

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

def train_s_learner(X_train, T_train, Y_train):
    """
    Trains an S-Learner using LightGBM as the base estimator.
    
    Args:
        X_train (np.ndarray): Training features.
        T_train (np.ndarray): Training treatment assignments.
        Y_train (np.ndarray): Training outcomes.
        
    Returns:
        SLearner: The fitted S-Learner model.
    """
    print("Training S-Learner...")
    overall_model = LGBMClassifier(**LGBM_PARAMS)
    s_learner = SLearner(overall_model=overall_model)
    
    s_learner.fit(Y_train, T_train, X=X_train)
    print("S-Learner training complete")
    
    return s_learner

def train_t_learner(X_train, T_train, Y_train):
    """
    Trains a T-Learner using LightGBM as the base estimator for both arms.
    
    Args:
        X_train (np.ndarray): Training features.
        T_train (np.ndarray): Training treatment assignments.
        Y_train (np.ndarray): Training outcomes.
        
    Returns:
        TLearner: The fitted T-Learner model.
    """
    print("Training T-Learner...")
    model_t0 = LGBMClassifier(**LGBM_PARAMS)
    model_t1 = LGBMClassifier(**LGBM_PARAMS)
    t_learner = TLearner(models=(model_t0, model_t1))
    
    t_learner.fit(Y_train, T_train, X=X_train)
    print("T-Learner training complete")
    
    return t_learner

def get_cate_estimates(model, X, model_name='model'):
    """
    Computes Conditional Average Treatment Effect (CATE) estimates.
    
    Args:
        model: Fitted metalearner model.
        X (np.ndarray): Feature matrix.
        model_name (str): Name of the model for display purposes.
        
    Returns:
        np.ndarray: CATE estimates.
    """
    print(f"Generating CATE estimates for {model_name}...")
    cate = model.effect(X)
    
    print(f"\n--- {model_name} CATE Summary ---")
    print(f"Mean CATE: {np.mean(cate):.6f}")
    print(f"Std CATE:  {np.std(cate):.6f}")
    print(f"Min CATE:  {np.min(cate):.6f}")
    print(f"Max CATE:  {np.max(cate):.6f}")
    
    return cate

if __name__ == '__main__':
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    import src.preprocessing as prep
    
    print("Starting metalearners smoke test pipeline...")
    
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
        X_cal, T_cal, Y_cal = prep.get_XTY(df_cal, outcome='conversion')
        X_test, T_test, Y_test = prep.get_XTY(df_test, outcome='conversion')
        
        # Standardize features
        print("Standardizing features...")
        X_train_scaled, X_cal_scaled, X_test_scaled, scaler = prep.standardize_features(X_train, X_cal, X_test)
        
        # Train S-Learner
        s_learner = train_s_learner(X_train_scaled, T_train, Y_train)
        
        # Train T-Learner
        t_learner = train_t_learner(X_train_scaled, T_train, Y_train)
        
        # Get CATE estimates on test set
        print("Evaluating models on test set...")
        cate_s = get_cate_estimates(s_learner, X_test_scaled, model_name="S-Learner")
        cate_t = get_cate_estimates(t_learner, X_test_scaled, model_name="T-Learner")
        
        # Save models
        s_learner_path = os.path.join(save_dir, 's_learner.pkl')
        t_learner_path = os.path.join(save_dir, 't_learner.pkl')
        
        print("Saving models...")
        joblib.dump(s_learner, s_learner_path)
        print(f"S-Learner saved to {s_learner_path}")
        
        joblib.dump(t_learner, t_learner_path)
        print(f"T-Learner saved to {t_learner_path}")
        
        print("\nMetalearners smoke test completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file exists or adjust the path before running the script.")
