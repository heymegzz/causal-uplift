import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_propensity_model(X_train, T_train):
    """
    Trains a logistic regression model to predict the probability of treatment P(T=1 | X).
    
    Args:
        X_train (np.ndarray): Training features.
        T_train (np.ndarray): Training treatment assignments.
        
    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    print("Training propensity model...")
    model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    model.fit(X_train, T_train)
    
    # Compute AUC-ROC on training set
    e_train = model.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(T_train, e_train)
    print(f"Training AUC-ROC: {auc:.4f}")
    
    if auc > 0.6:
        print("WARNING: AUC-ROC is unusually high (>0.6). Since treatment was randomized, "
              "AUC should be close to 0.5. Confounding may be present.")
        
    return model

def get_propensity_scores(model, X):
    """
    Computes propensity scores (probability of treatment) for a given feature matrix.
    
    Args:
        model (LogisticRegression): The trained propensity model.
        X (np.ndarray): Feature matrix.
        
    Returns:
        np.ndarray: Array of propensity scores.
    """
    return model.predict_proba(X)[:, 1]

def check_overlap(e_train, T_train, save_path='results/figures/01_propensity_overlap.png'):
    """
    Generates an overlap plot of propensity scores and computes overlap diagnostics.
    
    Args:
        e_train (np.ndarray): Propensity scores.
        T_train (np.ndarray): Treatment assignments.
        save_path (str): Path to save the histogram plot.
        
    Returns:
        dict: A dictionary containing diagnostics including frac_near_zero, 
              frac_near_one, and effective_sample_size.
    """
    # Separate scores
    e_treated = e_train[T_train == 1]
    e_control = e_train[T_train == 0]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(e_treated, bins=50, alpha=0.6, color='blue', density=True, label='Treated (T=1)')
    plt.hist(e_control, bins=50, alpha=0.6, color='orange', density=True, label='Control (T=0)')
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.title('Propensity Score Overlap')
    plt.legend()
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Overlap plot saved to {save_path}")
    
    # Diagnostics
    total_units = len(e_train)
    frac_near_zero = np.mean(e_train < 0.1)
    frac_near_one = np.mean(e_train > 0.9)
    
    # IPS weights
    # Treated weight: 1/e, Control weight: 1/(1-e)
    weights = np.where(T_train == 1, 1.0 / np.clip(e_train, 1e-6, 1.0), 1.0 / np.clip(1.0 - e_train, 1e-6, 1.0))
    sum_w = np.sum(weights)
    sum_w_sq = np.sum(weights ** 2)
    ess = (sum_w ** 2) / sum_w_sq if sum_w_sq > 0 else 0
    
    diagnostics = {
        'frac_near_zero': frac_near_zero,
        'frac_near_one': frac_near_one,
        'effective_sample_size': ess
    }
    
    print("\n--- Overlap Diagnostics ---")
    print(f"Fraction with propensity < 0.1: {frac_near_zero:.4%}")
    print(f"Fraction with propensity > 0.9: {frac_near_one:.4%}")
    print(f"Effective Sample Size (IPS): {ess:,.2f} (out of {total_units:,} original units)")
    
    return diagnostics

def clip_propensity(e, lower=0.05, upper=0.95):
    """
    Clips propensity scores to specified bounds to prevent extreme weights.
    
    Args:
        e (np.ndarray): Propensity scores.
        lower (float): Lower bound.
        upper (float): Upper bound.
        
    Returns:
        np.ndarray: Clipped propensity scores.
    """
    total = len(e)
    num_clipped = np.sum((e < lower) | (e > upper))
    pct_clipped = num_clipped / total
    
    print(f"\n--- Clipping Diagnostics ---")
    print(f"Values clipped: {num_clipped} ({pct_clipped:.2%})")
    
    return np.clip(e, lower, upper)

if __name__ == '__main__':
    # Add src path so we can import preprocessing
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    import src.preprocessing as prep
    
    print("Running propensity modeling smoke test...")
    
    data_path = os.path.join(project_root, 'data', 'criteo-research-uplift-v2.1.csv.gz')
    save_path = os.path.join(project_root, 'results', 'figures', '01_propensity_overlap.png')
    
    try:
        # Load and process data
        df = prep.load_data(data_path, sample_frac=0.1)
        df_train, df_cal, df_test = prep.split_data(df)
        
        X_train, T_train, Y_train = prep.get_XTY(df_train)
        X_cal, T_cal, Y_cal = prep.get_XTY(df_cal)
        X_test, T_test, Y_test = prep.get_XTY(df_test)
        
        X_train_scaled, X_cal_scaled, X_test_scaled, scaler = prep.standardize_features(X_train, X_cal, X_test)
        
        # Propensity modeling
        model = train_propensity_model(X_train_scaled, T_train)
        e_train = get_propensity_scores(model, X_train_scaled)
        
        diagnostics = check_overlap(e_train, T_train, save_path=save_path)
        
        e_train_clipped = clip_propensity(e_train)
        print("\nPropensity smoke test completed successfully!")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file exists or adjust the path before running the script.")
