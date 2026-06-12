import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_auuc(cate_estimates, T, Y, n_bins=100):
    """
    Computes the Area Under the Uplift Curve (AUUC).
    
    Sorts the population by predicted CATE descending. At each fraction k of the population,
    computes the cumulative uplift (difference in means between treated and control) 
    multiplied by the fraction k to produce a cumulative gain curve.
    
    Args:
        cate_estimates (np.ndarray): Predicted CATE per user.
        T (np.ndarray): Treatment indicators (0 or 1).
        Y (np.ndarray): Binary outcomes.
        n_bins (int): Number of points to evaluate along the curve.
        
    Returns:
        float: The normalized AUUC (Area under model curve minus area under random baseline).
    """
    n = len(Y)
    sort_idx = np.argsort(-cate_estimates)
    T_sorted = T[sort_idx]
    Y_sorted = Y[sort_idx]
    
    fractions = np.linspace(1/n_bins, 1.0, n_bins)
    uplift_curve = []
    
    for frac in fractions:
        k = int(frac * n)
        if k == 0:
            uplift_curve.append(0)
            continue
            
        T_topk = T_sorted[:k]
        Y_topk = Y_sorted[:k]
        
        n_treat = np.sum(T_topk)
        n_control = k - n_treat
        
        if n_treat == 0 or n_control == 0:
            uplift = 0
        else:
            uplift = (np.sum(Y_topk[T_topk == 1]) / n_treat) - (np.sum(Y_topk[T_topk == 0]) / n_control)
            
        # Multiply by fraction to get the cumulative gain curve which goes from 0 to ATE
        uplift_gain = uplift * frac
        uplift_curve.append(uplift_gain)
        
    uplift_curve = np.array(uplift_curve)
    
    # Random baseline: diagonal from 0 to overall ATE
    n_treat_total = np.sum(T)
    n_control_total = n - n_treat_total
    overall_ate = (np.sum(Y[T == 1]) / n_treat_total) - (np.sum(Y[T == 0]) / n_control_total)
    
    baseline_curve = fractions * overall_ate
    
    # Compute AUUC
    auuc = np.trapezoid(uplift_curve, fractions) - np.trapezoid(baseline_curve, fractions)
    
    return auuc


def compute_qini(cate_estimates, T, Y, n_bins=100):
    """
    Computes the Qini coefficient.
    
    Sorts by predicted CATE descending. The Qini curve at top-k is the number of incremental 
    conversions: (conversions in treated) - (treated count / control count) * (conversions in control).
    
    Args:
        cate_estimates (np.ndarray): Predicted CATE per user.
        T (np.ndarray): Treatment indicators (0 or 1).
        Y (np.ndarray): Binary outcomes.
        n_bins (int): Number of bins.
        
    Returns:
        float: Qini coefficient.
    """
    n = len(Y)
    sort_idx = np.argsort(-cate_estimates)
    T_sorted = T[sort_idx]
    Y_sorted = Y[sort_idx]
    
    fractions = np.linspace(1/n_bins, 1.0, n_bins)
    qini_curve = []
    
    for frac in fractions:
        k = int(frac * n)
        if k == 0:
            qini_curve.append(0)
            continue
            
        T_topk = T_sorted[:k]
        Y_topk = Y_sorted[:k]
        
        n_treat = np.sum(T_topk)
        n_control = k - n_treat
        
        if n_control == 0:
            q = 0
        else:
            conv_treat = np.sum(Y_topk[T_topk == 1])
            conv_control = np.sum(Y_topk[T_topk == 0])
            q = conv_treat - (n_treat / n_control) * conv_control
            
        qini_curve.append(q)
        
    qini_curve = np.array(qini_curve)
    
    # Overall Qini point at k=n
    overall_n_treat = np.sum(T)
    overall_n_control = n - overall_n_treat
    overall_conv_treat = np.sum(Y[T == 1])
    overall_conv_control = np.sum(Y[T == 0])
    overall_qini = overall_conv_treat - (overall_n_treat / overall_n_control) * overall_conv_control
    
    baseline_curve = fractions * overall_qini
    
    # Area normalized by total population
    qini_area = np.trapezoid(qini_curve, fractions * n) - np.trapezoid(baseline_curve, fractions * n)
    qini_coef = qini_area / n
    
    return qini_coef


def compute_calibration(cate_estimates, T, Y, n_bins=10):
    """
    Computes calibration metrics by binning users by predicted CATE and comparing 
    to the actual ATE observed within that bin.
    
    Args:
        cate_estimates (np.ndarray): Predicted CATE per user.
        T (np.ndarray): Treatment indicators (0 or 1).
        Y (np.ndarray): Binary outcomes.
        n_bins (int): Number of bins (e.g., 10 for deciles).
        
    Returns:
        tuple: (bin_centers, actual_ates, calibration_error)
    """
    n = len(Y)
    sort_idx = np.argsort(-cate_estimates)
    cate_sorted = cate_estimates[sort_idx]
    T_sorted = T[sort_idx]
    Y_sorted = Y[sort_idx]
    
    bin_size = n // n_bins
    bin_centers = []
    actual_ates = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n
        
        cate_bin = cate_sorted[start_idx:end_idx]
        T_bin = T_sorted[start_idx:end_idx]
        Y_bin = Y_sorted[start_idx:end_idx]
        
        mean_cate = np.mean(cate_bin)
        
        n_treat = np.sum(T_bin)
        n_control = len(T_bin) - n_treat
        
        if n_treat == 0 or n_control == 0:
            actual_ate = 0
        else:
            actual_ate = np.mean(Y_bin[T_bin == 1]) - np.mean(Y_bin[T_bin == 0])
            
        bin_centers.append(mean_cate)
        actual_ates.append(actual_ate)
        
    bin_centers = np.array(bin_centers)
    actual_ates = np.array(actual_ates)
    
    calibration_error = np.mean(np.abs(bin_centers - actual_ates))
    
    return bin_centers, actual_ates, calibration_error


def compute_policy_value(cate_estimates, T, Y, targeting_fractions=[0.05, 0.10, 0.20, 0.30]):
    """
    Computes the policy value (incremental conversion rate) for targeting the top K% of users.
    
    Args:
        cate_estimates (np.ndarray): Predicted CATE per user.
        T (np.ndarray): Treatment indicators.
        Y (np.ndarray): Binary outcomes.
        targeting_fractions (list): Fractions of the population to evaluate.
        
    Returns:
        dict: Policy values mapped to targeting fractions.
    """
    n = len(Y)
    sort_idx = np.argsort(-cate_estimates)
    T_sorted = T[sort_idx]
    Y_sorted = Y[sort_idx]
    
    n_treat_total = np.sum(T)
    n_control_total = n - n_treat_total
    overall_ate = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
    
    print("\n--- Policy Value Table ---")
    print(f"{'Fraction':<10} | {'Policy Value':<15} | {'Random Baseline':<15} | {'Lift':<10}")
    print("-" * 55)
    
    results = {}
    
    for frac in targeting_fractions:
        k = int(frac * n)
        if k == 0:
            continue
            
        T_topk = T_sorted[:k]
        Y_topk = Y_sorted[:k]
        
        n_treat = np.sum(T_topk)
        n_control = k - n_treat
        
        if n_treat == 0 or n_control == 0:
            policy_val = 0
        else:
            policy_val = np.mean(Y_topk[T_topk == 1]) - np.mean(Y_topk[T_topk == 0])
            
        lift = policy_val - overall_ate
        results[frac] = policy_val
        
        print(f"{frac:<10.2f} | {policy_val:<15.6f} | {overall_ate:<15.6f} | {lift:<10.6f}")
        
    return results


def run_placebo_test(X_train, T_train, Y_train, X_test, T_test, Y_test, model_trainer_fn, cate_getter_fn, original_auuc, n_shuffles=1, random_state=42):
    """
    Runs a placebo test by shuffling treatment assignments and retraining the model.
    The resulting AUUC should be around 0.
    
    Args:
        X_train, T_train, Y_train: Training data.
        X_test, T_test, Y_test: Testing data.
        model_trainer_fn: Function to train the model.
        cate_getter_fn: Function to extract CATE estimates.
        original_auuc (float): Original AUUC to compare against.
        n_shuffles (int): Number of shuffles.
        random_state (int): Random seed.
        
    Returns:
        float: Placebo AUUC.
    """
    np.random.seed(random_state)
    T_shuffled = np.random.permutation(T_train)
    
    print(f"\nRunning placebo test by shuffling treatment assignment...")
    model_shuffled = model_trainer_fn(X_train, T_shuffled, Y_train)
    
    cate_estimates_shuffled = cate_getter_fn(model_shuffled, X_test)
    if isinstance(cate_estimates_shuffled, tuple):
        cate_estimates_shuffled = cate_estimates_shuffled[0]
        
    placebo_auuc = compute_auuc(cate_estimates_shuffled, T_test, Y_test)
    
    diff = original_auuc - placebo_auuc
    
    print("\n--- Placebo Test Results ---")
    print(f"Original AUUC: {original_auuc:.6f}")
    print(f"Placebo AUUC:  {placebo_auuc:.6f}")
    print(f"Difference:    {diff:.6f}")
    
    return placebo_auuc


def print_evaluation_summary(model_name, auuc, qini, calibration_error, policy_values):
    """
    Prints a clean formatted summary table of evaluation metrics.
    """
    print(f"\n========================================")
    print(f" EVALUATION SUMMARY: {model_name}")
    print(f"========================================")
    print(f"AUUC (Area Under Uplift Curve): {auuc:.6f}")
    print(f"Qini Coefficient:               {qini:.6f}")
    print(f"Calibration Error (MAE):        {calibration_error:.6f}")
    print(f"Policy Values (Top K%):")
    for frac, val in policy_values.items():
        print(f"  - Top {frac*100:.0f}%: {val:.6f}")
    print(f"========================================\n")


if __name__ == '__main__':
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    import src.preprocessing as prep
    from src.metalearners import train_s_learner, train_t_learner, get_cate_estimates
    
    print("Starting Evaluation smoke test...")
    data_path = os.path.join(project_root, 'data', 'criteo-research-uplift-v2.1.csv.gz')
    
    try:
        print("Loading and splitting data...")
        df = prep.load_data(data_path, sample_frac=0.1)
        df_train, df_cal, df_test = prep.split_data(df)
        
        X_train, T_train, Y_train = prep.get_XTY(df_train, outcome='conversion')
        X_test, T_test, Y_test = prep.get_XTY(df_test, outcome='conversion')
        
        print("Standardizing features...")
        X_train_scaled, _, X_test_scaled, _ = prep.standardize_features(X_train, X_test, X_test)
        
        # Train S-Learner
        print("\n--- Training S-Learner ---")
        s_learner = train_s_learner(X_train_scaled, T_train, Y_train)
        s_cate = get_cate_estimates(s_learner, X_test_scaled)
        
        # Train T-Learner
        print("\n--- Training T-Learner ---")
        t_learner = train_t_learner(X_train_scaled, T_train, Y_train)
        t_cate = get_cate_estimates(t_learner, X_test_scaled)
        
        # Evaluate S-Learner
        print("\nEvaluating S-Learner...")
        s_auuc = compute_auuc(s_cate, T_test, Y_test)
        s_qini = compute_qini(s_cate, T_test, Y_test)
        _, _, s_cal_err = compute_calibration(s_cate, T_test, Y_test)
        s_policy = compute_policy_value(s_cate, T_test, Y_test)
        print_evaluation_summary("S-Learner", s_auuc, s_qini, s_cal_err, s_policy)
        
        # Evaluate T-Learner
        print("\nEvaluating T-Learner...")
        t_auuc = compute_auuc(t_cate, T_test, Y_test)
        t_qini = compute_qini(t_cate, T_test, Y_test)
        _, _, t_cal_err = compute_calibration(t_cate, T_test, Y_test)
        t_policy = compute_policy_value(t_cate, T_test, Y_test)
        print_evaluation_summary("T-Learner", t_auuc, t_qini, t_cal_err, t_policy)
        
        # Placebo test on S-Learner
        print("\nRunning Placebo Test on S-Learner...")
        run_placebo_test(X_train_scaled, T_train, Y_train, X_test_scaled, T_test, Y_test, 
                         train_s_learner, get_cate_estimates, s_auuc)
                         
        print("\nEvaluation smoke test completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
