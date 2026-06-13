import sys
import os
sys.path.insert(0, '.')

import json
import joblib
import numpy as np
import src.preprocessing as prep
import src.evaluation as ev
import src.visualization as viz

def main():
    print("="*50)
    print(" EXPERIMENT 05: FULL EVALUATION PIPELINE ")
    print("="*50)
    
    data_path = 'data/criteo-research-uplift-v2.1.csv.gz'
    print(f"Loading data from {data_path} (sample_frac=0.1)...")
    df = prep.load_data(data_path, sample_frac=0.1, random_state=42)
    
    print("Splitting and standardizing data...")
    df_train, df_cal, df_test = prep.split_data(df, random_state=42)
    X_train, T_train, Y_train = prep.get_XTY(df_train, outcome='conversion')
    X_test, T_test, Y_test = prep.get_XTY(df_test, outcome='conversion')
    X_train_s, X_cal_s, X_test_s, _ = prep.standardize_features(X_train, X_train, X_test)
    
    model_paths = {
        'S-Learner': 'results/s_learner.joblib',
        'T-Learner': 'results/t_learner.joblib',
        'X-Learner': 'results/x_learner.joblib',
        'R-Learner (DML)': 'results/r_learner.joblib',
        'Causal Forest': 'results/causal_forest.pkl'
    }
    
    models = {}
    print("\nLoading models...")
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"  Loaded {name} from {path}")
        else:
            print(f"  WARNING: {name} file not found at {path}. Skipping.")
    
    cates = {}
    metrics = {}
    
    print("\nEvaluating loaded models...")
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        
        # Get CATE estimates depending on model type
        if name == 'Causal Forest':
            cate_est = model.effect(X_test_s)
        else:
            from src.metalearners import get_cate_estimates
            cate_est = get_cate_estimates(model, X_test_s)
            
        if isinstance(cate_est, tuple):
            cate_est = cate_est[0]
        if isinstance(cate_est, np.ndarray):
            cate_est = cate_est.flatten()
            
        cates[name] = cate_est
        
        # Compute metrics
        auuc = ev.compute_auuc(cate_est, T_test, Y_test)
        qini = ev.compute_qini(cate_est, T_test, Y_test)
        _, _, cal_err = ev.compute_calibration(cate_est, T_test, Y_test)
        pv = ev.compute_policy_value(cate_est, T_test, Y_test, targeting_fractions=[0.05, 0.10])
        
        metrics[name] = {
            'auuc': auuc,
            'qini': qini,
            'calibration_error': cal_err,
            'policy_value_05': pv.get(0.05, 0),
            'policy_value_10': pv.get(0.10, 0)
        }
        
    print("\n--- Master Comparison Table ---")
    print(f"{'Model':<18} | {'AUUC':<10} | {'Qini':<10} | {'Cal. Err':<10} | {'PV @ 5%':<10} | {'PV @ 10%':<10}")
    print("-" * 75)
    for name, m in metrics.items():
        print(f"{name:<18} | {m['auuc']:<10.6f} | {m['qini']:<10.3f} | {m['calibration_error']:<10.6f} | {m['policy_value_05']:<10.6f} | {m['policy_value_10']:<10.6f}")

    print("\nGenerating visualizations...")
    os.makedirs('results/figures', exist_ok=True)
    if cates:
        viz.plot_auuc_comparison(cates, T_test, Y_test, save_path='results/figures/02_auuc_comparison.png')
        viz.plot_qini_curves(cates, T_test, Y_test, save_path='results/figures/03_qini_curves.png')
        viz.plot_cate_distributions(cates, save_path='results/figures/08_cate_distribution.png')
        viz.plot_policy_value(cates, T_test, Y_test, save_path='results/figures/06_policy_value_curves.png')
        
        # We can plot CI if causal forest is present
        if 'Causal Forest' in models and hasattr(models['Causal Forest'], 'effect_interval'):
            lower, upper = models['Causal Forest'].effect_interval(X_test_s)
            viz.plot_cf_confidence_intervals(cates['Causal Forest'], lower.flatten(), upper.flatten(), save_path='results/figures/09_cf_intervals.png')
            
    print("\nSaving metrics to results/metrics.json...")
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("\nComplete.")

if __name__ == '__main__':
    main()
