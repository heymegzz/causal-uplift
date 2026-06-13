import sys
import os
import time
import json
import joblib
import numpy as np

# Insert root directory to python path
sys.path.insert(0, '.')

start_time = time.time()

print("╔══════════════════════════════════════════════════════════╗")
print("║     CAUSAL UPLIFT MODELING — FULL EXPERIMENT PIPELINE    ║")
print("║     Heterogeneous Treatment Effect Estimation            ║")
print("║     Criteo Uplift Dataset v2.1                          ║")
print("╚══════════════════════════════════════════════════════════╝\n")

CONFIG = {
    "data_path": "data/criteo-research-uplift-v2.1.csv.gz",
    "sample_frac": 0.3,
    "outcome": "conversion",
    "random_state": 42,
    "results_dir": "results",
    "figures_dir": "results/figures"
}

os.makedirs(CONFIG['results_dir'], exist_ok=True)
os.makedirs(CONFIG['figures_dir'], exist_ok=True)

import src.preprocessing as prep
import src.propensity as prop
from src.metalearners import (train_s_learner, train_t_learner, train_x_learner, train_r_learner, get_cate_estimates)
from src.causal_forest import (train_causal_forest, get_cf_cate_with_intervals, get_cf_feature_importance)
import src.evaluation as ev
import src.visualization as viz

def print_step_header(step_num, title):
    print("\n" + "=" * 60)
    print(f"STEP {step_num}: {title.upper()}")
    print("=" * 60)

# Global variables to store state between steps
df_train, df_cal, df_test = None, None, None
X_train_scaled, X_cal_scaled, X_test_scaled = None, None, None
T_train, T_cal, T_test = None, None, None
Y_train, Y_cal, Y_test = None, None, None

propensity_model = None
e_train, e_cal, e_test = None, None, None

models = {}
model_cates = {}
all_results = {}
placebo_results = {}
cf_intervals = {}
feature_importances = None

# Step 1
try:
    print_step_header(1, "Data loading and splitting")
    df = prep.load_data(CONFIG['data_path'], sample_frac=CONFIG['sample_frac'], random_state=CONFIG['random_state'])
    df_train, df_cal, df_test = prep.split_data(df, random_state=CONFIG['random_state'])
    
    X_train, T_train, Y_train = prep.get_XTY(df_train, outcome=CONFIG['outcome'])
    X_cal, T_cal, Y_cal = prep.get_XTY(df_cal, outcome=CONFIG['outcome'])
    X_test, T_test, Y_test = prep.get_XTY(df_test, outcome=CONFIG['outcome'])
    
    X_train_scaled, X_cal_scaled, X_test_scaled, scaler = prep.standardize_features(X_train, X_cal, X_test)
    
    print(f"Train shapes - X: {X_train_scaled.shape}, T: {T_train.shape}, Y: {Y_train.shape}")
    print(f"Cal shapes   - X: {X_cal_scaled.shape}, T: {T_cal.shape}, Y: {Y_cal.shape}")
    print(f"Test shapes  - X: {X_test_scaled.shape}, T: {T_test.shape}, Y: {Y_test.shape}")
except Exception as e:
    print(f"Error in Step 1: {e}")

# Step 2
try:
    print_step_header(2, "Propensity model")
    propensity_model = prop.train_propensity_model(X_train_scaled, T_train)
    e_train = prop.get_propensity_scores(propensity_model, X_train_scaled)
    e_cal = prop.get_propensity_scores(propensity_model, X_cal_scaled)
    e_test = prop.get_propensity_scores(propensity_model, X_test_scaled)
    
    prop.check_overlap(e_train, T_train, save_path=os.path.join(CONFIG['figures_dir'], '01_propensity_overlap.png'))
    e_train_clipped = prop.clip_propensity(e_train)
    e_cal_clipped = prop.clip_propensity(e_cal)
    e_test_clipped = prop.clip_propensity(e_test)
except Exception as e:
    print(f"Error in Step 2: {e}")

# Step 3
try:
    print_step_header(3, "Train all metalearners")
    
    metalearners = [
        ('S-Learner', train_s_learner, lambda m, X: get_cate_estimates(m, X)),
        ('T-Learner', train_t_learner, lambda m, X: get_cate_estimates(m, X)),
        ('X-Learner', lambda X, T, Y: train_x_learner(X, T, Y, propensity_model), lambda m, X: get_cate_estimates(m, X)),
        ('R-Learner', train_r_learner, lambda m, X: get_cate_estimates(m, X))
    ]
    
    for name, train_fn, cate_fn in metalearners:
        print(f"\nTraining {name}...")
        model = train_fn(X_train_scaled, T_train, Y_train)
        models[name] = model
        
        cate_test = cate_fn(model, X_test_scaled)
        model_cates[name] = cate_test
        
        print(f"{name} CATE Summary:")
        print(f"Mean CATE: {np.mean(cate_test):.6f}, Std CATE: {np.std(cate_test):.6f}")
        
        save_path = os.path.join(CONFIG['results_dir'], f"{name.lower().replace('-', '_')}.pkl")
        joblib.dump(model, save_path)
        print(f"Saved {name} to {save_path}")
except Exception as e:
    print(f"Error in Step 3: {e}")

# Step 4
try:
    print_step_header(4, "Train Causal Forest")
    name = 'Causal Forest'
    print(f"Training {name}...")
    cf_model = train_causal_forest(X_train_scaled, T_train, Y_train)
    models[name] = cf_model
    
    cate_test, lb_test, ub_test = get_cf_cate_with_intervals(cf_model, X_test_scaled)
    model_cates[name] = cate_test
    cf_intervals['lb'] = lb_test
    cf_intervals['ub'] = ub_test
    
    feature_names = [f'f{i}' for i in range(X_train_scaled.shape[1])]
    feature_importances = get_cf_feature_importance(cf_model, feature_names)
    
    save_path = os.path.join(CONFIG['results_dir'], 'causal_forest.pkl')
    joblib.dump(cf_model, save_path)
    print(f"Saved Causal Forest to {save_path}")
except Exception as e:
    print(f"Error in Step 4: {e}")

# Step 5
try:
    print_step_header(5, "Evaluation")
    for name, cate_test in model_cates.items():
        auuc = ev.compute_auuc(cate_test, T_test, Y_test)
        qini = ev.compute_qini(cate_test, T_test, Y_test)
        _, _, cal_error = ev.compute_calibration(cate_test, T_test, Y_test)
        policy_vals = ev.compute_policy_value(cate_test, T_test, Y_test, targeting_fractions=[0.05, 0.10, 0.20, 0.30])
        
        all_results[name] = {
            'auuc': auuc,
            'qini': qini,
            'cal_error': cal_error,
            'policy_values': policy_vals
        }
    
    print("\nModel           | AUUC     | Qini     | Cal Error | PV@5%    | PV@10%")
    print("-" * 71)
    for name, res in all_results.items():
        auuc = res['auuc']
        qini = res['qini']
        cal = res['cal_error']
        pv5 = res['policy_values'].get(0.05, 0.0)
        pv10 = res['policy_values'].get(0.10, 0.0)
        print(f"{name:<15} | {auuc:8.6f} | {qini:8.6f} | {cal:9.6f} | {pv5:8.6f} | {pv10:8.6f}")
except Exception as e:
    print(f"Error in Step 5: {e}")

# Step 6
try:
    print_step_header(6, "Placebo tests")
    
    def cate_getter_cf(m, X):
        return get_cf_cate_with_intervals(m, X, alpha=0.05)[0]
    
    placebo_models_fixed = [
        ('S-Learner', train_s_learner, lambda m, X: get_cate_estimates(m, X)),
        ('Causal Forest', train_causal_forest, cate_getter_cf)
    ]
    
    for name, train_fn, cate_fn in placebo_models_fixed:
        if name in all_results:
            real_auuc = all_results[name]['auuc']
            print(f"\nRunning Placebo test for {name}...")
            placebo_auuc = ev.run_placebo_test(X_train_scaled, T_train, Y_train, X_test_scaled, T_test, Y_test, 
                                               train_fn, cate_fn, real_auuc, n_shuffles=1, random_state=CONFIG['random_state'])
            placebo_results[name] = placebo_auuc
except Exception as e:
    print(f"Error in Step 6: {e}")

# Step 7
try:
    print_step_header(7, "Generate all figures")
    if model_cates:
        viz.plot_auuc_comparison(model_cates, T_test, Y_test, save_path=os.path.join(CONFIG['figures_dir'], '02_auuc_comparison.png'))
        viz.plot_qini_curves(model_cates, T_test, Y_test, save_path=os.path.join(CONFIG['figures_dir'], '03_qini_curves.png'))
        viz.plot_cate_calibration(model_cates, T_test, Y_test, save_path=os.path.join(CONFIG['figures_dir'], '04_cate_calibration.png'))
        viz.plot_policy_value(model_cates, T_test, Y_test, save_path=os.path.join(CONFIG['figures_dir'], '06_policy_value_curves.png'))
        viz.plot_cate_distributions(model_cates, save_path=os.path.join(CONFIG['figures_dir'], '08_cate_distribution.png'))
        
    if 'Causal Forest' in models:
        feature_names = [f'f{i}' for i in range(X_train_scaled.shape[1])]
        viz.plot_shap_importance(models['Causal Forest'], X_test_scaled, feature_names, save_path=os.path.join(CONFIG['figures_dir'], '05_shap_summary.png'))
        
    if placebo_results:
        real_auucs = {name: all_results[name]['auuc'] for name in placebo_results.keys()}
        viz.plot_placebo_test(real_auucs, placebo_results, save_path=os.path.join(CONFIG['figures_dir'], '07_placebo_test.png'))
        
    if 'Causal Forest' in model_cates and 'lb' in cf_intervals:
        viz.plot_cf_confidence_intervals(model_cates['Causal Forest'], cf_intervals['lb'], cf_intervals['ub'], save_path=os.path.join(CONFIG['figures_dir'], '09_cf_confidence_intervals.png'))
except Exception as e:
    print(f"Error in Step 7: {e}")

# Step 8
try:
    print_step_header(8, "Save results to JSON")
    
    def convert_to_float(obj):
        if isinstance(obj, dict):
            return {str(k): convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj
        
    clean_results = convert_to_float(all_results)
    
    json_path = os.path.join(CONFIG['results_dir'], 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(clean_results, f, indent=4)
        
    print(f"Results saved to {json_path}")
except Exception as e:
    print(f"Error in Step 8: {e}")

# Step 9
try:
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    end_time = time.time()
    total_time = end_time - start_time
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    print(f"Total runtime: {mins} minutes and {secs} seconds")
    
    if all_results:
        best_auuc_model = max(all_results.keys(), key=lambda k: all_results[k]['auuc'])
        best_qini_model = max(all_results.keys(), key=lambda k: all_results[k]['qini'])
        
        print(f"Winning model by AUUC: {best_auuc_model} ({all_results[best_auuc_model]['auuc']:.6f})")
        print(f"Winning model by Qini: {best_qini_model} ({all_results[best_qini_model]['qini']:.6f})")
        
    if feature_importances is not None:
        print("Top 3 features driving treatment effect heterogeneity:")
        print(feature_importances.head(3).to_string())
        
    print("=" * 60)
except Exception as e:
    print(f"Error in Step 9: {e}")
