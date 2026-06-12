import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_COLORS = {
    'S-Learner': '#2196F3',
    'T-Learner': '#4CAF50',
    'X-Learner': '#FF9800',
    'R-Learner': '#9C27B0',
    'Causal Forest': '#F44336',
    'Random': '#9E9E9E'
}

def set_style():
    """Sets the global matplotlib style for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 150
    })

def plot_auuc_comparison(model_cates, T, Y, save_path='results/figures/02_auuc_comparison.png'):
    """
    Plots the cumulative AUUC (uplift) curves for multiple models.
    
    Args:
        model_cates (dict): Mapping of model names to their CATE estimate arrays.
        T (np.ndarray): Treatment indicators.
        Y (np.ndarray): Binary outcomes.
        save_path (str): File path to save the figure.
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    n = len(Y)
    n_bins = 100
    fractions = np.linspace(1/n_bins, 1.0, n_bins)
    
    n_treat_total = np.sum(T)
    n_control_total = n - n_treat_total
    overall_ate = (np.sum(Y[T == 1]) / n_treat_total) - (np.sum(Y[T == 0]) / n_control_total)
    baseline_curve = fractions * overall_ate
    
    for model_name, cate in model_cates.items():
        sort_idx = np.argsort(-cate)
        T_sorted = T[sort_idx]
        Y_sorted = Y[sort_idx]
        
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
            uplift_curve.append(uplift * frac)
            
        uplift_curve = np.array(uplift_curve)
        auuc = np.trapezoid(uplift_curve, fractions) - np.trapezoid(baseline_curve, fractions)
        color = MODEL_COLORS.get(model_name, '#333333')
        
        plt.plot(fractions, uplift_curve, label=f"{model_name} (AUUC={auuc:.6f})", color=color, linewidth=2)
        
    plt.plot(fractions, baseline_curve, label='Random Baseline', color=MODEL_COLORS['Random'], linestyle='--', linewidth=2)
    
    plt.title("Uplift Curve Comparison — All Models")
    plt.xlabel("Fraction of Population Targeted")
    plt.ylabel("Cumulative Incremental Conversion Rate")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved figure to {save_path}")

def plot_qini_curves(model_cates, T, Y, save_path='results/figures/03_qini_curves.png'):
    """
    Plots Qini curves for multiple models alongside a horizontal bar chart of Qini coefficients.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    n = len(Y)
    n_bins = 100
    fractions = np.linspace(1/n_bins, 1.0, n_bins)
    
    overall_n_treat = np.sum(T)
    overall_n_control = n - overall_n_treat
    overall_conv_treat = np.sum(Y[T == 1])
    overall_conv_control = np.sum(Y[T == 0])
    overall_qini = overall_conv_treat - (overall_n_treat / overall_n_control) * overall_conv_control
    baseline_curve = fractions * overall_qini
    
    qini_coefs = {}
    
    for model_name, cate in model_cates.items():
        sort_idx = np.argsort(-cate)
        T_sorted = T[sort_idx]
        Y_sorted = Y[sort_idx]
        
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
        qini_area = np.trapezoid(qini_curve, fractions * n) - np.trapezoid(baseline_curve, fractions * n)
        qini_coef = qini_area / n
        qini_coefs[model_name] = qini_coef
        
        color = MODEL_COLORS.get(model_name, '#333333')
        ax1.plot(fractions, qini_curve, label=f"{model_name} (Qini={qini_coef:.2f})", color=color, linewidth=2)
        
    ax1.plot(fractions, baseline_curve, label='Random Baseline', color=MODEL_COLORS['Random'], linestyle='--', linewidth=2)
    ax1.set_title("Qini Curves — All Models")
    ax1.set_xlabel("Fraction of Population Targeted")
    ax1.set_ylabel("Cumulative Incremental Conversions")
    ax1.legend()
    
    # Subplot 2: Bar chart
    sorted_models = sorted(qini_coefs.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_models]
    coefs = [x[1] for x in sorted_models]
    colors = [MODEL_COLORS.get(name, '#333333') for name in names]
    
    ax2.barh(names, coefs, color=colors)
    ax2.set_title("Qini Coefficients")
    ax2.set_xlabel("Qini Coefficient")
    for i, v in enumerate(coefs):
        ax2.text(v, i, f" {v:.2f}", va='center', fontsize=9)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved figure to {save_path}")

def plot_cate_calibration(model_cates, T, Y, save_path='results/figures/04_cate_calibration.png'):
    """
    Plots the predicted CATE vs actual ATE by decile for all models in a 2x3 grid.
    """
    set_style()
    n_models = len(model_cates)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    n_bins = 10
    n = len(Y)
    bin_size = n // n_bins
    
    for i, (model_name, cate) in enumerate(model_cates.items()):
        ax = axes[i]
        sort_idx = np.argsort(-cate)
        cate_sorted = cate[sort_idx]
        T_sorted = T[sort_idx]
        Y_sorted = Y[sort_idx]
        
        bin_centers = []
        actual_ates = []
        
        for j in range(n_bins):
            start_idx = j * bin_size
            end_idx = (j + 1) * bin_size if j < n_bins - 1 else n
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
        mae = np.mean(np.abs(bin_centers - actual_ates))
        
        color = MODEL_COLORS.get(model_name, '#333333')
        ax.scatter(bin_centers, actual_ates, color=color, alpha=0.7)
        
        # Diagonal reference line
        min_val = min(min(bin_centers), min(actual_ates))
        max_val = max(max(bin_centers), max(actual_ates))
        ax.plot([min_val, max_val], [min_val, max_val], '--', color=MODEL_COLORS['Random'])
        
        ax.set_title(model_name)
        ax.text(0.05, 0.95, f"MAE: {mae:.6f}", transform=ax.transAxes, va='top', fontsize=9)
        ax.set_xlabel("Predicted CATE (Bin Mean)")
        ax.set_ylabel("Actual ATE")
        
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])
        
    fig.suptitle("CATE Calibration — Predicted vs Actual Treatment Effect by Decile", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved figure to {save_path}")

def plot_policy_value(model_cates, T, Y, save_path='results/figures/06_policy_value_curves.png'):
    """
    Plots policy values across different targeting fractions.
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    n = len(Y)
    fractions = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    overall_ate = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
    
    for model_name, cate in model_cates.items():
        sort_idx = np.argsort(-cate)
        T_sorted = T[sort_idx]
        Y_sorted = Y[sort_idx]
        
        policy_vals = []
        for frac in fractions:
            k = int(frac * n)
            if k == 0:
                policy_vals.append(0)
                continue
            T_topk = T_sorted[:k]
            Y_topk = Y_sorted[:k]
            
            n_treat = np.sum(T_topk)
            n_control = k - n_treat
            if n_treat == 0 or n_control == 0:
                val = 0
            else:
                val = np.mean(Y_topk[T_topk == 1]) - np.mean(Y_topk[T_topk == 0])
            policy_vals.append(val)
            
        color = MODEL_COLORS.get(model_name, '#333333')
        plt.plot(fractions, policy_vals, marker='o', label=model_name, color=color, linewidth=2)
        
    plt.axhline(overall_ate, color=MODEL_COLORS['Random'], linestyle='--', label='Random Baseline', linewidth=2)
    plt.title("Policy Value vs Targeting Budget")
    plt.xlabel("Fraction of Population Targeted")
    plt.ylabel("Incremental Conversion Rate")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved figure to {save_path}")

def plot_cate_distributions(model_cates, save_path='results/figures/08_cate_distribution.png'):
    """
    Plots histograms of the predicted CATE distribution for all models.
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    for model_name, cate in model_cates.items():
        color = MODEL_COLORS.get(model_name, '#333333')
        plt.hist(cate, bins=50, density=True, alpha=0.5, label=model_name, color=color)
        plt.axvline(np.mean(cate), color=color, linestyle='--', alpha=0.8)
        
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title("CATE Distribution — All Models")
    plt.xlabel("Estimated CATE")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved figure to {save_path}")

def plot_shap_importance(model, X_test, feature_names, save_path='results/figures/05_shap_summary.png'):
    """
    Extracts and plots feature importances, using SHAP if possible, else falling back to a bar chart.
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    success = False
    try:
        import shap
        if hasattr(model, 'model_final'):
            explainer = shap.TreeExplainer(model.model_final)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            success = True
    except Exception as e:
        print(f"SHAP explanation failed: {e}. Falling back to standard feature importances.")
        
    if not success:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if callable(importances):
                importances = importances()
        elif hasattr(model, 'feature_importances'):
            importances = model.feature_importances()
            if callable(importances):
                importances = importances()
        else:
            importances = np.zeros(len(feature_names))
            
        importances = np.array(importances).flatten()
        imp_series = pd.Series(importances, index=feature_names).sort_values()
        
        imp_series.tail(15).plot(kind='barh', color=MODEL_COLORS.get('Causal Forest', '#F44336'))
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        
    plt.title("Feature Importance — Treatment Effect Heterogeneity Drivers")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved figure to {save_path}")

def plot_placebo_test(real_auucs, placebo_auucs, save_path='results/figures/07_placebo_test.png'):
    """
    Plots grouped bar chart of real AUUC vs placebo AUUC.
    """
    set_style()
    models = list(real_auucs.keys())
    x = np.arange(len(models))
    width = 0.35
    
    real_vals = [real_auucs[m] for m in models]
    placebo_vals = [placebo_auucs[m] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    real_colors = [MODEL_COLORS.get(m, '#333333') for m in models]
    
    rects1 = ax.bar(x - width/2, real_vals, width, label='Real AUUC', color=real_colors)
    rects2 = ax.bar(x + width/2, placebo_vals, width, label='Placebo AUUC', color=MODEL_COLORS['Random'])
    
    ax.set_ylabel('AUUC')
    ax.set_title("Placebo Test — AUUC Under Real vs Shuffled Treatment")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#333333', label='Real AUUC (Model Color)'),
                       Patch(facecolor=MODEL_COLORS['Random'], label='Placebo AUUC (Shuffled)')]
    ax.legend(handles=legend_elements)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.6f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved figure to {save_path}")

def plot_cf_confidence_intervals(cate_estimates, lower_bound, upper_bound, save_path='results/figures/09_cf_confidence_intervals.png'):
    """
    Plots a sample of Causal Forest CATE point estimates with 95% confidence intervals.
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    n_sample = min(500, len(cate_estimates))
    np.random.seed(42)
    indices = np.random.choice(len(cate_estimates), n_sample, replace=False)
    
    cates_sample = cate_estimates[indices]
    lb_sample = lower_bound[indices]
    ub_sample = upper_bound[indices]
    
    sort_idx = np.argsort(cates_sample)
    cates_sample = cates_sample[sort_idx]
    lb_sample = lb_sample[sort_idx]
    ub_sample = ub_sample[sort_idx]
    
    x = np.arange(n_sample)
    
    for i in range(n_sample):
        # Color red if statistically significant
        if lb_sample[i] > 0 or ub_sample[i] < 0:
            color = MODEL_COLORS.get('Causal Forest', '#F44336')
        else:
            color = MODEL_COLORS['Random']
            
        plt.errorbar(x[i], cates_sample[i], 
                     yerr=[[cates_sample[i] - lb_sample[i]], [ub_sample[i] - cates_sample[i]]], 
                     color=color, fmt='o', markersize=3, alpha=0.6)
                     
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title("Causal Forest — Individual CATE Estimates with 95% Confidence Intervals")
    plt.xlabel("Sampled Users (Sorted by CATE)")
    plt.ylabel("Estimated CATE")
    plt.text(0.02, 0.95, "Red points: CI excludes zero (statistically significant effect)", 
             transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')
             
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved figure to {save_path}")

if __name__ == '__main__':
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    import src.preprocessing as prep
    from src.metalearners import train_s_learner, train_t_learner, get_cate_estimates
    
    print("Starting Visualization smoke test...")
    data_path = os.path.join(project_root, 'data', 'criteo-research-uplift-v2.1.csv.gz')
    
    try:
        print("Loading and splitting data...")
        df = prep.load_data(data_path, sample_frac=0.1)
        df_train, df_cal, df_test = prep.split_data(df)
        
        X_train, T_train, Y_train = prep.get_XTY(df_train, outcome='conversion')
        X_test, T_test, Y_test = prep.get_XTY(df_test, outcome='conversion')
        
        print("Standardizing features...")
        X_train_scaled, _, X_test_scaled, _ = prep.standardize_features(X_train, X_test, X_test)
        
        print("\n--- Training S-Learner ---")
        s_learner = train_s_learner(X_train_scaled, T_train, Y_train)
        s_cate = get_cate_estimates(s_learner, X_test_scaled)
        
        print("\n--- Training T-Learner ---")
        t_learner = train_t_learner(X_train_scaled, T_train, Y_train)
        t_cate = get_cate_estimates(t_learner, X_test_scaled)
        
        model_cates = {
            'S-Learner': s_cate,
            'T-Learner': t_cate
        }
        
        print("\nGenerating Figures 1 through 5...")
        plot_auuc_comparison(model_cates, T_test, Y_test)
        plot_qini_curves(model_cates, T_test, Y_test)
        plot_cate_calibration(model_cates, T_test, Y_test)
        plot_policy_value(model_cates, T_test, Y_test)
        plot_cate_distributions(model_cates)
        
        print("\nVisualization smoke test completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
