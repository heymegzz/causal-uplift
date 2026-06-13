import sys
import os
sys.path.insert(0, '.')

import src.preprocessing as prep
import src.propensity as prop

def main():
    print("="*50)
    print(" EXPERIMENT 02: OVERLAP CHECK ")
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
    
    print("Getting propensity scores...")
    e_train = prop.get_propensity_scores(prop_model, X_train_s)
    
    print("Running check_overlap and generating figure...")
    os.makedirs('results/figures', exist_ok=True)
    prop.check_overlap(e_train, T_train, save_path='results/figures/01_propensity_overlap.png')
    
    print("Testing clipping...")
    e_train_clipped = prop.clip_propensity(e_train, lower=0.01, upper=0.99)
    
    print("\n--- Overlap Diagnostics ---")
    print(f"Min propensity: {e_train.min():.6f}")
    print(f"Max propensity: {e_train.max():.6f}")
    print(f"Mean propensity: {e_train.mean():.6f}")
    print(f"Clipped range: [{e_train_clipped.min():.6f}, {e_train_clipped.max():.6f}]")
    
    print("\nOverlap check complete. Figure saved.")
    print("Complete.")

if __name__ == '__main__':
    main()
