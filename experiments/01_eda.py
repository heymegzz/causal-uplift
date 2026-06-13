import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import src.preprocessing as prep

def main():
    print("="*50)
    print(" EXPERIMENT 01: EXPLORATORY DATA ANALYSIS ")
    print("="*50)
    
    data_path = 'data/criteo-research-uplift-v2.1.csv.gz'
    print(f"Loading data from {data_path} (sample_frac=0.05)...\n")
    df = prep.load_data(data_path, sample_frac=0.05, random_state=42)
    
    print("--- Basic Info ---")
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing value counts:\n{df.isnull().sum()}")
    
    print("\n--- Key Rates ---")
    treatment_rate = df['treatment'].mean()
    visit_rate = df['visit'].mean()
    conversion_rate = df['conversion'].mean()
    print(f"Treatment rate:  {treatment_rate:.4f}")
    print(f"Visit rate:      {visit_rate:.4f}")
    print(f"Conversion rate: {conversion_rate:.4f}")
    
    print("\n--- Group Sizes ---")
    treatment_size = df['treatment'].sum()
    control_size = len(df) - treatment_size
    print(f"Treatment group size: {treatment_size}")
    print(f"Control group size:   {control_size}")
    
    print("\n--- Feature Statistics (f0-f11) ---")
    features = [f"f{i}" for i in range(12)]
    print(df[features].agg(['mean', 'std']).T)
    
    print("\n--- Conversion by Treatment Arm ---")
    conv_t1 = df[df['treatment'] == 1]['conversion'].mean()
    conv_t0 = df[df['treatment'] == 0]['conversion'].mean()
    print(f"Conversion rate (T=1): {conv_t1:.6f}")
    print(f"Conversion rate (T=0): {conv_t0:.6f}")
    
    print("\n--- Raw ATE ---")
    raw_ate = conv_t1 - conv_t0
    print(f"Raw ATE (mean(Y|T=1) - mean(Y|T=0)): {raw_ate:.6f}")
    
    print("\nComplete.")

if __name__ == '__main__':
    main()
