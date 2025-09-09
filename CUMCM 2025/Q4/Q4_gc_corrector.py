# gc_corrector.py

import pandas as pd
from sklearn.linear_model import LinearRegression


def diagnose_and_correct_gc_bias(df, base_features):
    """
    Diagnoses and corrects for linear GC bias in Z-scores.
    """
    print("\n--- Root Cause Analysis: GC Bias Correction ---")
    correction_summary = []

    df_corrected = df.copy()
    df_normal = df[df['is_abnormal'] == 0]

    z_score_cols_to_correct = [f'{base}号染色体的Z值' for base in base_features]

    for z_col in z_score_cols_to_correct:
        corrected_col_name = f"{z_col.split('的')[0]}_GC_Corrected"

        X_train = df_normal[['GC含量']]
        y_train = df_normal[z_col]

        lr = LinearRegression().fit(X_train, y_train)

        # Apply correction to the entire dataset
        df_corrected[corrected_col_name] = df[z_col] - lr.predict(df[['GC含量']])

        # Quantify the effect
        var_before = y_train.var()
        var_after = df_corrected[df_corrected['is_abnormal'] == 0][corrected_col_name].var()
        reduction = (var_before - var_after) / var_before if var_before > 0 else 0
        correction_summary.append({'Z-Score': z_col, 'Variance_Reduction': f"{reduction:.2%}"})

    print("\nGC Correction Effect (Variance Reduction on Normal Samples):")
    print(pd.DataFrame(correction_summary).to_string())

    return df_corrected