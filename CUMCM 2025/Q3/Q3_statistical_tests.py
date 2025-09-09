# statistical_tests.py

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import f_oneway


def run_lmm_analysis(merged_data, group_labels):
    """
    Performs a Linear Mixed-Effects Model analysis to test for group differences.
    """
    print("\n--- Running Linear Mixed-Effects Model (LMM) Test ---")
    lmm_data = merged_data.copy()

    # Clean data specifically for LMM
    required_cols = ['Y染色体浓度', '检测孕期时间', 'risk_group', '孕妇代码']
    lmm_data.dropna(subset=required_cols, inplace=True)

    if len(lmm_data) < 10:
        print("Error: Not enough data to run LMM after cleaning.")
        return

    # Set reference group
    lmm_data['risk_group'] = pd.Categorical(lmm_data['risk_group'], categories=group_labels, ordered=False)

    try:
        model_formula = "Q('Y染色体浓度') ~ Q('检测孕期时间') * C(risk_group)"
        lmm = smf.mixedlm(model_formula, data=lmm_data, groups=lmm_data["孕妇代码"])
        lmm_result = lmm.fit()
        print("\nLMM Results Summary:")
        print(lmm_result.summary())

        # Interpret results
        interaction_p_values = lmm_result.pvalues.filter(like="Q('检测孕期时间'):C(risk_group)")
        if any(interaction_p_values < 0.05):
            print("\n[LMM Conclusion]: Significant interaction effect found. The groups show "
                  "statistically different Y-concentration growth rates.")
        else:
            print("\n[LMM Conclusion]: No significant interaction effect. The groups' growth rates "
                  "are not statistically different.")

    except Exception as e:
        print(f"\nAn error occurred during LMM fitting: {e}")


def run_anova_on_logistic_params(analysis_df):
    """
    Runs ANOVA tests on the fitted logistic parameters across risk groups.
    """
    print("\n--- Running ANOVA on Logistic Model Parameters ---")
    param_names = {'t0_fit': 'Midpoint Time (t0)', 'k_fit': 'Growth Rate (k)', 'L_fit': 'Saturation Limit (L)'}

    for param, title in param_names.items():
        if param in analysis_df.columns:
            groups = [group[param].dropna() for _, group in analysis_df.groupby('risk_group')]
            if len(groups) > 1 and all(len(g) > 1 for g in groups):
                f_stat, p_value = f_oneway(*groups)
                print(f"\nANOVA for '{title}': F-statistic={f_stat:.3f}, p-value={p_value:.4f}")
                if p_value < 0.05:
                    print(f"[Conclusion]: Significant differences found across groups for '{title}'.")
                else:
                    print(f"[Conclusion]: No significant differences found for '{title}'.")