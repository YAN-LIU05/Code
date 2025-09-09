# main.py

import pandas as pd
import Q3_config
import Q3_data_loader
import Q3_survival_analysis
import Q3_statistical_tests
import Q3_logistic_modeler
import Q3_decision_analyzer
import Q3_achievement_analyzer
import Q3_plotting


def main():
    """Main function to run the entire analysis pipeline."""
    print("Starting the NIPT analysis pipeline...")

    # Setup Q3_plotting
    Q3_plotting.setup_matplotlib(Q3_config.FONT_PREFERENCES)

    # Section 1: Data Loading
    df = Q3_data_loader.load_and_preprocess_data(Q3_config.INPUT_FILE)
    if df is None:
        return  # Stop if data loading failed

    # Section 2: Cox Model & Risk Grouping
    cox_data = Q3_survival_analysis.fit_cox_model_and_group(
        df, Q3_config.Y_THRESHOLD, Q3_config.BASELINE_COVARIATES, Q3_config.N_GROUPS, Q3_config.GROUP_LABELS
    )

    # Merge group info back to original data for trajectory plots
    merged_data = pd.merge(df, cox_data[['孕妇代码', 'risk_group']], on='孕妇代码', how='left')

    # Section 3: Visualization & Validation
    Q3_plotting.plot_km_validation(cox_data)
    Q3_plotting.plot_concentration_trajectory(merged_data, Q3_config.Y_THRESHOLD, Q3_config.GROUP_LABELS)

    # Section 4: Statistical Testing (LMM)
    Q3_statistical_tests.run_lmm_analysis(merged_data, Q3_config.GROUP_LABELS)

    # Section 5: Logistic Growth Modeling
    logistic_params_df = Q3_logistic_modeler.fit_logistic_model(
        merged_data, Q3_config.MIN_DATA_POINTS_FOR_FIT, Q3_config.LOGISTIC_BOUNDS
    )
    # Merge logistic params with cox data for further analysis
    logistic_analysis_df = pd.merge(cox_data, logistic_params_df, on='孕妇代码')

    # Section 6: Logistic Parameter Analysis & Visualization
    Q3_plotting.plot_logistic_parameter_distributions(logistic_analysis_df)
    Q3_statistical_tests.run_anova_on_logistic_params(logistic_analysis_df)

    # Section 7: Decision & Sensitivity Analysis
    decision_results = Q3_decision_analyzer.run_decision_analysis(logistic_analysis_df, Q3_config)
    Q3_plotting.plot_decision_model_results(decision_results, Q3_config)

    sensitivity_df = Q3_decision_analyzer.run_sensitivity_analysis(logistic_analysis_df, Q3_config)
    Q3_plotting.plot_sensitivity_analysis(sensitivity_df)

    # Section 8: Achievement Rate Analysis
    kmf_objects, achievement_results_df = Q3_achievement_analyzer.analyze_achievement_rates(cox_data, Q3_config)
    Q3_plotting.plot_achievement_timeline(kmf_objects, achievement_results_df)

    # Section 9: Save Final Output
    final_output_df = pd.merge(df, cox_data[['孕妇代码', 'risk_group', 'risk_score']], on='孕妇代码', how='left')
    try:
        final_output_df.to_excel(Q3_config.OUTPUT_FILE, index=False)
        print(f"\nSuccessfully saved final results with group assignments to '{Q3_config.OUTPUT_FILE}'.")
    except Exception as e:
        print(f"\nError saving output file: {e}")

    print("\nAnalysis pipeline finished successfully.")


if __name__ == '__main__':
    main()