# main.py

import Q4_config
import Q4_data_loader
import Q4_modeling
import Q4_plotting
import Q4_qc_analyzer
import Q4_gc_corrector


def main():
    """Main function to run the entire analysis pipeline."""

    # --- 0. Setup ---
    Q4_plotting.setup_matplotlib(Q4_config.FONT_PREFERENCES)

    # --- 1. Data Loading ---
    df, label_col = Q4_data_loader.load_and_prepare_data(
        Q4_config.FILE_PATH, Q4_config.SHEET_NAME, Q4_config.LABEL_COLUMN_CANDIDATES, Q4_config.NORMAL_LABEL
    )

    # --- 2 & 3. Baseline Model & Optimization ---
    print("\n" + "=" * 60)
    print("--- Step 2 & 3: Baseline Model Optimization ---")
    print("=" * 60)

    results_df, best_contamination_rate = Q4_modeling.optimize_contamination_rate(
        df, Q4_config.ANOMALY_FEATURES_RAW, Q4_config.CONTAMINATION_LEVELS, Q4_config.ISO_FOREST_PARAMS
    )
    print("\nIsolation Forest Parameter Optimization Report (on all data):")
    print(results_df.to_string())

    best_recall_before_correction = results_df['Abnormal_Recall'].max()
    print(f"\nOptimization Complete:")
    print(f"  Best contamination rate for raw features: {best_contamination_rate:.2f}")
    print(f"  This achieves the highest recall (our golden standard): {best_recall_before_correction:.2%}")

    # --- 4. Sequencing Depth QC ---
    print("\n" + "=" * 60)
    print("--- Step 4: Sequencing Depth QC Analysis ---")
    print("=" * 60)

    Q4_plotting.plot_depth_distribution(df, Q4_config.DEPTH_COLUMN, Q4_config.DEPTH_DISTRIBUTION_PLOT_PATH)
    df_filtered = Q4_qc_analyzer.analyze_and_filter_by_depth(
        df, Q4_config.DEPTH_COLUMN, Q4_config.SEQUENCING_DEPTH_THRESHOLD
    )

    # Optional: Re-validate model on high-quality subset
    if not df_filtered.empty and df_filtered['is_abnormal'].nunique() > 1:
        print("\nRe-optimizing model on high-quality (high-depth) subset...")
        filtered_results, _ = Q4_modeling.optimize_contamination_rate(
            df_filtered, Q4_config.ANOMALY_FEATURES_RAW, Q4_config.CONTAMINATION_LEVELS, Q4_config.ISO_FOREST_PARAMS
        )
        print("\nOptimization Report (on high-quality subset):")
        print(filtered_results.to_string())
    else:
        print("\nSkipping model re-validation due to insufficient data after filtering.")

    # --- 5. GC Bias Correction ---
    print("\n" + "=" * 60)
    print("--- Step 5: GC Bias Correction ---")
    print("=" * 60)

    df_corrected = Q4_gc_corrector.diagnose_and_correct_gc_bias(df, Q4_config.GC_CORRECTED_BASE_FEATURES)
    Q4_plotting.plot_gc_correction_effect(
        df_corrected, [f'{b}号染色体的Z值' for b in Q4_config.GC_CORRECTED_BASE_FEATURES], Q4_config.GC_CORRECTION_PLOT_PATH
    )

    # --- 6. Final Validation ---
    print("\n" + "=" * 60)
    print("--- Step 6: Final Validation with Corrected Features ---")
    print("=" * 60)

    print(
        f"\nFinal Model Report (using GC-corrected features and best contamination rate of {best_contamination_rate:.2f}):")
    recall_after_correction = Q4_modeling.evaluate_final_model(
        df_corrected, Q4_config.GC_CORRECTED_FEATURES_FINAL, best_contamination_rate, Q4_config.ISO_FOREST_PARAMS
    )

    # --- 7. Conclusion ---
    print("\n" + "#" * 25 + " Final Conclusion " + "#" * 25)
    print(f"Golden Standard: Best recall with original features was: {best_recall_before_correction:.2%}")
    print(f"Final Result: Recall with GC-corrected features is: {recall_after_correction:.2%}")

    if recall_after_correction > best_recall_before_correction * 1.01:  # Require >1% relative improvement
        improvement = (recall_after_correction - best_recall_before_correction) / best_recall_before_correction
        print(f"\nAssessment: SUCCESS! Linear GC correction improved recall by {improvement:.1%}.")
        print("This suggests that linear GC bias was a key limiting factor.")
    else:
        print(f"\nAssessment: INEFFECTIVE. Linear GC correction did not meaningfully improve recall.")
        print("This suggests the root cause is likely non-linear bias or other factors like low fetal fraction.")
    print("#" * 62)


if __name__ == '__main__':
    main()