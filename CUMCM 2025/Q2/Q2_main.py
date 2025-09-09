# main.py

import pandas as pd
import numpy as np
import Q2_config
import Q2_data_loader
import Q2_bmi_grouper
import Q2_nipt_analyzer
import Q2_plotting


def main():
    """
    Main function to run the entire analysis pipeline.
    """
    # Setup Q2_plotting aesthetics
    Q2_plotting.setup_matplotlib(Q2_config.FONT_PREFERENCES)

    # 1. Load and preprocess data
    try:
        df = Q2_data_loader.load_and_preprocess_data(Q2_config.INPUT_FILE_PATH, Q2_config.SELECTED_COLUMNS)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{Q2_config.INPUT_FILE_PATH}'. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # 2. Group data by BMI
    try:
        df_grouped, breakpoints = Q2_bmi_grouper.group_bmi_data(df, Q2_config.NUM_BMI_GROUPS)
    except ValueError as e:
        print(f"Error during BMI grouping: {e}")
        return

    # 3. Calculate optimal NIPT times
    optimal_times = Q2_nipt_analyzer.calculate_optimal_nipt_times(df_grouped, Q2_config)
    print("\n--- Summary of Optimal NIPT Times ---")
    for i, time in enumerate(optimal_times):
        print(f"Group {i}: {f'{time:.2f} weeks' if not np.isnan(time) else 'Not Determined'}")

    # 4. Generate final plots
    Q2_plotting.plot_bmi_distribution(df_grouped, breakpoints, Q2_config.BMI_DISTRIBUTION_PLOT_PATH)
    Q2_plotting.plot_optimal_nipt_barchart(optimal_times, Q2_config.NIPT_BAR_PLOT_PATH)

    # 5. Save results to Excel
    df_grouped['Optimal_NIPT_Week'] = np.nan
    for group_idx, opt_time in enumerate(optimal_times):
        df_grouped.loc[df_grouped['Group'] == group_idx, 'Optimal_NIPT_Week'] = opt_time

    df_grouped.to_excel(Q2_config.OUTPUT_GROUPED_DATA_PATH, index=False)
    print(f"\nFinal grouped data saved to '{Q2_config.OUTPUT_GROUPED_DATA_PATH}'")
    print("\nAnalysis finished successfully.")


if __name__ == '__main__':
    main()