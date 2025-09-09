# qc_analyzer.py

def analyze_and_filter_by_depth(df, depth_col, threshold):
    """
    Analyzes sequencing depth, prints stats, and filters the DataFrame.
    """
    print("\n--- Sequencing Depth QC Analysis ---")
    print("\nSequencing Depth Descriptive Statistics:")
    print(df.groupby('is_abnormal')[depth_col].describe())

    df_filtered = df[df[depth_col] >= threshold].copy()
    print("\n" + "-" * 50)
    print(f"Filtering data with depth threshold >= {threshold:.0f}")
    print(f"Original sample count: {len(df)}")
    print(f"Filtered sample count: {len(df_filtered)} ({len(df_filtered) / len(df):.1%} retained)")

    if len(df_filtered[df_filtered['is_abnormal'] == 1]) == 0:
        print("Warning: No 'Abnormal' samples remain after filtering!")
    print("-" * 50)

    return df_filtered