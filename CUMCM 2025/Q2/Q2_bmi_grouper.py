# bmi_grouper.py

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def group_bmi_data(df, num_groups):
    """
    Groups the data into a specified number of clusters based on BMI.

    Args:
        df (pd.DataFrame): The input DataFrame containing '孕妇BMI'.
        num_groups (int): The number of groups to create.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame with an added 'Group' column.
            - list: A list of BMI breakpoints defining the upper limit of each group.
    """
    print(f"Step 2: Grouping BMI data into {num_groups} clusters...")

    # Prepare BMI data
    bmi_data = df['孕妇BMI'].dropna()
    if len(bmi_data) == 0:
        raise ValueError("No valid BMI data available for grouping.")

    bmi_data_sorted = np.sort(bmi_data).reshape(-1, 1)

    # Standardize and fit GMM
    scaler = StandardScaler()
    bmi_scaled = scaler.fit_transform(bmi_data_sorted)
    gmm = GaussianMixture(n_components=num_groups, random_state=42)
    labels = gmm.fit_predict(bmi_scaled)

    # Order group labels by BMI mean
    sorted_indices = np.argsort(gmm.means_.flatten())
    label_mapping = {old: new for new, old in enumerate(sorted_indices)}
    ordered_labels = np.array([label_mapping[label] for label in labels])

    # Assign labels to the original DataFrame
    df_filtered = df[df['孕妇BMI'].notna()].copy().sort_values('孕妇BMI').reset_index(drop=True)
    df_filtered['Group'] = ordered_labels

    # Determine breakpoints
    breakpoints = []
    for i in range(num_groups):
        max_value = df_filtered[df_filtered['Group'] == i]['孕妇BMI'].max()
        breakpoints.append(max_value)

    print("BMI grouping complete.")
    group_stats = df_filtered.groupby('Group')['孕妇BMI'].agg(['count', 'mean', 'std', 'min', 'max'])
    print("Group Statistics:\n", group_stats)
    print("\nGroup Breakpoints (Max BMI per group):", [round(bp, 4) for bp in breakpoints])
    print("-" * 30)

    return df_filtered, breakpoints