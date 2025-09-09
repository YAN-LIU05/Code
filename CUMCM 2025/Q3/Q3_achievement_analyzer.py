# achievement_analyzer.py

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter


def analyze_achievement_rates(cox_data, config):
    """
    Calculates and visualizes key time points based on population achievement rates.

    Args:
        cox_data (pd.DataFrame): DataFrame with survival and group info.
        config (module): The configuration module.

    Returns:
        tuple: A dictionary of KaplanMeierFitter objects and a DataFrame of results.
    """
    print("\n--- Analyzing Key Time Nodes Based on Achievement Rates ---")
    results = []
    kmf_objects = {}

    for group_name, grouped_df in cox_data.groupby('risk_group'):
        # Fit KMF to get the curve
        kmf = KaplanMeierFitter()
        kmf.fit(grouped_df['time'], event_observed=grouped_df['status'], label=group_name)
        kmf_objects[group_name] = kmf

        # Analyze quantiles of actual achievers
        achievers_df = grouped_df[grouped_df['status'] == 1]
        if achievers_df.empty:
            continue

        for rate in config.TARGET_ACHIEVEMENT_RATES:
            time_at_quantile = achievers_df['time'].quantile(rate)
            # Find the corresponding survival probability on the KM curve for plotting
            prob_on_km = kmf.predict(time_at_quantile)

            print(f"For {group_name}, {int(rate * 100)}% of achievers reach target by: "
                  f"{time_at_quantile:.1f} weeks")

            results.append({
                'group': group_name,
                'achievement_rate': rate,
                'time_weeks': time_at_quantile,
                'prob_on_km': prob_on_km
            })

    return kmf_objects, pd.DataFrame(results)