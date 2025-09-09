# logistic_modeler.py

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def logistic_function(t, L, k, t0):
    """3-parameter logistic growth function."""
    try:
        return L / (1 + np.exp(-k * (t - t0)))
    except OverflowError:
        return np.inf


def fit_logistic_model(merged_data, min_points, bounds):
    """
    Fits a logistic growth model to each individual's time-series data.

    Args:
        merged_data (pd.DataFrame): DataFrame with time-series data and group info.
        min_points (int): The minimum number of data points required to attempt a fit.
        bounds (tuple): Parameter bounds for the curve fitting.

    Returns:
        pd.DataFrame: A DataFrame with fitted logistic parameters for each individual.
    """
    print("\n--- Fitting Logistic Growth Model for each individual ---")
    logistic_params_list = []
    for name, group in merged_data.groupby('孕妇代码'):
        if len(group) < min_points:
            continue

        t_data = group['检测孕期时间'].values
        y_data = group['Y染色体浓度'].values

        try:
            # Dynamic bounds for L, based on observed data
            l_min_bound = y_data.max()
            l_max_bound = y_data.max() * 2 if y_data.max() > 0 else 1.0

            # Initial guesses
            p0 = [np.median(y_data) * 2, 0.5, np.median(t_data)]

            params, _ = curve_fit(
                logistic_function, t_data, y_data, p0=p0,
                bounds=([l_min_bound, bounds[0][1], bounds[0][2]],
                        [l_max_bound, bounds[1][1], bounds[1][2]]),
                maxfev=5000
            )
            logistic_params_list.append({'孕妇代码': name, 'L_fit': params[0], 'k_fit': params[1], 't0_fit': params[2]})
        except (RuntimeError, ValueError):
            pass  # Skip individuals where fitting fails

    logistic_params_df = pd.DataFrame(logistic_params_list)
    print(f"Successfully fitted logistic models for {len(logistic_params_df)} individuals.")
    return logistic_params_df