# nipt_analyzer.py

import numpy as np
from scipy.stats import norm, linregress
from scipy.interpolate import interp1d
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from plotting import plot_model_evaluation


def _predict_with_linear_regression(weeks, y_concentrations, target_y, z_score):
    """Uses linear regression to find the optimal week."""
    slope, intercept, r_value, p_value, std_err = linregress(weeks, y_concentrations)
    if slope <= 0:
        return None, r_value ** 2, slope  # Return None if trend is not increasing

    target_week = (target_y - intercept) / slope
    n = len(weeks)
    mean_weeks = np.mean(weeks)
    ss_weeks = np.sum((weeks - mean_weeks) ** 2)
    se_pred = std_err * np.sqrt(1 / n + (target_week - mean_weeks) ** 2 / ss_weeks)
    ci_lower = target_week - z_score * se_pred / slope

    # Return the lower bound of the CI if it's within the observed range, else return the point estimate
    if min(weeks) <= ci_lower <= max(weeks):
        return ci_lower, r_value ** 2, slope
    elif min(weeks) <= target_week <= max(weeks):
        return target_week, r_value ** 2, slope
    return None, r_value ** 2, slope


def _predict_with_gbr(weeks, y_concentrations, target_y):
    """Uses Gradient Boosting Regressor and bootstrapping to find the optimal week."""
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbr.fit(weeks.reshape(-1, 1), y_concentrations)

    # Predict on a fine grid to find where the target is met
    week_range = np.linspace(min(weeks), max(weeks), 100).reshape(-1, 1)
    y_pred = gbr.predict(week_range)

    # Use bootstrapping to estimate the 80% confidence interval lower bound
    bootstrap_times = []
    for _ in range(100):
        indices = np.random.choice(len(weeks), len(weeks), replace=True)
        gbr.fit(weeks[indices].reshape(-1, 1), y_concentrations[indices])
        boot_pred = gbr.predict(week_range)
        valid_indices = np.where(boot_pred >= target_y)[0]
        if len(valid_indices) > 0:
            bootstrap_times.append(week_range[valid_indices[0]][0])

    if bootstrap_times:
        ci_lower = np.percentile(bootstrap_times, 10)  # 10th percentile for 80% CI lower bound
        if min(weeks) <= ci_lower <= max(weeks):
            return ci_lower

    # Fallback to the point estimate if CI is out of range
    valid_indices = np.where(y_pred >= target_y)[0]
    if len(valid_indices) > 0:
        target_week = week_range[valid_indices[0]][0]
        if min(weeks) <= target_week <= max(weeks):
            return target_week

    return None


def calculate_optimal_nipt_times(df, config):
    """
    Calculates the optimal NIPT time for each BMI group.

    Args:
        df (pd.DataFrame): The grouped DataFrame.
        config (module): The configuration module.

    Returns:
        list: A list of optimal NIPT times for each group.
    """
    print("\nStep 3: Calculating optimal NIPT times for each group...")
    optimal_nipt_times = []
    z_score = norm.ppf((1 + config.CONFIDENCE_LEVEL) / 2)

    for group in range(config.NUM_BMI_GROUPS):
        group_data = df[df['Group'] == group][['孕妇代码', '孕周_数值', 'Y染色体浓度']].dropna()
        if len(group_data) < 2:
            optimal_nipt_times.append(np.nan)
            continue

        group_optimal_times = []
        all_actual_y, all_predicted_y = [], []
        print(f"\n--- Processing Group {group} ---")

        for code in group_data['孕妇代码'].unique():
            woman_data = group_data[group_data['孕妇代码'] == code].sort_values('孕周_数值')
            if len(woman_data) < 2: continue

            weeks = woman_data['孕周_数值'].values
            y_concs = woman_data['Y染色体浓度'].values

            # First, try linear regression to get R²
            opt_time, r_squared, slope = _predict_with_linear_regression(weeks, y_concs, config.TARGET_Y_CONCENTRATION,
                                                                         z_score)

            # Decide model based on R² and slope
            if r_squared < config.R2_THRESHOLD and slope > 0:
                # Switch to GBR
                print(f"  Woman {code}: R²({r_squared:.3f}) < {config.R2_THRESHOLD}. Switching to GBR.")
                gbr_time = _predict_with_gbr(weeks, y_concs, config.TARGET_Y_CONCENTRATION)
                if gbr_time: group_optimal_times.append(gbr_time)
                # For plotting
                gbr = GradientBoostingRegressor().fit(weeks.reshape(-1, 1), y_concs)
                all_predicted_y.extend(gbr.predict(weeks.reshape(-1, 1)))
            else:
                # Use linear regression result
                if slope > 0:
                    print(f"  Woman {code}: R²({r_squared:.3f}) >= {config.R2_THRESHOLD}. Using Linear Regression.")
                    if opt_time: group_optimal_times.append(opt_time)
                    all_predicted_y.extend(slope * weeks + intercept)
                else:
                    print(f"  Woman {code}: Slope ({slope:.3f}) is not positive. Skipping.")

            all_actual_y.extend(y_concs)

        # Group-level model evaluation and plotting
        if all_actual_y and all_predicted_y:
            r2 = r2_score(all_actual_y, all_predicted_y)
            print(f"Group {group} Model Evaluation - R²: {r2:.4f}")
            plot_model_evaluation(all_actual_y, all_predicted_y, group, config.SCATTER_PLOT_PATH, config.QQ_PLOT_PATH)

        # Calculate the median optimal time for the group
        if group_optimal_times:
            median_time = np.median(group_optimal_times)
            optimal_nipt_times.append(median_time)
            print(f"Group {group} Optimal NIPT Time (Median): {median_time:.2f} weeks")
        else:
            optimal_nipt_times.append(np.nan)
            print(f"Group {group}: Could not determine an optimal NIPT time.")

    print("\nNIPT time calculation complete.")
    return optimal_nipt_times