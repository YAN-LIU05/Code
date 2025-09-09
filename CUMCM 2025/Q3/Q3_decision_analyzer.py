# decision_analyzer.py

import numpy as np
import pandas as pd
from logistic_modeler import logistic_function


def _calculate_risks(t, params, group_name, config):
    """Helper function to calculate the three risk components."""
    L, k, t0 = params['L_fit'], params['k_fit'], params['t0_fit']

    # 1. False-Negative Risk
    y_t = logistic_function(t, L, k, t0)
    risk_fn = np.exp(-config.FN_RISK_SENSITIVITY * y_t)

    # 2. Time Cost Risk
    risk_time = t - config.T_START
    risk_time[risk_time < 0] = 0

    # 3. Operational Risk
    multiplier = config.OP_RISK_MULTIPLIERS.get(group_name, 1.0)
    rate = config.OP_RISK_EXP_RATE * multiplier
    risk_op = config.OP_RISK_BASE * np.exp(rate * (t - config.T_START))

    return risk_fn, risk_time, risk_op


def run_decision_analysis(logistic_params_df, config):
    """
    Calculates and returns the optimal testing time based on a weighted risk model.
    """
    print("\n--- Running Final Decision Analysis Model ---")
    time_range = np.linspace(config.T_START, config.T_MAX, 300)
    decision_results = {}

    group_param_summary = logistic_params_df.groupby('risk_group')[['L_fit', 'k_fit', 't0_fit']].mean()

    for group_name, params in group_param_summary.iterrows():
        risk_fn, risk_time, risk_op = _calculate_risks(time_range, params, group_name, config)

        # Normalize risks (except operational, which is already scaled)
        risk_fn_norm = (risk_fn - risk_fn.min()) / (risk_fn.max() - risk_fn.min())
        risk_time_norm = (risk_time - risk_time.min()) / (risk_time.max() - risk_time.min())

        total_risk = (config.WEIGHT_FN * risk_fn_norm +
                      config.WEIGHT_TIME * risk_time_norm +
                      config.WEIGHT_OP * risk_op)

        optimal_idx = np.argmin(total_risk)
        optimal_time = time_range[optimal_idx]

        print(f"Optimal testing time for {group_name}: {optimal_time:.1f} weeks")

        decision_results[group_name] = {
            'time_range': time_range,
            'params': params,
            'risk_fn_norm': risk_fn_norm,
            'risk_time_norm': risk_time_norm,
            'risk_op': risk_op,
            'total_risk': total_risk,
            'optimal_time': optimal_time
        }

    return decision_results


def run_sensitivity_analysis(logistic_params_df, config):
    """
    Analyzes how the optimal time changes with varying risk weights.
    """
    print("\n--- Running Weight Sensitivity Analysis ---")
    sensitivity_results = []
    time_range = np.linspace(config.T_START, config.T_MAX, 300)
    group_param_summary = logistic_params_df.groupby('risk_group')[['L_fit', 'k_fit', 't0_fit']].mean()

    # Determine ratio for remaining weights
    op_time_sum = config.WEIGHT_OP + config.WEIGHT_TIME
    ratio_op = config.WEIGHT_OP / op_time_sum if op_time_sum > 0 else 0.5

    for w_fn in np.linspace(0.1, 0.8, config.SENSITIVITY_WEIGHT_VALUES):
        w_rem = 1.0 - w_fn
        w_op = w_rem * ratio_op
        w_time = w_rem * (1 - ratio_op)

        for group_name, params in group_param_summary.iterrows():
            risk_fn, risk_time, risk_op = _calculate_risks(time_range, params, group_name, config)
            risk_fn_norm = (risk_fn - risk_fn.min()) / (risk_fn.max() - risk_fn.min())
            risk_time_norm = (risk_time - risk_time.min()) / (risk_time.max() - risk_time.min())

            total_risk = w_fn * risk_fn_norm + w_time * risk_time_norm + w_op * risk_op
            optimal_time = time_range[np.argmin(total_risk)]

            sensitivity_results.append({
                'weight_fn': w_fn, 'group': group_name, 'optimal_time': optimal_time
            })

    return pd.DataFrame(sensitivity_results)