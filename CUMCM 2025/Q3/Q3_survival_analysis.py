# survival_analysis.py

import pandas as pd
from lifelines import CoxPHFitter
from plotting import plot_correlation_heatmap


def prepare_survival_data(df, y_threshold):
    """Builds the survival dataset (time and status) for each individual."""
    survival_data_list = []
    for name, group in df.groupby('孕妇代码'):
        event_occurred = group[group['Y染色体浓度'] >= y_threshold]
        if not event_occurred.empty:
            status = 1
            time = event_occurred['检测孕期时间'].iloc[0]
        else:
            status = 0
            time = group['检测孕期时间'].iloc[-1]
        survival_data_list.append({'孕妇代码': name, 'time': time, 'status': status})
    return pd.DataFrame(survival_data_list)


def get_baseline_features(df, covariates):
    """Extracts the first recorded features for each individual."""
    baseline_df = df.groupby('孕妇代码').first().reset_index()
    return baseline_df[['孕妇代码'] + covariates]


def fit_cox_model_and_group(df, y_threshold, covariates, n_groups, group_labels):
    """
    Fits the Cox model, predicts risk, and assigns individuals to risk groups.

    Args:
        df (pd.DataFrame): The main preprocessed DataFrame.
        y_threshold (float): The Y-concentration threshold for an 'event'.
        covariates (list): List of baseline features for the model.
        n_groups (int): Number of risk groups to create.
        group_labels (list): Labels for the risk groups.

    Returns:
        pd.DataFrame: A DataFrame containing individual ID, survival data,
                      features, risk score, and risk group.
    """
    # 1. Prepare data
    survival_df = prepare_survival_data(df, y_threshold)
    baseline_df = get_baseline_features(df, covariates)
    cox_data = pd.merge(survival_df, baseline_df, on='孕妇代码')

    # 2. Handle missing values
    print("\nHandling missing values for Cox model...")
    for col in cox_data.select_dtypes(include='number').columns:
        if cox_data[col].isnull().any():
            median_val = cox_data[col].median()
            cox_data[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in '{col}' with median {median_val:.2f}.")

    # 3. Check for collinearity
    print("\nChecking for covariate collinearity...")
    plot_correlation_heatmap(cox_data[covariates])

    # 4. Fit Cox Proportional Hazards model
    cph = CoxPHFitter()
    cph.fit(cox_data.drop('孕妇代码', axis=1), duration_col='time', event_col='status')
    print("\nCox model summary:")
    cph.print_summary()

    # 5. Calculate risk scores and create groups
    cox_data['risk_score'] = cph.predict_partial_hazard(cox_data.drop('孕妇代码', axis=1))
    cox_data['risk_group'] = pd.qcut(cox_data['risk_score'], q=n_groups, labels=group_labels)

    print("\nRisk group distribution:")
    print(cox_data['risk_group'].value_counts())

    return cox_data