# modeling.py

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


def optimize_contamination_rate(df, features, contamination_levels, model_params, normal_label='Normal',
                                abnormal_label='Abnormal'):
    """
    Systematically finds the best 'contamination' parameter to maximize recall.

    Args:
        df (pd.DataFrame): The dataset to use.
        features (list): List of feature names for the model.
        contamination_levels (list): A list of contamination rates to test.
        model_params (dict): Base parameters for the IsolationForest model.

    Returns:
        tuple: A DataFrame with optimization results, and the best contamination rate found.
    """
    if df.empty or df['is_abnormal'].nunique() < 2:
        print("Warning: Not enough data or classes to perform optimization.")
        return pd.DataFrame(), 0.0

    X = df[features]
    X_normal = df[df['is_abnormal'] == 0][features]
    true_labels = df['is_abnormal'].map({0: normal_label, 1: abnormal_label})

    results = []
    for rate in contamination_levels:
        iso_forest = IsolationForest(contamination=rate, **model_params).fit(X_normal)
        predictions = pd.Series(iso_forest.predict(X)).map({1: normal_label, -1: abnormal_label})
        report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)

        if abnormal_label in report:
            results.append({
                'Contamination_Rate': rate,
                'Abnormal_Recall': report[abnormal_label]['recall'],
                'Abnormal_Precision': report[abnormal_label]['precision'],
                'Abnormal_F1_Score': report[abnormal_label]['f1-score']
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        best_rate = results_df.loc[results_df['Abnormal_Recall'].idxmax()]['Contamination_Rate']
    else:
        best_rate = 0.0

    return results_df, best_rate


def evaluate_final_model(df, features, best_contamination_rate, model_params, normal_label='Normal',
                         abnormal_label='Abnormal'):
    """
    Trains and evaluates the final Isolation Forest model on a given feature set.
    """
    X = df[features]
    X_normal = df[df['is_abnormal'] == 0][features]
    true_labels = df['is_abnormal'].map({0: normal_label, 1: abnormal_label})

    final_model = IsolationForest(contamination=best_contamination_rate, **model_params).fit(X_normal)
    predictions = pd.Series(final_model.predict(X)).map({1: normal_label, -1: abnormal_label})

    print(classification_report(true_labels, predictions, zero_division=0))
    report_dict = classification_report(true_labels, predictions, output_dict=True, zero_division=0)

    return report_dict.get(abnormal_label, {}).get('recall', 0.0)