# data_loader.py

import pandas as pd
import numpy as np


def _convert_gestational_age(ga):
    """Converts gestational age string (e.g., '12w+2') to a numeric value (e.g., 12.2857)."""
    if pd.isna(ga):
        return np.nan
    try:
        ga_str = str(ga).strip()
        if 'w+' in ga_str:
            weeks, days = ga_str.split('w+')
            return int(weeks) + (int(days) / 7 if days else 0)
        elif 'w' in ga_str:
            return float(ga_str.replace('w', ''))
        return float(ga_str)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert gestational age '{ga}'. It will be treated as missing.")
        return np.nan


def load_and_preprocess_data(file_path, columns):
    """
    Loads data from an Excel file, preprocesses it, and returns a clean DataFrame.

    Args:
        file_path (str): The path to the Excel file.
        columns (list): A list of column names to select.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    print("Step 1: Loading and preprocessing data...")
    df = pd.read_excel(file_path)
    df = df[columns]

    # Convert gestational age and handle missing values
    df['孕周_数值'] = df['检测孕周'].apply(_convert_gestational_age)
    if df['孕周_数值'].isna().any():
        print("Warning: Some gestational ages could not be converted and resulted in missing values.")

    # Remove duplicates
    df = df.drop_duplicates(subset=['孕妇代码', '检测日期'], keep='last')

    print("Data loading and preprocessing complete.\n")
    return df