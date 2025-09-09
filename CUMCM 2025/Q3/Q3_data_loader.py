# data_loader.py

import pandas as pd


def load_and_preprocess_data(file_path):
    """
    Loads data from an Excel file and performs initial preprocessing.

    Args:
        file_path (str): The path to the input Excel file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame, or None if the file is not found.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"\nSuccessfully loaded '{file_path}' with {df.shape[0]} records "
              f"for {len(df['孕妇代码'].unique())} individuals.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path in config.py.")
        return None

    # Basic type conversion and sorting
    df['检测日期'] = pd.to_datetime(df['检测日期'])
    df = df.sort_values(by=['孕妇代码', '检测孕期时间'])

    return df