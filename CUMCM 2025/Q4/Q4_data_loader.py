# data_loader.py

import pandas as pd
import sys

def _transform_and_explode_data(df, label_col):
    """
    Handles cases where a single sample has multiple abnormality labels (e.g., T21, T18).
    It creates a separate row for each label.
    """
    new_rows = []
    known_abnormalities = ['T13', 'T18', 'T21']
    for _, row in df.iterrows():
        label = str(row[label_col])
        found_labels = [ab for ab in known_abnormalities if ab in label]
        if len(found_labels) > 1:
            for single_label in found_labels:
                new_row = row.copy()
                new_row[label_col] = single_label
                new_rows.append(new_row)
        else:
            new_rows.append(row)
    return pd.DataFrame(new_rows).reset_index(drop=True)

def load_and_prepare_data(file_path, sheet_name, label_candidates, normal_label):
    """
    Loads data from an Excel file and performs all initial preprocessing.

    Returns:
        pd.DataFrame: A DataFrame ready for analysis.
    """
    try:
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Fatal Error: The file '{file_path}' was not found.")
        sys.exit()
    except Exception as e:
        print(f"Fatal Error: Could not read the Excel file. Reason: {e}")
        sys.exit()

    # Find the correct label column name
    label_col = next((col for col in label_candidates if col in df_raw.columns), None)
    if not label_col:
        print(f"Fatal Error: Could not find any of the target label columns: {label_candidates}")
        sys.exit()

    df_raw[label_col] = df_raw[label_col].fillna(normal_label)
    df = _transform_and_explode_data(df_raw, label_col)

    # Feature Engineering
    df['is_abnormal'] = df[label_col].apply(lambda x: 0 if x == normal_label else 1)
    df['Z21_minus_Z18'] = df['21号染色体的Z值'] - df['18号染色体的Z值']
    df['Z18_minus_Z13'] = df['18号染色体的Z值'] - df['13号染色体的Z值']

    print(f"Successfully loaded and processed {len(df)} rows from '{file_path}'.")
    return df, label_col