# config.py

# --- File and Data Settings ---
FILE_PATH = 'data_cleaned2.xlsx'
SHEET_NAME = '女胎数据_已清洗'
# Dynamically determine the label column name during data loading
LABEL_COLUMN_CANDIDATES = ['染色体的非整倍体', '染色体的非倍性']
NORMAL_LABEL = 'Normal'

# --- Feature Engineering ---
# Features used for the anomaly detection model
ANOMALY_FEATURES_RAW = [
    '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
    'Z21_minus_Z18', 'Z18_minus_Z13'
]

# Features after GC correction (names are generated programmatically)
GC_CORRECTED_BASE_FEATURES = ['13号染色体', '18号染色体', '21号染色体']
# We will still use the original X Z-score as it's not corrected
GC_CORRECTED_FEATURES_FINAL = [f'{base}_GC_Corrected' for base in GC_CORRECTED_BASE_FEATURES] + ['X染色体的Z值']


# --- Model Parameters ---
# Contamination levels to test for optimization
CONTAMINATION_LEVELS = [0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# Isolation Forest model parameters
ISO_FOREST_PARAMS = {'random_state': 42, 'n_estimators': 150}

# --- Quality Control (QC) Settings ---
# Column for sequencing depth analysis
DEPTH_COLUMN = '唯一比对的读段数'
# !!! KEY PARAMETER: Set this threshold based on the distribution plot !!!
SEQUENCING_DEPTH_THRESHOLD = 2e6

# --- Plotting Settings ---
# Matplotlib font for Chinese characters
FONT_PREFERENCES = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# Output filenames for plots
DEPTH_DISTRIBUTION_PLOT_PATH = 'sequencing_depth_distribution.png'
GC_CORRECTION_PLOT_PATH = 'gc_correction_evaluation.png'