# config.py

# =============================================================================
# Configuration File for NIPT Analysis
# =============================================================================

# --- File I/O ---
INPUT_FILE = 'cleaned_data_origin.xlsx'
OUTPUT_FILE = 'result_with_groups.xlsx'

# --- Core Survival Analysis Parameters ---
Y_THRESHOLD = 0.06  # Y-chromosome concentration "achievement" threshold
BASELINE_COVARIATES = [
    '年龄', '孕妇BMI', 'IVF妊娠_数值', '怀孕次数_数值', '生产次数'
]
N_GROUPS = 3  # Number of risk groups to create
GROUP_LABELS = ['低风险组(慢)', '中风险组', '高风险组(快)']

# --- Logistic Growth Model Parameters ---
LOGISTIC_BOUNDS = (
    [0.01, 0.01, 5],   # Lower bounds for [L, k, t0]
    [2.0, 2.0, 35]     # Upper bounds for [L, k, t0] -> Note: L bound is relative to data max
)
MIN_DATA_POINTS_FOR_FIT = 3 # Minimum data points required for a logistic fit

# --- Decision Analysis Model Parameters ---
# Risk component weights
WEIGHT_FN = 0.5  # Weight for False-Negative risk
WEIGHT_OP = 0.3  # Weight for Operational/Surgical risk
WEIGHT_TIME = 0.2  # Weight for Time cost

# False-Negative risk model: R_fn(t) = exp(-alpha * y(t))
FN_RISK_SENSITIVITY = 15.0  # Parameter alpha

# Operational risk model: R_op(t) = A * exp(B * (t - T_start))
OP_RISK_BASE = 1.0          # Base risk (A)
OP_RISK_EXP_RATE = 0.1      # Base exponential growth rate (B)
OP_RISK_MULTIPLIERS = {     # Group-specific multipliers for the growth rate
    '低风险组(慢)': 1.2,
    '中风险组': 1.0,
    '高风险组(快)': 0.9
}

# Time range for analysis
T_START = 12
T_MAX = 40

# --- Sensitivity Analysis ---
SENSITIVITY_WEIGHT_VALUES = 20  # Number of weight scenarios to test

# --- Achievement Rate Analysis ---
TARGET_ACHIEVEMENT_RATES = [0.80, 0.90, 0.95]

# --- Plotting & Display ---
# Ensure you have a font that supports Chinese characters
FONT_PREFERENCES = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']