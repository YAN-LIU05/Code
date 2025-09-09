# config.py

# --- File Paths ---
INPUT_FILE_PATH = '../data.xlsx'
OUTPUT_GROUPED_DATA_PATH = 'bmi_grouped_data_with_nipt.xlsx'

# --- Data Columns ---
SELECTED_COLUMNS = ['孕妇代码', '检测日期', '检测孕周', '孕妇BMI', 'Y染色体浓度']

# --- Model Parameters ---
NUM_BMI_GROUPS = 5
TARGET_Y_CONCENTRATION = 0.04  # Target Y-chromosome concentration (4%)
CONFIDENCE_LEVEL = 0.80      # 80% confidence level for interval calculation
R2_THRESHOLD = 0.5           # R² threshold to switch from Linear Regression to GBR

# --- Plotting Settings ---
SCATTER_PLOT_PATH = 'group_{group}_scatter_plot.png'
QQ_PLOT_PATH = 'group_{group}_qq_plot.png'
NIPT_BAR_PLOT_PATH = 'optimal_nipt_times_bar_plot.png'
BMI_DISTRIBUTION_PLOT_PATH = 'bmi_grouped_distribution.png'

# --- Matplotlib Settings ---
# Ensure you have a font that supports Chinese characters installed
FONT_PREFERENCES = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']