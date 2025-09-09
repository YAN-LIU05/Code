# plotting.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import probplot


def setup_matplotlib(font_prefs):
    """Sets up Matplotlib parameters for font and unicode support."""
    plt.rcParams['font.sans-serif'] = font_prefs
    plt.rcParams['axes.unicode_minus'] = False


def plot_model_evaluation(actual_y, predicted_y, group_index, scatter_path, qq_path):
    """
    Generates and saves scatter and Q-Q plots for model evaluation.

    Args:
        actual_y (list): The actual target values.
        predicted_y (list): The predicted values from the model.
        group_index (int): The index of the current group.
        scatter_path (str): The file path template for the scatter plot.
        qq_path (str): The file path template for the Q-Q plot.
    """
    # Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_y, predicted_y, alpha=0.5, label='Predicted vs Actual')
    plt.plot([min(actual_y), max(actual_y)], [min(actual_y), max(actual_y)], 'r--', label='Ideal Line')
    plt.xlabel('Actual Y-chromosome Concentration')
    plt.ylabel('Predicted Y-chromosome Concentration')
    plt.title(f'Group {group_index} - Predicted vs Actual Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig(scatter_path.format(group=group_index))
    plt.close()

    # Q-Q Plot for residuals
    residuals = np.array(actual_y) - np.array(predicted_y)
    plt.figure(figsize=(8, 6))
    probplot(residuals, dist="norm", plot=plt)
    plt.title(f'Group {group_index} - Residuals Q-Q Plot')
    plt.savefig(qq_path.format(group=group_index))
    plt.close()


def plot_bmi_distribution(df, breakpoints, save_path):
    """
    Plots the BMI distribution histogram with group boundaries.

    Args:
        df (pd.DataFrame): DataFrame containing BMI and Group data.
        breakpoints (list): A list of BMI breakpoints.
        save_path (str): The file path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    num_groups = df['Group'].nunique()
    for i in range(num_groups):
        group_data = df[df['Group'] == i]['孕妇BMI']
        plt.hist(group_data, bins=30, alpha=0.5, label=f'Group {i}', density=True)

    for bp in breakpoints[:-1]:
        plt.axvline(bp, color='red', linestyle='--', label='Breakpoint' if bp == breakpoints[0] else "")

    plt.title('BMI Group Distribution', fontsize=14)
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 5)  # Adjust if necessary
    plt.savefig(save_path)
    plt.close()
    print(f"BMI distribution plot saved to '{save_path}'")


def plot_optimal_nipt_barchart(optimal_times, save_path):
    """
    Creates and saves a bar chart of the optimal NIPT times for each group.

    Args:
        optimal_times (list): A list of optimal times, may contain NaNs.
        save_path (str): The file path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    valid_groups = [i for i, t in enumerate(optimal_times) if not np.isnan(t)]
    valid_times = [t for t in optimal_times if not np.isnan(t)]

    if not valid_times:
        print("No valid optimal NIPT times to plot.")
        return

    plt.bar(valid_groups, valid_times, color='skyblue', label='Optimal NIPT Time')
    plt.xlabel('Group')
    plt.ylabel('Gestational Week')
    plt.title('Optimal NIPT Time for Each BMI Group')
    plt.legend()
    plt.grid(True, axis='y')
    plt.xticks(valid_groups)
    plt.savefig(save_path)
    plt.close()
    print(f"Optimal NIPT times bar chart saved to '{save_path}'")