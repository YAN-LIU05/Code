# plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


def setup_matplotlib(font_prefs):
    """Sets up Matplotlib parameters for font and unicode support."""
    try:
        plt.rcParams['font.sans-serif'] = font_prefs
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"Warning: Failed to set Chinese font: {e}.")


def plot_depth_distribution(df, depth_col, save_path):
    """Generates and saves a violin plot of sequencing depth distribution."""
    plt.figure(figsize=(10, 7))
    sns.violinplot(data=df, x='is_abnormal', y=depth_col, hue='is_abnormal', palette=['#377eb8', '#e41a1c'],
                   legend=False)
    plt.title('Distribution of Effective Sequencing Depth (Normal vs Abnormal)', fontsize=16)
    plt.gca().set_xticks([0, 1])
    plt.gca().set_xticklabels(['Normal', 'Abnormal'])
    plt.ylabel('Unique Mapped Reads (Effective Depth)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sequencing depth distribution plot saved to '{save_path}'")


def plot_gc_correction_effect(df, z_score_cols, save_path):
    """Visualizes the effect of GC correction on Z-scores."""
    fig, axes = plt.subplots(len(z_score_cols), 2, figsize=(12, len(z_score_cols) * 5))
    fig.suptitle('GC Bias Correction (Based on Normal Samples)', fontsize=20, y=1.02)

    df_normal = df[df['is_abnormal'] == 0]

    for i, z_col in enumerate(z_score_cols):
        corrected_col = f"{z_col.split('的')[0]}_GC_Corrected"

        # Before correction plot
        sns.regplot(data=df_normal, x='GC含量', y=z_col, ax=axes[i, 0],
                    line_kws={'color': 'red'}, scatter_kws={'alpha': 0.3})
        axes[i, 0].set_title(f'Before Correction: {z_col}')

        # After correction plot
        sns.regplot(data=df_normal, x='GC含量', y=corrected_col, ax=axes[i, 1],
                    line_kws={'color': 'red'}, scatter_kws={'alpha': 0.3})
        axes[i, 1].set_title(f'After Correction: {corrected_col}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"GC correction effect plot saved to '{save_path}'")