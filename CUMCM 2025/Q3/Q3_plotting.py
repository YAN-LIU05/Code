# plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter

def setup_matplotlib(font_prefs):
    """Configures Matplotlib for Chinese character support."""
    plt.rcParams['font.sans-serif'] = font_prefs
    plt.rcParams['axes.unicode_minus'] = False

def plot_correlation_heatmap(data):
    """Plots a heatmap of the correlation matrix."""
    correlation_matrix = data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Covariate Correlation Heatmap')
    plt.show()

def plot_km_validation(cox_data):
    """Plots Kaplan-Meier curves for each risk group for validation."""
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    for name, group_df in cox_data.groupby('risk_group'):
        kmf = KaplanMeierFitter()
        kmf.fit(group_df['time'], event_observed=group_df['status'], label=name)
        kmf.plot_survival_function(ax=ax)
    plt.title('Kaplan-Meier Survival Curves by Risk Group')
    plt.xlabel('Gestational Weeks')
    plt.ylabel('Probability of Not Reaching Threshold')
    plt.grid(True)
    plt.show()

def plot_concentration_trajectory(merged_data, y_threshold, group_labels):
    """Plots the average Y-concentration trajectory for each group."""
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=merged_data, x='检测孕期时间', y='Y染色体浓度', hue='risk_group',
                 hue_order=group_labels, errorbar='sd')
    plt.axhline(y=y_threshold, color='red', linestyle='--', label=f'Threshold ({y_threshold})')
    plt.title('Y-Chromosome Concentration Trajectories by Risk Group')
    plt.xlabel('Gestational Weeks')
    plt.ylabel('Y-Chromosome Concentration')
    plt.legend(title='Risk Group')
    plt.grid(True)
    plt.show()

def plot_logistic_parameter_distributions(analysis_df):
    """Generates boxplots for the fitted logistic parameters."""
    param_names = {'t0_fit': 'Midpoint Time (t0)', 'k_fit': 'Growth Rate (k)', 'L_fit': 'Saturation Limit (L)'}
    for param, title in param_names.items():
        if param in analysis_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='risk_group', y=param, data=analysis_df, order=['低风险组(慢)', '中风险组', '高风险组(快)'])
            plt.title(f'Distribution of {title} by Risk Group')
            plt.xlabel('Risk Group')
            plt.ylabel(f'Fitted Parameter: {title}')
            plt.grid(axis='y', linestyle='--')
            plt.show()

def plot_decision_model_results(decision_results, config):
    """Visualizes the results of the decision analysis model."""
    plt.figure(figsize=(15, 9))
    for group, data in decision_results.items():
        p = plt.plot(data['time_range'], data['total_risk'], label=f'{group} - Total Risk', linewidth=3)
        color = p[0].get_color()
        plt.plot(data['time_range'], config.WEIGHT_FN * data['risk_fn_norm'], '--', color=color, alpha=0.6, label=f'{group} - FN Risk')
        plt.plot(data['time_range'], config.WEIGHT_OP * data['risk_op'], '-.', color=color, alpha=0.7, label=f'{group} - Op Risk')
        plt.scatter(data['optimal_time'], data['total_risk'].min(), s=150, zorder=5, color=color, edgecolor='black',
                    label=f'{group} - Optimum ({data["optimal_time"]:.1f} wks)')
    # Plot a single reference for time cost
    ref_time_risk = decision_results[next(iter(decision_results))]['risk_time_norm']
    plt.plot(decision_results[next(iter(decision_results))]['time_range'], config.WEIGHT_TIME * ref_time_risk, ':', color='grey', label='Time Cost (Weighted)')
    plt.title('Decision Model: Optimal Testing Time Analysis')
    plt.xlabel('Gestational Weeks (t)')
    plt.ylabel('Weighted Relative Risk')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sensitivity_analysis(sensitivity_df):
    """Plots the results of the weight sensitivity analysis."""
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=sensitivity_df, x='weight_fn', y='optimal_time', hue='group',
                 hue_order=['低风险组(慢)', '中风险组', '高风险组(快)'], linewidth=2.5, marker='o')
    plt.title('Sensitivity Analysis: Impact of Risk Weighting on Optimal Time')
    plt.xlabel('Weight on False-Negative Risk')
    plt.ylabel('Optimal Gestational Week')
    plt.grid(True)
    plt.legend(title='Risk Group')
    plt.show()

def plot_achievement_timeline(kmf_objects, results_df):
    """Visualizes the achievement rate timeline on the KM curves."""
    plt.figure(figsize=(15, 9))
    ax = plt.subplot(111)
    for group_name, kmf in kmf_objects.items():
        kmf.plot_survival_function(ax=ax)
    sns.scatterplot(data=results_df, x='time_weeks', y='prob_on_km', hue='group', style='achievement_rate',
                    hue_order=['低风险组(慢)', '中风险组', '高风险组(快)'], s=250, edgecolor='black', zorder=10, legend='full', ax=ax)
    plt.title('Key Time Nodes on Kaplan-Meier Curves')
    plt.xlabel('Gestational Weeks')
    plt.ylabel('Probability of Not Reaching Threshold')
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()