# ==============================================================================
# SCRIPT: plot_thesis_graphs.py
# PURPOSE:
# This script processes raw training metric data exported from TensorBoard
# and generates the 9 core evaluation graphs for the thesis Results chapter.
#
# HOW IT WORKS:
# 1. It loads raw .csv files containing training step/value pairs.
# 2. It plots the raw data as a faded background shadow to maintain data integrity.
# 3. It applies a rolling window moving average to generate a clear trend line.
# 4. It enforces strict academic formatting (Times New Roman, 300 DPI, clean grids).
# 5. It outputs high-resolution .png images ready for document insertion.
#
# USAGE:
# Place this script in the same folder as the TensorBoard-exported .csv files,
# then run it. Output graphs are written to a 'Thesis_Ready_Graphs' subfolder.
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

# 1. Set global plot style for strict thesis standards
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

data_dir = os.path.dirname(os.path.abspath(__file__))  # Run from a folder containing the TensorBoard CSVs
output_dir = os.path.join(data_dir, "Thesis_Ready_Graphs")
os.makedirs(output_dir, exist_ok=True)

graphs_to_plot = {
    'mean_reward.csv': ('Algorithm Convergence: Mean Episode Reward', 'Cumulative Reward'),
    'mean_episode length.csv': ('Survivability: Mean Episode Length', 'Steps'),
    'lin_vel_xy.csv': ('Kinematic Tracking: Forward Velocity Reward', 'Velocity Reward'),
    'feet_air_time.csv': ('Gait Articulation: Feet Air Time', 'Reward'),
    'dof_torque_l2.csv': ('Hardware Safety: Actuator Torque Regularization', 'Torque Penalty'),
    'action_rate_l2.csv': ('Hardware Safety: Motor Command Smoothness', 'Action Rate Penalty'),
    'undesired.csv': ('Hardware Safety: Undesired Contact Penalty', 'Penalty'),
    'flat_orientation.csv': ('Postural Regulation: Chassis Orientation', 'Orientation Penalty'),
    'hip_posture.csv': ('Postural Regulation: Hip Roll Constraint', 'Posture Penalty')
}

print("Generating thesis-grade graphs...")

for filename, (title, ylabel) in graphs_to_plot.items():
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot raw data (faded background)
        ax.plot(df['Step'], df['Value'], color='#8bb0db', alpha=0.7, linewidth=1.2, label='Raw Data')

        # Plot smoothed trend line (clean foreground)
        smoothing_window = max(1, len(df) // 20)
        smoothed_data = df['Value'].rolling(window=smoothing_window, min_periods=1).mean()
        ax.plot(df['Step'], smoothed_data, color='#1f77b4', linewidth=2, label='Smoothed Trend')

        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xlabel('Training Iterations (Steps)')
        ax.set_ylabel(ylabel)


        def format_steps(x, pos):
            if x >= 1e6:
                return f'{x * 1e-6:g}M'
            elif x >= 1e3:
                return f'{x * 1e-3:g}K'
            return str(int(x))


        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_steps))

        ax.grid(True, which='major', linestyle='-', alpha=0.3, color='gray')
        ax.legend(loc='best', framealpha=0.9)

        plt.savefig(os.path.join(output_dir, filename.replace('.csv', '.png')))
        plt.close()
        print(f"Generated: {filename.replace('.csv', '.png')}")