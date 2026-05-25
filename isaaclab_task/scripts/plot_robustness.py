# ==============================================================================
# SCRIPT: plot_robustness.py
# PURPOSE:
# This script visualizes the quadruped's ability to recover from unexpected
# physical disturbances (Domain Randomization pushes) in the simulation.
#
# HOW IT WORKS:
# 1. It loads the 50Hz telemetry data ('merkoval_telemetry.csv').
# 2. It mathematically searches for the exact moment of maximum deceleration
#    to automatically pinpoint when the physical impulse was applied to the robot.
# 3. It dynamically crops the massive dataset into a clean 10-second window
#    (2 seconds prior to impact, 8 seconds of recovery).
# 4. It plots the velocity drop and recovery against the commanded 0.5m/s target.
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 300, 'savefig.bbox': 'tight'
})

print("Loading telemetry data...")
try:
    df = pd.read_csv('merkoval_telemetry.csv')
except FileNotFoundError:
    print("Error: 'merkoval_telemetry.csv' not found.")
    exit()

# Auto-Detect the Push
smoothed_vel = df['velocity'].rolling(window=10, min_periods=1).mean()
impact_idx = smoothed_vel.diff().idxmin()
impact_time = df['time'].iloc[impact_idx]

print(f"Push automatically detected at t = {impact_time:.2f} seconds!")

# Crop to 10-second window
start_time = max(0, impact_time - 2.0)
end_time = impact_time + 8.0
df_window = df[(df['time'] >= start_time) & (df['time'] <= end_time)].copy()

df_window['display_time'] = df_window['time'] - start_time
display_impact_time = impact_time - start_time

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(df_window['display_time'], df_window['velocity'], color='#8bb0db', alpha=0.6, linewidth=1.2, label='Raw Velocity')
window_smoothed = df_window['velocity'].rolling(window=5, min_periods=1).mean()
ax.plot(df_window['display_time'], window_smoothed, color='#1f77b4', linewidth=2.5, label='Smoothed Velocity')

ax.axhline(y=0.5, color='#2ca02c', linestyle='--', linewidth=1.5, label='Commanded Target (0.5 m/s)')
ax.axvline(x=display_impact_time, color='#d62728', linestyle=':', linewidth=2, label='Randomized Impulse Applied')
ax.annotate('Impact Detected', xy=(display_impact_time + 0.15, 0.1), color='#d62728', fontsize=11, fontweight='bold')

ax.set_title('Robustness Evaluation: Time-Series Push Recovery', fontweight='bold', pad=15)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Forward Velocity (m/s)')
ax.set_ylim(-0.2, 0.8)
ax.set_xlim(0, 10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right', framealpha=0.95)

plt.savefig('push_recovery_graph.png')
print("Success! Graph saved as push_recovery_graph.png")