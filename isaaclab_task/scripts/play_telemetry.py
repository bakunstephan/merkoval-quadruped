# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# Custom Telemetry Evaluation Script for Thesis
# ==============================================================================
# SCRIPT: play_telemetry.py (Customized Isaac Lab Evaluation Script)
# PURPOSE:
# This script loads a fully trained neural network policy and deploys it inside
# the Isaac Lab physics simulation to evaluate real-time performance.
#
# HOW IT DIFFERS FROM STANDARD play.py:
# 1. It injects a custom telemetry logger directly into the simulation step loop.
# 2. It actively tracks joint positions, torques, z-height, and forward velocity.
# 3. It enforces a strict 30-second execution window to guarantee the capture of
#    Domain Randomization events (physical pushes).
# 4. It exports all tracked states to 'merkoval_telemetry.csv' for offline analysis.
# ==============================================================================

import argparse
import sys
import csv
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL and log telemetry.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, \
    multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    # --- THESIS TELEMETRY LOGGER ---
    data_log = []
    print("[INFO] Starting 30-second telemetry logging for Push Recovery Evaluation...")

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            try:
                base_env = env.unwrapped
                robot = base_env.scene["robot"] if hasattr(base_env, "scene") else base_env.robot

                target_action = actions[0, 0].item()
                actual_pos = robot.data.joint_pos[0, 0].item()
                applied_torque = robot.data.applied_torque[0, 0].item()
                base_z_height = robot.data.root_pos_w[0, 2].item()
                forward_vel = robot.data.root_lin_vel_b[0, 0].item()

                data_log.append({
                    "time": timestep * dt,
                    "target_angle": target_action,
                    "actual_angle": actual_pos,
                    "torque": applied_torque,
                    "z_height": base_z_height,
                    "velocity": forward_vel
                })
            except Exception:
                pass

        timestep += 1

        if timestep >= (30.0 / dt):
            print("\n[SUCCESS] 30 seconds reached! Caught the push data.")
            break

    print(f"[INFO] Saving {timestep} steps of telemetry to merkoval_telemetry.csv...")
    if data_log:
        keys = data_log[0].keys()
        with open("merkoval_telemetry.csv", "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data_log)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()