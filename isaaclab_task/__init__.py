"""
==============================================================================
SCRIPT: __init__.py
PURPOSE:
Registers the custom Merkoval quadruped environments within the Gymnasium API.
Links the simulated physical environments to their respective RSL-RL Proximal
Policy Optimization (PPO) runner configurations.
==============================================================================
"""

import gymnasium as gym
from . import rough_env_cfg, flat_env_cfg

gym.register(
    id="Merkoval-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.MerkovalRoughEnvCfg,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.merkoval.agents.rsl_rl_ppo_cfg:MerKovalRoughPPORunnerCfg",
    },
)

gym.register(
    id="Merkoval-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MerkovalFlatEnvCfg,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.merkoval.agents.rsl_rl_ppo_cfg:MerKovalFlatPPORunnerCfg",
    },
)