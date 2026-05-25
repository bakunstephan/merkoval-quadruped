"""
==============================================================================
SCRIPT: rsl_rl_ppo_cfg.py
PURPOSE:
Defines the Proximal Policy Optimization (PPO) hyperparameters and neural
network architectures for the Merkoval quadruped. Utilizes the RSL-RL
library for high-throughput actor-critic policy learning.
==============================================================================
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class MerKovalRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Base PPO configuration for the rough-terrain / generalized locomotion curriculum.
    """
    num_steps_per_env = 24
    max_iterations = 7500
    save_interval = 50
    experiment_name = "merkoval_rough"
    run_name = ""

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # Neural network architecture matching baseline specifications
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MerKovalFlatPPORunnerCfg(MerKovalRoughPPORunnerCfg):
    """
    PPO configuration tailored for the flat-terrain baseline synthesis.
    Inherits network architecture and core algorithm settings from the base configuration.
    """
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 500
        self.experiment_name = "merkoval_flat"