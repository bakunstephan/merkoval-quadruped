"""
==============================================================================
SCRIPT: flat_env_cfg.py
PURPOSE:
Defines the specialized planar terrain curriculum for the Merkoval quadruped.
Overrides the rough terrain generator to isolate and synthesize a baseline
sagittal-plane walking gait. Applies specific postural regularizations to
prevent asymmetric lateral skew during unconstrained exploration.
==============================================================================
"""

import torch
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewardTerm

from .rough_env_cfg import MerkovalRoughEnvCfg


def hip_roll_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalizes the hip roll (coronal plane) joints for deviating from the 0.0 radian baseline.
    Enforces a strict parallel leg alignment to suppress asymmetric, widened stances
    during initial gait synthesis.
    """
    hip_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.square(hip_pos)
    return torch.sum(error, dim=1)


@configclass
class MerkovalFlatEnvCfg(MerkovalRoughEnvCfg):
    """
    Configuration framework for the flat-terrain gait synthesis curriculum.
    """
    def __post_init__(self):
        # Inherit foundational hardware limits and randomizations
        super().__post_init__()

        # ==========================================================
        # TOPOGRAPHICAL OVERRIDES
        # Isolates kinematics by disabling procedural terrain and restitution
        # ==========================================================
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        )

        if hasattr(self, "curriculum"):
            self.curriculum.terrain_levels = None

        # ==========================================================
        # KINEMATIC VELOCITY CONSTRAINTS
        # Restricts exploration to strict sagittal-plane forward translation
        # ==========================================================
        if hasattr(self, "commands") and hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.6)  # Nominal forward velocity bound
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)  # Suppress lateral translation
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)  # Suppress yaw rotation

        # ==========================================================
        # SPECIALIZED REWARD ECONOMY
        # Modifies base weights for optimal planar convergence
        # ==========================================================
        if hasattr(self, "rewards"):

            # Shift reward prioritization toward structural gait articulation
            if hasattr(self.rewards, "track_lin_vel_xy_exp"):
                self.rewards.track_lin_vel_xy_exp.weight = 3.0  # Scaled down from baseline

            if hasattr(self.rewards, "feet_air_time"):
                self.rewards.feet_air_time.weight = 1.0  # Scaled up to enforce deliberate stepping

            # Postural Regularization: Enforce level chassis orientation (suppress pitch/roll)
            if hasattr(self.rewards, "flat_orientation_l2"):
                self.rewards.flat_orientation_l2.weight = -7.0

            # Postural Regularization: Penalize generic deviation from default state
            if hasattr(self.rewards, "dof_pos_limits"):
                self.rewards.dof_pos_limits.weight = -0.1

            # Postural Regularization: Coronal Plane Constraint
            # Prevents asymmetric lateral splay during baseline training
            self.rewards.hip_posture = RewardTerm(
                func=hip_roll_penalty,
                weight=-2.0,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_0"])}
            )