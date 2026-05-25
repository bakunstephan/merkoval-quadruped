from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from .merkoval import MERKOVAL_CFG

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class MerkovalRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    Configuration framework for the Merkoval quadruped digital twin.

    This class inherits from the base locomotion environment and applies
    strict physical regularizations and domain randomizations to synthesize
    a deployment-ready, hardware-safe locomotion policy for the 64.36kg platform.
    """

    def __post_init__(self):
        super().__post_init__()

        # ==========================================================
        # ASSET INITIALIZATION
        # ==========================================================
        self.scene.robot = MERKOVAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ==========================================================
        # SENSOR & OBSERVATION ALIGNMENT
        # Maps simulated sensors to specific physical hardware geometries
        # ==========================================================
        if hasattr(self.scene, "contact_forces"):
            self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/dog1/.*"
            self.scene.contact_forces.update_period = 0.0

        if hasattr(self.scene, "height_scanner"):
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/dog1/base_link"

        if hasattr(self, "observations") and hasattr(self.observations, "policy"):
            for attr_name in dir(self.observations.policy):
                if not attr_name.startswith("_"):
                    attr = getattr(self.observations.policy, attr_name)
                    if hasattr(attr, "params") and "asset_cfg" in attr.params:
                        if getattr(attr.params["asset_cfg"], "body_names", None) in ["base", ["base"]]:
                            attr.params["asset_cfg"].body_names = ["base_link"]

        if hasattr(self, "events"):
            for attr_name in dir(self.events):
                if not attr_name.startswith("_"):
                    attr = getattr(self.events, attr_name)
                    if hasattr(attr, "params") and "asset_cfg" in attr.params:
                        if getattr(attr.params["asset_cfg"], "body_names", None) in ["base", ["base"]]:
                            attr.params["asset_cfg"].body_names = ["base_link"]

        # ==========================================================
        # KINEMATIC INITIALIZATION
        # Initializes agent with a vertical offset to train dynamic recovery
        # ==========================================================
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.8)
        self.events.reset_base.params["pose_range"] = {"z": (0.76, 0.85)}

        # ==========================================================
        # CURRICULUM CONSTRAINTS
        # Isolates sagittal-plane locomotion (0.4 to 0.6 m/s)
        # ==========================================================
        if hasattr(self, "commands") and hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.6)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ==========================================================
        # HARDWARE REGULARIZATION & REWARD ECONOMY
        # Binds the RL policy to the physical limits of the actuators
        # ==========================================================
        if hasattr(self, "rewards"):
            # Target Velocity Tracking
            if hasattr(self.rewards, "track_lin_vel_xy_exp"):
                self.rewards.track_lin_vel_xy_exp.weight = 15.0
            if hasattr(self.rewards, "track_ang_vel_z_exp"):
                self.rewards.track_ang_vel_z_exp.weight = 1.0
            if hasattr(self.rewards, "ang_vel_xy_l2"):
                self.rewards.ang_vel_xy_l2.weight = -0.001

            # Enforces proper gait articulation (0.25s swing phase baseline)
            if hasattr(self.rewards, "feet_air_time"):
                if "sensor_cfg" in self.rewards.feet_air_time.params:
                    self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*_Foot"]
                self.rewards.feet_air_time.weight = 0.01
                self.rewards.feet_air_time.params["threshold"] = 0.25

            # Hardware Safety: Prevents chassis ground collisions
            if hasattr(self.rewards, "undesired_contacts"):
                if "sensor_cfg" in self.rewards.undesired_contacts.params:
                    self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base_link", ".*_Hip",
                                                                                       ".*_Thigh", ".*_Calf"]
                self.rewards.undesired_contacts.weight = -10.0

            if hasattr(self.rewards, "feet_contact_forces") and "sensor_cfg" in self.rewards.feet_contact_forces.params:
                self.rewards.feet_contact_forces.params["sensor_cfg"].body_names = [".*_Foot"]

            # Hardware Safety: Suppresses high-frequency motor oscillation
            if hasattr(self.rewards, "action_rate_l2"):
                self.rewards.action_rate_l2.weight = -0.15
            if hasattr(self.rewards, "dof_acc_l2"):
                self.rewards.dof_acc_l2.weight = -2.5e-7

            # Hardware Safety: Strict torque penalty to prevent Harmonic Drive overload
            if hasattr(self.rewards, "dof_torques_l2"):
                self.rewards.dof_torques_l2.weight = -0.000010

            # Suppresses inefficient vertical kinetic energy loss
            if hasattr(self.rewards, "lin_vel_z_l2"):
                self.rewards.lin_vel_z_l2.weight = -2.0

        # ==========================================================
        # TERMINATIONS
        # ==========================================================
        if hasattr(self, "terminations"):
            if hasattr(self.terminations, "base_contact"):
                if "sensor_cfg" in self.terminations.base_contact.params:
                    self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]
                self.terminations.base_contact.active = True

        # ==========================================================
        # DOMAIN RANDOMIZATION PIPELINE
        # Bridges the sim-to-real gap by simulating mechanical wear and payload variance
        # ==========================================================
        if hasattr(self, "events"):
            # Dynamic Perturbation: Tests lateral robustness (1.5 m/s impulse)
            if hasattr(self.events, "push_robot"):
                self.events.push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}

            # Payload Variance: Simulates diverse equipment loads
            if hasattr(self.events, "add_base_mass"):
                self.events.add_base_mass.params["mass_distribution_params"] = (-5.0, 15.0)

            # Center of Mass (CoM) Displacement
            if hasattr(self.events, "base_com"):
                self.events.base_com.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}

            # Contact Mechanics: Scrambles floor friction across wide margins
            if hasattr(self.events, "physics_material"):
                self.events.physics_material.params["static_friction_range"] = (0.3, 1.25)
                self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.25)

            # Actuator Degradation: Scales PD gains to simulate gear wear
            self.events.randomize_actuator_gains = EventTerm(
                func=mdp.randomize_actuator_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "stiffness_distribution_params": (0.85, 1.15),
                    "damping_distribution_params": (0.85, 1.15),
                    "operation": "scale",
                    "distribution": "uniform",
                }
            )

            # Internal Friction: Scales joint friction parameters
            self.events.randomize_joint_parameters = EventTerm(
                func=mdp.randomize_joint_parameters,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "friction_distribution_params": (0.75, 1.25),
                    "operation": "scale",
                    "distribution": "uniform",
                }
            )

        # Simulates real-world sensor noise (IMU and Encoders)
        if hasattr(self, "observations") and hasattr(self.observations, "policy"):
            self.observations.policy.enable_corruption = True