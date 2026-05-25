"""
==============================================================================
SCRIPT: merkoval.py
PURPOSE:
Asset configuration for the Merkoval quadruped robot. Defines the physical
USD payload, rigid body parameters, initial kinematic states, and the strict
baseline hardware actuator limits for the digital twin.
==============================================================================
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

MERKOVAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "..", "robot", "merkoval1.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Initial spatial initialization (0.8m drop) to train dynamic weight-catch recovery
        pos=(0.0, 0.0, 0.8),

        # Nominal baseline stance for low-level PD controller initialization
        joint_pos={
            "Joint_1_1": -0.2,
            "Joint_1_2": -0.4,
            "Joint_2_1": 0.2,
            "Joint_2_2": 0.4,
            "Joint_3_1": -0.2,
            "Joint_3_2": 0.4,
            "Joint_4_1": 0.2,
            "Joint_4_2": -0.4,
            "Joint_.*_0": 0.0,  # Coronal plane (hip-roll) constraint baseline
        },
        joint_vel={"Joint_.*": 0.0},
    ),
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=["Joint_.*"],
            # Absolute physical boundaries (Harmonic Drive CSG-25 constraints)
            effort_limit=229.0,
            saturation_effort=229.0,
            velocity_limit=3.27,
            # Nominal PD tracking gains (Subject to Domain Randomization scaling in env_cfg)
            stiffness=800.0,
            damping=50.0,
            # Compensatory simulation friction (Physical hardware static stiction is ~14 Nm)
            friction=3.5,
        ),
    },
)