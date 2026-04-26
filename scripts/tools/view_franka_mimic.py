#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open Isaac Sim GUI and load the Forge Franka USD for quick inspection.

Usage:
    ./isaaclab.sh -p scripts/tools/view_franka_mimic.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open Isaac Sim and view the local Forge Franka USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path


@configclass
class FrankaViewSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = None


def create_robot_cfg(usd_path: str) -> ArticulationCfg:
    """Create the articulation config used by the Factory peg-insert tasks."""
    return ArticulationCfg(
        prim_path="/World/Franka",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.4536,
                "panda_joint2": 0.35,
                "panda_joint3": 0.25,
                "panda_joint4": -1.95,
                "panda_joint5": -0.1029,
                "panda_joint6": 2.05,
                "panda_joint7": 0.7862,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )


def design_scene() -> tuple[str, ArticulationCfg]:
    """Create a minimal scene and return the robot config."""
    usd_path = "/mnt/home/uber/IsaacLab/franka_gelsight_no_finger_collision.usd"
    if check_file_path(usd_path) == 0:
        raise FileNotFoundError(f"Could not find asset: {usd_path}")

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/lightDistant", light_cfg, translation=(2.5, 0.0, 4.0))
    return usd_path, create_robot_cfg(usd_path)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    usd_path, robot_cfg = design_scene()
    scene_cfg = FrankaViewSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    scene_cfg.robot = robot_cfg
    scene = InteractiveScene(scene_cfg)
    robot = Articulation(robot_cfg)

    sim.set_camera_view([2.0, 1.4, 1.5], [0.0, 0.0, 0.6])
    sim.reset()
    scene.update(sim.get_physics_dt())

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = torch.zeros_like(default_joint_pos)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)

    print(f"[INFO]: Loaded USD: {usd_path}")

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        robot.set_joint_position_target(default_joint_pos)


if __name__ == "__main__":
    main()
    simulation_app.close()
