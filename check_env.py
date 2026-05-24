import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from isaaclab.app import AppLauncher
import cv2
# Start simulation app
launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = launcher.app

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.stack.stack_box.stack_box_env_cfg import FrankaStackBoxEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack.stack_peg.stack_peg_env_cfg import FrankaStackPegEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack.stack_lamp_bulb.stack_lamp_bulb_env_cfg import FrankaStackLampBulbEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack.stack_lighter.stack_lighter_env_cfg import FrankaStackLighterEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack.stack_banana.stack_banana_env_cfg import FrankaStackBananaEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack.stack_bowl.stack_bowl_env_cfg import FrankaStackBowlEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack.stack_mug.stack_mug_env_cfg import FrankaStackMugEnvCfg
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import compute_tactile_shear_image

# --- IK and Teleop imports ---
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg


def save_gelsight_full_visualization(sensor_data, filename_prefix, title_prefix=""):
    """
    Save tactile visualizations using the same approach as the official tacsl_sensor.py demo.
    """
    H, W = 20, 25

    if (
        sensor_data.tactile_normal_force is not None
        and sensor_data.tactile_shear_force is not None
    ):
        normal = sensor_data.tactile_normal_force[0].cpu().numpy().reshape(H, W)
        shear  = sensor_data.tactile_shear_force[0].cpu().numpy().reshape(H, W, 2)

        # compute_tactile_shear_image returns a float32 image in [0, 1], BGR channel order
        tactile_ff_img = compute_tactile_shear_image(normal, shear)         # (H*scale, W*scale, 3)
        cv2.imwrite(
            f"{filename_prefix}_force_field.png",
            (tactile_ff_img * 255).astype(np.uint8),
        )
        print(f"Saved: {filename_prefix}_force_field.png")
    else:
        tactile_ff_img = None
        print(f"[Warning] Cannot save visualization for {title_prefix} sensor: tactile_normal_force or tactile_shear_force is None. Ensure 'enable_force_field=True' and that the robot elastomer makes contact.")


def run_manual_test(env, gripper_fixed_position=0.015, pos_sensitivity=0.05, rot_sensitivity=0.15):
    """
    States: 
    0: Move above box
    1: Lower to box
    2: Close gripper
    3: Lift box
    """
    for cube_name in ["stack_object", "target_cube"]:
        cube_asset = env.unwrapped.scene[cube_name]
        try:
            materials = cube_asset.root_physx_view.get_material_properties()
            mass = cube_asset.root_physx_view.get_masses() 
            print(f"Asset: {cube_name}, Materials: {materials}, Mass: {mass}")
        except Exception as e:
            print(f"Could not get physics properties for {cube_name}: {e}")

    state = 0
    sim_steps = 0
    global_step = 0
    
    # Reset the environment
    obs, _ = env.reset()
    left_sensor = env.unwrapped.scene["left_tactile_sensor"] if "left_tactile_sensor" in env.unwrapped.scene.keys() else None
    right_sensor = env.unwrapped.scene["right_tactile_sensor"] if "right_tactile_sensor" in env.unwrapped.scene.keys() else None

    # Setup Teleop Interface for manual control and callbacks
    teleop_cfg = Se3KeyboardCfg(
        pos_sensitivity=pos_sensitivity,
        rot_sensitivity=rot_sensitivity,
        sim_device=env.unwrapped.device,
    )
    teleop_interface = Se3Keyboard(teleop_cfg)

    # State variables for callbacks
    user_state = {
        "mode": "auto", # "auto" or "manual"
        "capture_idx": 0,
        "capture_requested": False
    }

    def toggle_mode():
        user_state["mode"] = "manual" if user_state["mode"] == "auto" else "auto"
        print(f"\n--- Mode toggled to: {user_state['mode'].upper()} ---")

    def request_capture():
        user_state["capture_requested"] = True

    teleop_interface.add_callback("T", toggle_mode)
    teleop_interface.add_callback("P", request_capture)

    print("\n---------------------------------------------------------")
    print("Starting Manual Test...")
    print("Press 'T' to toggle Auto (State Machine) vs Manual Mode.")
    print("Press 'K' to take a force picture.")
    print("---------------------------------------------------------")
    print(teleop_interface)

    while simulation_app.is_running():
        # Check if capture requested
        if user_state["capture_requested"]:
            idx = user_state["capture_idx"]
            print(f"Capturing force pictures (Index {idx})...")
            if left_sensor is not None:
                save_gelsight_full_visualization(left_sensor.data, f"gelsight_left_capture{idx}", "Left")
            if right_sensor is not None:
                save_gelsight_full_visualization(right_sensor.data, f"gelsight_right_capture{idx}", "Right")
            user_state["capture_idx"] += 1
            user_state["capture_requested"] = False

        # Get manual command from keyboard
        manual_actions = teleop_interface.advance()

        if user_state["mode"] == "manual":
            # Direct teleop
            actions = torch.zeros(env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
            actions[:, 0:6] = manual_actions[0:6]
            if manual_actions[6] > 0.0:
                actions[:, 6:8] = gripper_fixed_position
            else:
                actions[:, 6:8] = 0
        else:
            # 1. Define your action tensor [num_envs, action_dim]
            actions = torch.zeros(env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)

            # Get box and gripper positions directly from the physics scene state
            stack_object_pos = env.unwrapped.scene["stack_object"].data.root_pos_w[:, 0:3] - env.unwrapped.scene.env_origins
            gripper_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, 0:3] - env.unwrapped.scene.env_origins

            # 2. Simple State Machine Logic
            if state == 0: # Move above box
                target_pos = stack_object_pos + torch.tensor([0.0, 0.0, 0.1], device=env.unwrapped.device)
                actions[:, 0:3] = (target_pos - gripper_pos) * 5.0 # P-control to target
                actions[:, 6:8] = 0.04 # Keep gripper open
                if torch.norm(target_pos - gripper_pos) < 0.02: 
                    state = 1
                    print("Auto State: Lower to box")

            elif state == 1: # Lower to box
                actions[:, 0:3] = (stack_object_pos - gripper_pos) * 5.0
                actions[:, 6:8] = 0.04 
                if torch.norm(stack_object_pos - gripper_pos) < 0.01: 
                    state = 2
                    print("Auto State: Close gripper")

            elif state == 2: # CLOSE GRIPPER
                actions[:, 0:3] = 0.0 # Stay still
                actions[:, 6:8] = gripper_fixed_position # Send target fixed position command
                sim_steps += 1
                if sim_steps > 50: 
                    state = 3 # Wait for 50 steps to ensure grip
                    print("Auto State: Lift")

            elif state == 3: # Lift
                actions[:, 2] = 0.5 # Move Up
                actions[:, 6:8] = gripper_fixed_position # Maintain target fixed position command

        # 3. Step the environment
        obs, rew, terminated, truncated, info = env.step(actions)

        # 4. Print individual reward terms
        global_step += 1
        if global_step % 50 == 0:
            print(f"\n--- Step {global_step} Rewards ---")
            stack_obj_pos = env.unwrapped.scene["stack_object"].data.root_pos_w[0, :3] - env.unwrapped.scene.env_origins[0, :3]
            target_cube_pos = env.unwrapped.scene["target_cube"].data.root_pos_w[0, :3] - env.unwrapped.scene.env_origins[0, :3]
            target_cube_pos = env.unwrapped.scene["target_cube"].data.root_pos_w[0, :3] - env.unwrapped.scene.env_origins[0, :3]
            stack_obj_default = env.unwrapped.scene["stack_object"].data.default_root_state[0, :3]
            print(f"  stack_object pos: {stack_obj_pos.tolist()}, default: {stack_obj_default.tolist()}")
            print(f"  target_cube pos: {target_cube_pos.tolist()}")

            dt = env.unwrapped.step_dt
            # env.reset()
            for term_idx, term_name in enumerate(env.unwrapped.reward_manager.active_terms):
                # _step_reward stores the reward rate (reward / dt), so we multiply by dt to get the actual value added
                term_reward = env.unwrapped.reward_manager._step_reward[0, term_idx].item() * dt
                if abs(term_reward) > 1e-6:  # Only print non-zero rewards for clarity
                    print(f"  {term_name}: {term_reward:.4f}")
            print(f"  Total Step Reward: {rew[0].item():.4f}")


def main():
    import sys
    gripper_fixed_position = 0.04
    if "--gripper_pos" in sys.argv:
        idx = sys.argv.index("--gripper_pos")
        if idx + 1 < len(sys.argv):
            gripper_fixed_position = float(sys.argv[idx + 1])

    pos_sensitivity = 0.05
    if "--pos_sens" in sys.argv:
        idx = sys.argv.index("--pos_sens")
        if idx + 1 < len(sys.argv):
            pos_sensitivity = float(sys.argv[idx + 1])

    rot_sensitivity = 0.15
    if "--rot_sens" in sys.argv:
        idx = sys.argv.index("--rot_sens")
        if idx + 1 < len(sys.argv):
            rot_sensitivity = float(sys.argv[idx + 1])

    print(f"Creating Isaac-Stack-Banana-Franka-Gelsight-v0 environment with IK control:")
    print(f"  gripper fixed position: {gripper_fixed_position}")
    print(f"  pos_sensitivity: {pos_sensitivity}")
    print(f"  rot_sensitivity: {rot_sensitivity}")
    
    env_cfg = FrankaStackLighterEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1000.0 # Extend episode length to 1000 seconds for manual debugging
    
    # Enable force fields for tactile sensors to calculate forces and render
    if hasattr(env_cfg, "left_tactile_sensor"):
        env_cfg.left_tactile_sensor.enable_force_field = True
    if hasattr(env_cfg, "right_tactile_sensor"):
        env_cfg.right_tactile_sensor.enable_force_field = True

    # Enable external forces iteration flag to resolve compliant contact force field accurately
    env_cfg.sim.physx.enable_external_forces_every_iteration = True
    # Instead of completely overwriting the robot config (which breaks the custom USD path for sensors),
    # we update the PD gains to be stiffer for better IK tracking.
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    env_cfg.scene.robot.actuators["panda_shoulder"].stiffness = 400.0
    env_cfg.scene.robot.actuators["panda_shoulder"].damping = 80.0
    env_cfg.scene.robot.actuators["panda_forearm"].stiffness = 400.0
    env_cfg.scene.robot.actuators["panda_forearm"].damping = 80.0

    # Set arm actions for the specific robot type to use IK
    env_cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    # Set gripper actions to use continuous joint position control
    env_cfg.actions.gripper_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        scale=1.0,
        use_default_offset=False,
    )
    
    env = gym.make("Isaac-Stack-Lighter-Franka-Gelsight-v0", cfg=env_cfg)
    
    run_manual_test(
        env,
        gripper_fixed_position=gripper_fixed_position,
        pos_sensitivity=pos_sensitivity,
        rot_sensitivity=rot_sensitivity
    )

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
