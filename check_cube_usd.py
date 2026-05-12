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
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_gelsight_env_cfg import FrankaGelsightEnvCfg
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import compute_tactile_shear_image

# --- IK and Teleop imports ---
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
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

        print(f"Env 0 Normal Force (min/max): {normal.min().item():.4f} / {normal.max().item():.4f}")
        print(f"Env 0 Shear Force (min/max): {shear.min().item():.4f} / {shear.max().item():.4f}")

        # compute_tactile_shear_image returns a float32 image in [0, 1], BGR channel order
        tactile_ff_img = compute_tactile_shear_image(normal, shear)         # (H*scale, W*scale, 3)
        cv2.imwrite(
            f"{filename_prefix}_force_field.png",
            (tactile_ff_img * 255).astype(np.uint8),
        )
        print(f"Saved: {filename_prefix}_force_field.png")
    else:
        tactile_ff_img = None


def run_manual_test(env):
    """
    States: 
    0: Move above box
    1: Lower to box
    2: Close gripper
    3: Lift box
    """
    state = 0
    sim_steps = 0
    global_step = 0
    
    # Reset the environment
    obs, _ = env.reset()
    left_sensor = env.unwrapped.scene["left_tactile_sensor"]
    right_sensor = env.unwrapped.scene["right_tactile_sensor"]

    while simulation_app.is_running():
        # Check if capture requested

        # 1. Define your action tensor [num_envs, action_dim]
        actions = torch.zeros(env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)

        # Get box and gripper positions directly from the physics scene state
        box_pos = env.unwrapped.scene["cube_1"].data.root_pos_w[:, 0:3] - env.unwrapped.scene.env_origins
        gripper_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, 0:3] - env.unwrapped.scene.env_origins

        # 2. Simple State Machine Logic
        if state == 0: # Move above box
            target_pos = box_pos + torch.tensor([0.0, 0.0, 0.1], device=env.unwrapped.device)
            actions[:, 0:3] = (target_pos - gripper_pos) * 5.0 # P-control to target
            actions[:, -1] = 1.0 # Keep gripper open
            if torch.norm(target_pos - gripper_pos) < 0.02: 
                state = 1
                print("Auto State: Lower to box")

        elif state == 1: # Lower to box
            actions[:, 0:3] = (box_pos - gripper_pos) * 5.0
            actions[:, -1] = 1.0 
            if torch.norm(box_pos - gripper_pos) < 0.01: 
                state = 2
                print("Auto State: Close gripper")

        elif state == 2: # CLOSE GRIPPER
            actions[:, 0:3] = 0.0 # Stay still
            actions[:, -1] = -1.0 # Send maximum CLOSE command
            sim_steps += 1
            if sim_steps > 50: 
                state = 3 # Wait for 50 steps to ensure grip
                print("Auto State: Lift")

        elif state == 3: # Lift
            actions[:, 2] = 0.5 # Move Up
            actions[:, -1] = -1.0 # Maintain grip

        # 3. Step the environment
        obs, rew, terminated, truncated, info = env.step(actions)

        # 4. Print individual reward terms
        global_step += 1
        if global_step % 50 == 0:
            print(f"\n--- Step {global_step}")
            save_gelsight_full_visualization(left_sensor.data, f"gelsight_left_capture{global_step}")
            save_gelsight_full_visualization(right_sensor.data, f"gelsight_right_capture{global_step}")
            
        '''
        if global_step % 50 == 0:
            print(f"\n--- Step {global_step} Rewards ---")
            dt = env.unwrapped.step_dt
            for term_idx, term_name in enumerate(env.unwrapped.reward_manager.active_terms):
                # _step_reward stores the reward rate (reward / dt), so we multiply by dt to get the actual value added
                term_reward = env.unwrapped.reward_manager._step_reward[0, term_idx].item() * dt
                if abs(term_reward) > 1e-6:  # Only print non-zero rewards for clarity
                    print(f"  {term_name}: {term_reward:.4f}")
            print(f"  Total Step Reward: {rew[0].item():.4f}")
        '''
def main():
    env_cfg = FrankaGelsightEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1000.0 # Extend episode length to 1000 seconds for manual debugging
    
    env = gym.make("Isaac-Stack-Cube-Franka-Gelsight-v0", cfg=env_cfg)
    
    run_manual_test(env)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
