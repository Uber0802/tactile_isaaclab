# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Stack-Potted-Meat-Can-Franka-Gelsight-v0",
    entry_point="isaaclab_tasks.manager_based.manipulation.stack.stack_tactile_env:StackTactileEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_potted_meat_can_env_cfg:FrankaStackPottedMeatCanEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
