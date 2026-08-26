# tactile_isaaclab — Tactile Reward Learning (TaRL)

A research fork of Isaac Lab for **learning manipulation from tactile reward**. A
[Tactile-ReWiND](external/third-party/Tactile-ReWiND) transformer reads GelSight
force fields and predicts task progress in `[0, 1]`, which is fed to the policy as
a dense shaping reward alongside the task reward.

The upstream Isaac Lab README follows below — installation, Isaac Sim version
requirements, and `./isaaclab.sh` usage are unchanged.

## Setup

Snapshot of the [Isaac Lab quickstart](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html),
pinned to the versions this fork runs on:

```bash
# create a virtual environment named env_isaaclab with python3.11 and pip
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab

pip install --upgrade pip
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

./isaaclab.sh --install
```
## Tasks

| Family | Task id | Scripts |
|---|---|---|
| Peg insert | `Isaac-Forge-PegInsert-PickPlace-Direct-v0` | `scripts/TaRL/run_pegpickplace_*.sh` |
| Nut thread | `Isaac-Forge-NutThread-PickPlace-Direct-v0` | `scripts/TaRL/run_nutpickplace_*.sh` |
| Gear mesh | `Isaac-Forge-GearMesh-PickPlace-Direct-v0` | `scripts/TaRL/run_gearpickplace_*.sh` |
| Stack cube | `Isaac-Stack-Cube-Franka-Gelsight-v0` | `scripts/TaRL/run_stack_box*.sh` |
| Stack meat can | `Isaac-Stack-Potted-Meat-Can-Franka-Gelsight-v0` | `scripts/TaRL/run_stack_meat_can*.sh` |

The forge tasks are direct-workflow envs; the stack tasks are manager-based.

## Running

```bash
bash scripts/TaRL/run_gearpickplace_TaRL.sh
```

Each task has the same variants, distinguished by suffix:

| Suffix | Reward / observation |
|---|---|
| `_baseline` | task reward only |
| `_TaRL` | task reward + tactile progress reward |
| `_baseline_visual` | task reward + visual (DINOv2 → ReWiND) progress reward |
| `_TaRL_visual` | both tactile and visual progress rewards |
| `_tactile_state` | frozen tactile encoder embedding as a policy observation |
| `_tactile_state_TaRL` | encoder observation + tactile reward |
| `_datacollection` | no training reward; dumps tactile/camera trajectories |

### Full cycle

The reward model is trained on trajectories that a baseline policy collects, so
the loop bootstraps itself:

**1. Train a baseline.** Run the `_baseline` variant to produce policy
checkpoints spanning a range of skill levels.

**2. Collect tactile trajectories.** Run `_datacollection`, loading those
checkpoints, to sample tactile data at each skill level. Every episode is written
as one `.npy` holding `{"Task", "Tactile", "Success"}` under
`env.tactile_save.save_dir`.

**3. Train the reward model** on the collected trajectories:

```bash
python external/third-party/Tactile-ReWiND/scripts/train_taRL.py \
  --from_scratch \
  --data_dirs tactile_dataset/gearpickplace_tactile \
  --ckpt_dir tactile_dataset/gearpickplace_curriculum \
  --run_name gearpickplace_curriculum \
  --task_texts \
    "pick up the gear and mesh it onto the shaft" \
    "grasp the gear and slide it onto the shaft" \
    "lift the gear and fit it onto the shaft" \
    "place the gear onto the shaft" \
    "use the gripper to pick up the gear and mesh it with the shaft" \
  --in_channels 3 \
  --hidden_dim 512 --num_heads 8 --num_layers 4 \
  --per_hand_dim 384 --num_strided_layers 3 \
  --epochs 30 --batch_size 64 --steps_per_epoch 100 \
  --num_workers 4 --prefetch_factor 4 \
  --lr 1e-4 --min_lr 1e-7 \
  --rewind_ratio 0.5 --success_prob 0.5 --zero_contact_prob 0.15 \
  --max_length 16 --normalize global \
  --max_episodes 10000 --test_ratio 0.1 --test_eval_every 1 \
  --amp bf16 --seed 42
```

Writes one checkpoint per epoch as `{ckpt_dir}/{run_name}_epoch{N}.pth`. Pass
several paraphrases to `--task_texts`: the model is text-conditioned, and
training on paraphrases keeps it from overfitting to one exact wording. Drop
`--from_scratch` and pass `--pretrained <ckpt>` to fine-tune instead.

**4. Train with the tactile reward.** Run `_TaRL`, pointing
`env.tactile_reward.ckpt` at the checkpoint from step 3 and
`env.tactile_reward.instruction` at one of the `--task_texts` used to train it.

The architecture flags above (`--in_channels`, `--hidden_dim`, `--num_heads`,
`--num_layers`, `--per_hand_dim`, `--num_strided_layers`, `--max_length`,
`--normalize`) are saved into the checkpoint under `"args"` and read back when
the reward model loads it. That is why `env.tactile_reward.*` has no fields for
them: a checkpoint always runs with the geometry and normalization it was
trained under, and you never restate them at run time.

## Configuration

The tactile/visual features are configured through the env config, so they go
through Hydra like any other field, are rejected on typos, and are recorded in
each run's `logs/.../params/env.yaml`:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-GearMesh-PickPlace-Direct-v0 \
    "env.tactile_reward.ckpt=assets/TactileModel/gear_seed2_scratch_epoch29.pth" \
    "env.tactile_reward.scale=0.2" \
    "env.tactile_reward.smooth_alpha=0.2"
```

Four config groups: `env.tactile_reward.*`, `env.visual_reward.*`,
`env.tactile_encoder.*`, `env.tactile_save.*`. An empty `ckpt` disables its
feature. Flags that decide scene construction (`FORGE_ENABLE_FRONT_CAM`,
`FORGE_SKIP_TACTILE_SENSORS`, `FORGE_ENABLE_SENSOR`, `FORGE_DISABLE_YAW_DIFF_OBS`)
remain environment variables — they are read in the config's `__post_init__`,
before Hydra applies overrides.

## Layout

| Path | Contents |
|---|---|
| [`tactile_reward_model/`](tactile_reward_model/) | the progress-reward model and its config — see [its README](tactile_reward_model/README.md) |
| `scripts/TaRL/` | one runnable script per task × variant |
| `external/third-party/Tactile-ReWiND/` | vendored reward-model architecture and training code |
| `source/isaaclab_tasks/.../direct/forge*/` | forge pick-place envs |
| `source/isaaclab_tasks/.../manager_based/manipulation/stack/` | stack envs |

## Model weights are not in git

`assets/` and `*.pth` are gitignored. Checkpoints live outside version control
and are referenced by path from the config — put them in `assets/TactileModel/`
and pass `env.tactile_reward.ckpt=assets/TactileModel/<name>.pth`, or point at a
shared filesystem path. **Do not commit `.pth` files**: GitHub hard-rejects any
blob over 100 MB, and a checkpoint that lands in history has to be removed by
rewriting it.

---

![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows,
such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html),
it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real
transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based
cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and
computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks.
Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.

A detailed description of Isaac Lab can be found in our [arXiv paper](https://arxiv.org/abs/2511.04831).

## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:

- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with more than 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.


## Getting Started

### Documentation

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including
detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Isaac Sim Version Dependency

Isaac Lab is built on top of Isaac Sim and requires specific versions of Isaac Sim that are compatible with each
release of Isaac Lab. Below, we outline the recent Isaac Lab releases and GitHub branches and their corresponding
dependency versions for Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version         |
| ----------------------------- | ------------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.3.X`                      | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0       |
| `v2.1.X`                      | Isaac Sim 4.5             |
| `v2.0.X`                      | Isaac Sim 4.5             |


## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Inspiration

We encourage you to utilize our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell)
area in the `Discussions` section of this repository. This space is designed for you to:

* Share the tutorials you've created
* Showcase your learning content
* Present exciting projects you've developed

By sharing your work, you'll inspire others and contribute to the collective knowledge
of our community. Your contributions can spark new ideas and collaborations, fostering
innovation in robotics and simulation.

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas,
  asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of
  work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features,
  or general updates.

## Connect with the NVIDIA Omniverse Community

Do you have a project or resource you'd like to share more widely? We'd love to hear from you!
Reach out to the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to explore opportunities
to spotlight your work.

You can also join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to
connect with other developers, share your projects, and help grow a vibrant, collaborative ecosystem
where creativity and technology intersect. Your contributions can make a meaningful impact on the Isaac Lab
community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its
corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its
dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

Note that Isaac Lab requires Isaac Sim, which includes components under proprietary licensing terms. Please see the [Isaac Sim license](docs/licenses/dependencies/isaacsim-license.txt) for information on Isaac Sim licensing.

Note that the `isaaclab_mimic` extension requires cuRobo, which has proprietary licensing terms that can be found in [`docs/licenses/dependencies/cuRobo-license.txt`](docs/licenses/dependencies/cuRobo-license.txt).


## Citation

If you use Isaac Lab in your research, please cite the technical report:

```
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework.
We gratefully acknowledge the authors of Orbit for their foundational contributions.
