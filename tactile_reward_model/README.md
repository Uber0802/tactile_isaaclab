# `tactile_reward_model`

Dense task-progress reward from GelSight force fields, predicted by a
[Tactile-ReWiND](../external/third-party/Tactile-ReWiND) transformer.

The package is **standalone** — it imports `torch` and `transformers`, never
IsaacLab. You hand it a tactile frame each step and get progress in `[0, 1]`
back. That keeps it usable from analysis scripts and notebooks, not just from a
running simulator.

```
tactile_reward_model/
├── __init__.py               # re-exports TactileRewardCfg, TactileRewardModel
├── tactile_reward_model.py   # the config + the model
└── README.md
```

## Quick start

```python
from tactile_reward_model import TactileRewardCfg, TactileRewardModel

cfg = TactileRewardCfg(ckpt="assets/TactileModel/gear_seed2_scratch_epoch29.pth",
                       instruction="pick up the gear and mesh it onto the shaft",
                       smooth_alpha=0.2)

model = TactileRewardModel.from_cfg(cfg, num_envs=128, device="cuda:0",
                                    max_episode_length=128)
if model is not None:                      # None when cfg.ckpt is empty
    progress = model.compute(frame)        # (num_envs,) in [0, 1]
    model.reset_idx(env_ids)               # on episode reset
```

## Wiring it into an environment

Three hooks. Sketch of what `ForgeEnv` / `StackTactileEnv` actually do:

```python
class MyTactileEnv(SomeIsaacLabEnv):

    # ---- 1. build it once, in __init__ ---------------------------------
    def _init_tactile_reward(self):
        self._tactile_reward_model = None
        rew_cfg = self.cfg.tactile_reward          # a TactileRewardCfg field
        if not rew_cfg.ckpt.strip():
            return                                  # feature disabled, no cost

        self._tactile_reward_model = TactileRewardModel.from_cfg(
            rew_cfg,
            num_envs=self.num_envs,
            device=self.device,
            max_episode_length=self.max_episode_length,
            default_instruction="pick up the gear and mesh it onto the shaft",
        )
        if self._tactile_reward_model is None:
            return                                  # ckpt empty / ReWiND missing

        # reward shaping stays HERE, not in the model
        self._scale = float(rew_cfg.scale)

    # ---- 2. per step: build the frame, scale the progress --------------
    def _compute_tactile_reward(self) -> torch.Tensor:
        if self._tactile_reward_model is None:
            return torch.zeros(self.num_envs, device=self.device)

        left  = self.scene.sensors["left_tactile_sensor"]
        right = self.scene.sensors["right_tactile_sensor"]
        rows, cols = left.cfg.tactile_array_size                    # (20, 25)

        def pad(sensor):                            # -> (N, 20, 25, 3)
            normal = sensor.data.tactile_normal_force.view(self.num_envs, rows, cols, 1)
            shear  = sensor.data.tactile_shear_force.view(self.num_envs, rows, cols, 2)
            return torch.cat([normal, shear], dim=-1)   # (normal, shear_x, shear_y)

        # stack the two pads on the ROW axis -> (N, 40, 25, 3)
        frame = torch.cat([pad(left), pad(right)], dim=1).float()

        progress = self._tactile_reward_model.compute(frame)        # (N,) in [0, 1]
        return progress * self._scale

    # ---- 3. clear per-env history on episode reset ---------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._tactile_reward_model is not None:
            self._tactile_reward_model.reset_idx(env_ids)
```

Then add the reward wherever your env sums its terms — for a direct-workflow env
that is the reward dict:

```python
rew_dict["tactile_progress"] = self._compute_tactile_reward()
rew_scales["tactile_progress"] = 1.0        # scale already applied above
```

and for a manager-based env, a `RewTerm` that calls into the env:

```python
rewind_tactile_reward = RewTerm(func=mdp.rewind_tactile_reward, weight=1.0)
```

Three things this sketch is deliberate about:

- **`_reset_idx` is not optional.** Skip it and a new episode inherits the
  previous one's rolling history and EMA state, which biases early-episode
  reward upward and quietly ruins the first ~`history` steps of every episode.
- **Scaling happens in the env**, after `compute()`. The model returns raw
  progress so one checkpoint can back several reward formulations.
- **Both `from_cfg` returning `None` and an empty `ckpt` are normal**, not
  errors — they are the "reward disabled" path, and every call site has to
  tolerate it.

## The frame contract

`compute()` takes `(num_envs, rows, cols, 3)` with channels ordered
`(normal, shear_x, shear_y)`, and the two finger pads **concatenated along the
row axis** — for the GelSight R15 that is two `20x25` pads giving `(N, 40, 25, 3)`:

```python
left  = left_sensor.data                   # 20 x 25
right = right_sensor.data                  # 20 x 25
frame = torch.cat([left_full, right_full], dim=1)   # -> (N, 40, 25, 3)
```

Row-stacking is not a convention we picked. `TactileCNNEncoder` splits the frame
in half along `bimanual_axis` to recover the two hands, so an odd row count or a
single pad produces a wrong or failed split. `compute()` validates this up front
(`_validate_frame`) and raises in the caller's terms rather than letting the
error surface inside a reshaped `(B*T, C, H, W)` tensor the caller never built.

Note this differs from the layout used when *saving* datasets in
`stack_tactile_env.py`, which stacks the pads on the channel axis
(`(T, 20, 25, 6)`). Loaders convert.

## What it does per step

1. Select the checkpoint's channels (`shear_channels`: `(0,1,2)` for 3-channel
   models, `(1,2)` for shear-only).
2. Push the frame into a per-env rolling history of `history` frames.
3. Subsample that history down to the checkpoint's `max_length` — linspace when
   the window is full, consecutive-with-repeat-last while it is still filling.
   This reproduces training's `_sample_forward + _resize` stride, so inference
   sees the same effective stride the model was trained on.
4. Apply the checkpoint's `normalize_mode` (`off` / `global` / `per_channel`).
   Skipping this feeds the model out-of-distribution magnitudes.
5. Forward pass, take the newest frame's progress, optionally EMA-smooth it.

`compute()` returns a **copy**, so `reset_idx()` zeroing the internal EMA buffer
cannot mutate a tensor the caller is still holding.

## What it deliberately does *not* do

Reward shaping. No scale, no curriculum fade, no clipping — `compute()` returns
raw progress and the caller multiplies. `TactileRewardCfg` carries `scale`,
`scale_end` and `anneal_steps` as data for the caller, and `from_cfg` ignores
them. In this repo the shaping stack is:

| Layer | Applies |
|---|---|
| `TactileRewardModel.compute()` | progress in `[0, 1]`, EMA-smoothed |
| env `compute_tactile_reward()` | `× cfg.tactile_reward.scale` (+ annealing, forge) |
| `mdp.rewind_tactile_reward()` | `×` curriculum fade on success rate (stack) |
| `RewTerm(weight=...)` | `×` static per-task weight |

Keeping the predictor free of these means one checkpoint can back several reward
formulations, and the shaping stays visible where the RL config lives.

## Configuration

`TactileRewardCfg` is a plain stdlib `@dataclass`, **not** an IsaacLab
`@configclass` — that is what keeps the package IsaacLab-free. It still nests
inside a `@configclass` env config and is fully Hydra-overridable:

```bash
env.tactile_reward.ckpt=assets/TactileModel/gear_scratch_epoch18.pth \
env.tactile_reward.scale=0.1 \
env.tactile_reward.smooth_alpha=0.2
```

This works because `configclass` wraps the nested instance in a
`default_factory` (so instances don't share state), `class_to_dict` recurses
into anything with a `__dict__`, and `update_class_from_dict` assigns via
`setattr`.

**No field may default to `None`.** IsaacLab's `update_class_from_dict`
type-checks an override against `type(current_value)`, so a `None` default makes
the field reject *every* override with `Expected: <class 'NoneType'>`. Empty
string and `0` are the "unset" sentinels; `from_cfg` normalizes them.

| Field | Default | Meaning |
|---|---|---|
| `ckpt` | `""` | checkpoint path; empty disables the reward entirely |
| `scale` | `1.0` | applied by the caller, not here |
| `scale_end` / `anneal_steps` | `0.0` / `0` | linear scale fade; `scale_end` read only when `anneal_steps > 0` |
| `instruction` | `""` | empty keeps the calling env's default wording |
| `history` | `0` | rolling-window length; `0` = episode length |
| `smooth_alpha` | `1.0` | EMA on progress; `1.0` disables |
| `rewind_root` | `""` | Tactile-ReWiND checkout; empty = the vendored copy |
| `log_env` / `curve_log_dir` | `0` / `""` | per-episode progress-curve PNGs |

Everything else — `max_length`, `hidden_dim`, `num_layers`, `shear_channels`,
`normalize_mode`, `bimanual_axis` — is read from the checkpoint's `args`, so a
model always runs with the geometry it was trained under.

## Using it from IsaacLab

The package sits at the repo root, which is not on `sys.path` when
`isaaclab_tasks` is installed. Import through the bridge, which fixes the path
and re-exports:

```python
from isaaclab_tasks.utils.tactile_reward_import import TactileRewardCfg, TactileRewardModel
```

The path fix runs at *import* time, not inside a function, because env **config**
modules need `TactileRewardCfg` while their class bodies execute.

## Logging

`logging.getLogger(__name__)`, with no handlers configured — that is the host's
job. Failures (missing checkpoint, unavailable matplotlib) are `WARNING`, so they
reach stderr through `logging.lastResort` even with zero logging setup. The
startup banner is `INFO` and therefore quiet by default; under IsaacLab it lands
in the run's log file, and `SimulationCfg(logging_level="INFO")` surfaces it on
the console.

There is deliberately no `logging.NullHandler()`. The usual library advice
predates Python 3.2's `lastResort`: a `NullHandler` satisfies the handler search,
so `lastResort` never fires and the `WARNING`s go silent in any script that
doesn't configure logging.

## Gotchas

- **`from_cfg` returns `None`** when `ckpt` is empty. Callers must handle it;
  that is the "reward disabled" path, not an error.
- **A missing Tactile-ReWiND checkout** raises `ImportError` inside `from_cfg`,
  which catches it, logs a warning, and returns `None` — training continues
  without the bonus rather than dying.
- **The history buffer is allocated lazily** from the first frame, so the sensor
  resolution is picked up automatically. A later change in resolution is an
  error, not a silent reallocation.
- **Curve PNGs** are written on reset of `log_env` only, and require matplotlib.
  Missing matplotlib warns once and skips.
