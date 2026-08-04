"""Pretrain a tactile force-field autoencoder for the `raw_tactile` RL baseline.

Trains  frame -> TactileCNNEncoder -> z -> deconv decoder -> frame  with MSE
reconstruction loss on the float16 episodes saved by
FORGE_SAVE_TACTILE_FORCE_FIELD. Episodes land on disk in either layout —
(T, 20, 25, 6) with the hands on the channel axis (what StackTactileEnv's
dataset dump writes) or (T, 40, 25, 3) with them on the row axis — and are
normalized on load to the row-stacked form the encoder splits on: channels =
normal, shear_x, shear_y; rows 0-19 = left finger, 20-39 = right finger. The
decoder is thrown away at RL time; the frozen encoder produces the
`tactile_embedding` obs key.

Unlike the ReWiND reward-model encoder (Baseline B2), this latent is trained
purely for reconstruction — it contains "what the sensor feels", not task
progress — so the `raw_tactile` baseline isolates tactile-as-state from
tactile-as-reward.

Checkpoint format is compatible with `ForgeEnv._init_tactile_encoder`:
  {"model_state_dict": {encoder.* , decoder.*}, "args": {...}}
with `args` carrying in_channels / per_hand_dim / num_strided_layers /
bimanual_axis / global_scale so the runtime rebuilds and feeds the encoder
exactly as trained.

Example (subdirs like <root>/ep_100/ep3_env071.npy are discovered recursively;
pass one or more roots):
    python train_tactile_ae.py \
        --data_dir /mnt/tank/tactile/tactile_dataset/pegpickplace_paired \
        --out_dir  /mnt/tank/uber/Tactile-Reward/tactile_ae_peg \
        --per_hand_dim 64 --epochs 40
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.tactile_model import TactileCNNEncoder  # noqa: E402


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class TactileAEDecoder(nn.Module):
    """Shared-weight per-hand deconv decoder (mirror of TactileCNNEncoder).

    z (B, 2*per_hand_dim) -> split into two per-hand codes -> shared
    fc + upsample-conv stack -> per-hand (C, 20, 25) -> concat on H
    -> (B, C, 40, 25).
    """

    # Encoder with 3 stride-2 layers maps (20, 25) -> (3, 4) before pooling.
    _SEED_HW = (3, 4)
    _SEED_CH = 128

    def __init__(self, in_channels: int, per_hand_dim: int, latent_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.per_hand_dim = per_hand_dim
        # Undo the encoder's fusion linear.
        self.split = nn.Linear(latent_dim, 2 * per_hand_dim)
        h, w = self._SEED_HW
        self.fc = nn.Linear(per_hand_dim, self._SEED_CH * h * w)

        def up_block(cin, cout):
            return [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.GroupNorm(8, cout),
                nn.GELU(),
            ]

        # (128, 3, 4) -> (64, 6, 8) -> (32, 12, 16) -> (16, 24, 32)
        self.deconv = nn.Sequential(
            *up_block(self._SEED_CH, 64),
            *up_block(64, 32),
            *up_block(32, 16),
            nn.Conv2d(16, in_channels, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h, w = self._SEED_HW
        codes = self.split(z).chunk(2, dim=-1)          # 2 x (B, per_hand_dim)
        hands = []
        for code in codes:
            f = self.fc(code).view(-1, self._SEED_CH, h, w)
            img = self.deconv(f)                        # (B, C, 24, 32)
            img = F.interpolate(img, size=(20, 25), mode="bilinear", align_corners=False)
            hands.append(img)
        return torch.cat(hands, dim=-2)                 # (B, C, 40, 25)


class TactileAE(nn.Module):
    """Encoder must stay under the attribute name `encoder` — the RL-side
    loader (`ForgeEnv._init_tactile_encoder`) filters state_dict keys by the
    `encoder.` prefix."""

    def __init__(self, in_channels: int, per_hand_dim: int):
        super().__init__()
        latent_dim = 2 * per_hand_dim                   # B2 loader convention
        self.encoder = TactileCNNEncoder(
            in_channels=in_channels,
            per_hand_dim=per_hand_dim,
            output_dim=latent_dim,
            num_strided_layers=3,
            bimanual_axis="height",
        )
        self.decoder = TactileAEDecoder(in_channels, per_hand_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def _to_row_stacked(frames: np.ndarray) -> np.ndarray | None:
    """Normalize an episode to the row-stacked layout (T, 2*rows, cols, 3).

    Two on-disk layouts exist, because the recorder and the reward path
    concatenate the hands on different axes:

      * (T, 20, 25, 6) — hands on the CHANNEL axis, [left(n,sx,sy),
        right(n,sx,sy)]; written by StackTactileEnv's dataset dump
        (`torch.cat((left_field, right_field), dim=-1)`).
      * (T, 40, 25, 3) — hands on the ROW axis; what the ReWiND reward path
        feeds the encoder (`torch.cat([l_full, r_full], dim=1)`).

    The encoder splits on height (`bimanual_axis="height"`), so the row-stacked
    form is canonical — convert the channel-stacked one to match, keeping the
    same left-then-right order the runtime uses.
    """
    c = frames.shape[-1]
    if c == 3:
        return frames
    if c == 6:
        left, right = frames[..., :3], frames[..., 3:]
        return np.concatenate([left, right], axis=1)
    return None


def _load_episode(path: str) -> np.ndarray | None:
    """Return (T, 2*rows, cols, 3) float16 frames, or None if unreadable."""
    try:
        raw = np.load(path, allow_pickle=True)
    except Exception as e:  # noqa: BLE001
        print(f"[data] skip {path}: {e}")
        return None
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        payload = raw.item()
        frames = payload.get("Tactile") if isinstance(payload, dict) else None
    else:
        frames = raw
    if frames is None or getattr(frames, "ndim", 0) != 4 or frames.shape[0] == 0:
        print(f"[data] skip {path}: no (T, H, W, C) tactile array")
        return None
    frames = np.asarray(frames, dtype=np.float16)
    stacked = _to_row_stacked(frames)
    if stacked is None:
        print(f"[data] skip {path}: expected 3 (row-stacked) or 6 (channel-stacked) "
              f"channels, got shape {frames.shape}")
        return None
    return stacked


def discover_files(data_dirs):
    """Recursively find episode files, e.g. <root>/ep_100/ep3_env071.npy.

    Any nesting depth works — files are matched by basename ep*.npy and
    *_camera.npy companions are skipped. Prints a per-subdir census so it is
    obvious which rollout batches were picked up.
    """
    files = []
    for d in data_dirs:
        hits = glob.glob(os.path.join(d, "**", "ep*.npy"), recursive=True)
        files += [f for f in hits if not f.endswith("_camera.npy")]
    files = sorted(dict.fromkeys(files))          # dedupe overlapping dirs
    if not files:
        raise FileNotFoundError(f"no ep*.npy under {data_dirs}")
    groups = {}
    for f in files:
        groups.setdefault(os.path.dirname(f), []).append(f)
    for g in sorted(groups):
        print(f"[data] {g}: {len(groups[g])} episodes")
    print(f"[data] {len(files)} episode files total in {len(groups)} dirs")
    return files, groups


def load_frames(data_dirs, val_frac: float, max_frames: int, seed: int):
    """Load episodes and return float16 (N, 40, 25, 3) train/val arrays.

    * Train/val split is at the EPISODE level — consecutive frames are
      near-duplicates, so a frame-level split would leak train data into val.
    * When the dataset holds more frames than --max_frames, episodes are
      subsampled BEFORE loading, stratified across subdirs so every rollout
      batch (ep_100/, ep_200/, ...) stays proportionally represented. This
      keeps peak RAM (and NAS reads) at the cap instead of the full dataset
      size — with thousands of files only the kept episodes are ever opened.
    """
    files, groups = discover_files(data_dirs)
    rng = np.random.default_rng(seed)

    if max_frames and len(files) > 8:
        # Probe a few episodes to estimate frames/episode, then figure out
        # how many episodes we actually need to open.
        probe_paths = [files[i] for i in rng.choice(len(files), size=8, replace=False)]
        probe = [ep for ep in map(_load_episode, probe_paths) if ep is not None]
        if not probe:
            raise RuntimeError("probe episodes unreadable — check the data files")
        est_fpe = float(np.mean([ep.shape[0] for ep in probe]))
        # train cap + val cap (max_frames // 10), +15% slack for estimate error.
        want_files = int(np.ceil(1.1 * max_frames / est_fpe * 1.15))
        if want_files < len(files):
            keep = []
            for fs in groups.values():
                k = max(1, int(round(len(fs) / len(files) * want_files)))
                pick = rng.choice(len(fs), size=min(k, len(fs)), replace=False)
                keep += [fs[i] for i in pick]
            keep = [keep[i] for i in rng.permutation(len(keep))][:want_files]
            files = keep
            print(f"[data] subsampled to {len(files)} episodes "
                  f"(~{est_fpe:.0f} frames/ep -> ~{int(len(files) * est_fpe)} frames "
                  f"for max_frames={max_frames})")

    if len(files) < 2:
        raise RuntimeError("need at least 2 episodes for a train/val split")
    files = [files[i] for i in rng.permutation(len(files))]
    n_val = max(1, int(round(len(files) * val_frac)))
    val_files, train_files = files[:n_val], files[n_val:]

    def _load_many(paths, tag):
        eps = []
        for i, path in enumerate(paths):
            ep = _load_episode(path)
            if ep is not None:
                eps.append(ep)
            if (i + 1) % 500 == 0:
                print(f"[data] {tag}: loaded {i + 1}/{len(paths)} episodes")
        if not eps:
            raise RuntimeError(f"empty {tag} split — no readable episodes")
        return np.concatenate(eps, axis=0)

    train = _load_many(train_files, "train")
    val = _load_many(val_files, "val")
    # Exact frame caps (uniform subsample) in case the estimate overshot.
    if max_frames and train.shape[0] > max_frames:
        keep = rng.choice(train.shape[0], size=max_frames, replace=False)
        train = train[np.sort(keep)]
    if max_frames and val.shape[0] > max_frames // 10:
        keep = rng.choice(val.shape[0], size=max_frames // 10, replace=False)
        val = val[np.sort(keep)]
    return train, val


def frame_max_abs(frames: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Per-frame max |F| as float32 (chunked so fp16 arrays stay cheap)."""
    out = np.empty(frames.shape[0], dtype=np.float32)
    for s in range(0, frames.shape[0], chunk):
        blk = frames[s : s + chunk].astype(np.float32)
        out[s : s + blk.shape[0]] = np.abs(blk).max(axis=(1, 2, 3))
    return out


def compute_global_scale(frames: np.ndarray, contact_idx: np.ndarray,
                         percentile: float, seed: int) -> float:
    """Fixed dataset-wide scale: percentile of |F| over contact frames.

    A single constant (stored in the ckpt, applied identically at RL time)
    keeps inputs bounded WITHOUT destroying magnitude information the way
    per-frame max-abs normalization would — grip strength survives.
    """
    rng = np.random.default_rng(seed)
    pick = contact_idx if contact_idx.size <= 2000 else \
        rng.choice(contact_idx, size=2000, replace=False)
    vals = np.abs(frames[np.sort(pick)].astype(np.float32)).ravel()
    scale = float(np.percentile(vals[vals > 0], percentile)) if (vals > 0).any() else 1.0
    return scale if scale > 0 else 1.0


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def make_batch(frames, pool_contact, pool_rest, bs, contact_frac, scale, device, rng):
    n_c = min(int(round(bs * contact_frac)), pool_contact.size) if pool_contact.size else 0
    n_r = bs - n_c
    idx = []
    if n_c:
        idx.append(rng.choice(pool_contact, size=n_c, replace=pool_contact.size < n_c))
    if n_r:
        pool = pool_rest if pool_rest.size else pool_contact
        idx.append(rng.choice(pool, size=n_r, replace=pool.size < n_r))
    idx = np.sort(np.concatenate(idx))
    x = torch.from_numpy(frames[idx].astype(np.float32)).to(device)
    x = (x / scale).permute(0, 3, 1, 2).contiguous()     # (B, C, 40, 25)
    return x


def weighted_mse(recon, x, contact_weight: float):
    if contact_weight == 1.0:
        return F.mse_loss(recon, x)
    # SDF forces are exactly 0 outside contact, so >0 is a clean pixel mask.
    mask = (x.abs().amax(dim=1, keepdim=True) > 0).float()
    weight = 1.0 + (contact_weight - 1.0) * mask
    return (weight * (recon - x) ** 2).mean()


@torch.no_grad()
def evaluate(model, frames, contact_mask, scale, device, bs=512):
    model.eval()
    tot, tot_c, n, n_c = 0.0, 0.0, 0, 0
    for s in range(0, frames.shape[0], bs):
        blk = frames[s : s + bs]
        x = torch.from_numpy(blk.astype(np.float32)).to(device)
        x = (x / scale).permute(0, 3, 1, 2).contiguous()
        recon, _ = model(x)
        per = ((recon - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        m = contact_mask[s : s + bs]
        tot += float(per.sum()); n += per.size
        tot_c += float(per[m].sum()); n_c += int(m.sum())
    model.train()
    return tot / max(n, 1), tot_c / max(n_c, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", nargs="+", required=True,
                   help="dir(s) with ep*.npy force-field episodes (searched recursively)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--in_channels", type=int, default=3, choices=(2, 3),
                   help="3 = (normal, shear_x, shear_y); 2 = shear-only")
    p.add_argument("--per_hand_dim", type=int, default=64,
                   help="latent = 2*per_hand_dim (default 128 total)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--contact_frac", type=float, default=0.7,
                   help="fraction of each batch drawn from contact frames")
    p.add_argument("--contact_thresh", type=float, default=0.0,
                   help="abs force above which a frame counts as contact; "
                        "0 = auto (1%% of the p99.9 per-frame max)")
    p.add_argument("--contact_weight", type=float, default=3.0,
                   help="MSE weight on contact pixels (1.0 = plain MSE)")
    p.add_argument("--scale_percentile", type=float, default=99.9,
                   help="percentile of |F| (contact frames) used as the fixed input scale")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--max_frames", type=int, default=400_000,
                   help="RAM cap on train frames (0 = unlimited)")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train, val = load_frames(args.data_dir, args.val_frac, args.max_frames, args.seed)
    if args.in_channels == 2:
        train, val = train[..., 1:], val[..., 1:]        # drop normal, keep shear
    print(f"[data] train {train.shape}  val {val.shape}  (float16)")

    tr_max = frame_max_abs(train)
    va_max = frame_max_abs(val)
    p999 = float(np.percentile(tr_max, 99.9))
    thresh = args.contact_thresh or 0.01 * p999
    pool_contact = np.nonzero(tr_max > thresh)[0]
    pool_rest = np.nonzero(tr_max <= thresh)[0]
    val_contact_mask = va_max > thresh
    print(f"[data] per-frame max|F|: p50={np.percentile(tr_max, 50):.4g} "
          f"p99.9={p999:.4g}  contact_thresh={thresh:.4g}")
    print(f"[data] contact frames: {pool_contact.size}/{tr_max.size} "
          f"({100 * pool_contact.size / tr_max.size:.1f}%)")
    if pool_contact.size == 0:
        raise RuntimeError("no contact frames found — lower --contact_thresh")

    scale = compute_global_scale(train, pool_contact, args.scale_percentile, args.seed)
    print(f"[data] global_scale = {scale:.6g}")

    model = TactileAE(args.in_channels, args.per_hand_dim).to(args.device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[model] latent={2 * args.per_hand_dim}  params={n_params / 1e6:.2f}M")

    steps = max(1, train.shape[0] // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)

    ckpt_args = {
        "in_channels": args.in_channels,
        "per_hand_dim": args.per_hand_dim,
        "num_strided_layers": 3,
        "bimanual_axis": "height",
        "global_scale": scale,
        "normalize_mode": "fixed_scale",
        "ae_pretrained": True,
        "contact_thresh": thresh,
    }

    def save(path, epoch, val_loss):
        torch.save({"model_state_dict": model.state_dict(), "args": ckpt_args,
                    "epoch": epoch, "val_loss": val_loss}, path)

    best = float("inf")
    for epoch in range(args.epochs):
        running = 0.0
        for _ in range(steps):
            x = make_batch(train, pool_contact, pool_rest, args.batch_size,
                           args.contact_frac, scale, args.device, rng)
            recon, _ = model(x)
            loss = weighted_mse(recon, x, args.contact_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        val_mse, val_mse_contact = evaluate(model, val, val_contact_mask, scale, args.device)
        print(f"epoch {epoch:3d}  train {running / steps:.6f}  "
              f"val {val_mse:.6f}  val_contact {val_mse_contact:.6f}")
        if val_mse < best:
            best = val_mse
            save(os.path.join(args.out_dir, "ae_best.pth"), epoch, val_mse)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save(os.path.join(args.out_dir, f"ae_epoch{epoch}.pth"), epoch, val_mse)

    print(f"\ndone. best val MSE {best:.6f}. To train the raw_tactile baseline:")
    print(f"  FORGE_TACTILE_ENCODER_CKPT={os.path.join(args.out_dir, 'ae_best.pth')} \\")
    print(f"  FORGE_TACTILE_ENCODER_DIM={2 * args.per_hand_dim} \\")
    print("  bash run_pegpickplace_baseline_raw_tactile.sh")


if __name__ == "__main__":
    main()
