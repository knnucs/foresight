"""
Shared code for the fall-prediction model: skeleton preprocessing + the TCN.
Used by training now, and later by the webcam demo / Raspberry Pi code, so
every stage turns MediaPipe output into model input in exactly the same way.

Per frame, MediaPipe gives 132 numbers (33 landmarks x [x, y, z, visibility]).
The model sees 166 numbers per frame:
  66 = 33 landmarks x (x, y), centered on the hips and scaled by torso length
  66 = how far each landmark moved since the previous frame (velocity)
  33 = landmark visibility
   1 = whether a pose was actually detected in this frame (0 = gap-filled)
"""
import numpy as np
import torch
import torch.nn as nn

NUM_LM = 33
FEAT_DIM = NUM_LM * 5 + 1
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24

_FLIP_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
               (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
FLIP_PERM = np.arange(NUM_LM)
for _a, _b in _FLIP_PAIRS:
    FLIP_PERM[_a], FLIP_PERM[_b] = _b, _a


def fill_missing(poses, detected):
    """Frames with no detected pose copy the nearest earlier pose (the start copies the first one found)."""
    poses = np.array(poses, dtype=np.float32, copy=True)
    found = np.where(detected)[0]
    if len(found) == 0:
        return poses
    last = found[0]
    for i in range(len(poses)):
        if detected[i]:
            last = i
        else:
            poses[i] = poses[last]
    return poses


def window_features(poses, detected, flip=False):
    """poses: (T, 132) gap-filled MediaPipe output, detected: (T,) bool -> (T, FEAT_DIM) float32."""
    lm = np.asarray(poses, dtype=np.float32).reshape(-1, NUM_LM, 4)
    xy = lm[:, :, :2].copy()
    vis = lm[:, :, 3].copy()
    if flip:
        xy[:, :, 0] = 1.0 - xy[:, :, 0]
        xy = xy[:, FLIP_PERM]
        vis = vis[:, FLIP_PERM]
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    sh = (xy[:, L_SH] + xy[:, R_SH]) / 2
    center = hip.mean(axis=0)
    scale = max(float(np.median(np.linalg.norm(sh - hip, axis=1))), 1e-2)
    xy = np.clip((xy - center) / scale, -10.0, 10.0)
    vel = np.zeros_like(xy)
    vel[1:] = np.clip(xy[1:] - xy[:-1], -3.0, 3.0)
    det = np.asarray(detected, dtype=np.float32)[:, None]
    return np.concatenate([xy.reshape(len(xy), -1), vel.reshape(len(xy), -1), vis, det],
                          axis=1).astype(np.float32)


class CausalBlock(nn.Module):
    """Dilated convolution that only looks at past frames, plus a skip connection."""
    def __init__(self, c_in, c_out, dilation, dropout):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=3, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = nn.functional.pad(x, (self.pad, 0))
        return self.drop(self.act(self.conv(y))) + self.skip(x)


class FallTCN(nn.Module):
    """Input (batch, 15 frames, FEAT_DIM) -> one risk logit per window, read at the newest frame.
    Dilations 1, 2, 4 give a 15-frame (1 s at 15 fps) receptive field."""
    def __init__(self, in_dim=FEAT_DIM, channels=64, dropout=0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            CausalBlock(in_dim, channels, 1, dropout),
            CausalBlock(channels, channels, 2, dropout),
            CausalBlock(channels, channels, 4, dropout),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        h = self.blocks(x.permute(0, 2, 1))
        return self.head(h[:, :, -1]).squeeze(-1)
