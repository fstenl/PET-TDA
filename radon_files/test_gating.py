# %%
%matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
import array_api_compat.torch as xp

from radon_files.scanner import get_mCT_scanner, sample_events_2
from radon_files.gating import window_events, compute_window_pcfs, compute_distance_matrix

# %%
dev = "cpu"
print(f"Using device: {dev}")

proj = get_mCT_scanner(xp, dev)

xcat = np.load("data/respiratory_only.npy")
print(f"Phantom shape: {xcat.shape}")  # expected: (20, z, y, x)

n_resp_frames = 20      # all frames — 2 respiratory cycles of 10 frames each
n_events_per_frame = 5_000

# %%
# Simulate list-mode acquisition: sample events from each respiratory frame
# and concatenate into a single ordered event stream.
print(f"Sampling {n_events_per_frame} events from each of {n_resp_frames} frames ...")
all_events_list = []
frame_labels_list = []

for f in range(n_resp_frames):
    image = xp.asarray(xcat[f, 300:300 + 109, :, :])
    events = sample_events_2(image, proj, n_events_per_frame)
    all_events_list.append(np.asarray(events))
    frame_labels_list.append(np.full(n_events_per_frame, f, dtype=int))

all_events = np.concatenate(all_events_list)
frame_labels = np.concatenate(frame_labels_list)
print(f"Total events in stream: {len(all_events)}")

# %%
# Window the event stream — one window per respiratory frame
window_size = n_events_per_frame
windows = window_events(all_events, window_size)
gt_frame = [int(frame_labels[w * window_size]) for w in range(len(windows))]
print(f"{len(windows)} windows of {window_size} events")
print(f"Ground-truth frames: {gt_frame}")

# %%
# Compute per-window Betti curve PCFs via masspcf
print("Computing per-window Betti curve PCFs ...")
pcf_tensors = compute_window_pcfs(
    windows,
    proj,
    max_dim=1,
)
print(f"{len(pcf_tensors)} PcfTensor(s), each shape {pcf_tensors[0].shape}")

# %%
# Compute all-pairs L2 distance matrix between windows
print("Computing inter-window distance matrix ...")
D = compute_distance_matrix(pcf_tensors)
print(f"Shape: {D.shape}  range: [{D.min():.4f}, {D.max():.4f}]")

# %%
# Visualise the distance matrix.
# With 2 respiratory cycles of 10 frames each, windows w and w+10 represent
# the same respiratory state — expect small off-diagonal distances at lag 10.
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(D, cmap="viridis")
ax.set_title("Inter-window Betti curve L2 distance matrix (H₀ + H₁)")
ax.set_xlabel("Window (ground-truth frame)")
ax.set_ylabel("Window (ground-truth frame)")
ticks = list(range(len(windows)))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels([str(gt_frame[i]) for i in ticks], fontsize=8)
ax.set_yticklabels([str(gt_frame[i]) for i in ticks], fontsize=8)
plt.colorbar(im, ax=ax, label="L2 distance")
fig.tight_layout()
plt.show()

# %%
