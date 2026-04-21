"""TDA directly on TOF event point clouds.

Phantom -> forward-project -> sample TOF events -> localise to TOF-bin
centres -> point cloud -> witness/Vietoris-Rips persistence ->
Betti and Euler-characteristic curves -> inter-frame Betti-curve L2
distance matrix.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.utils.device import get_device
from src.phantom.generator import load_xcat
from src.simulation.scanner import get_mct_projector
from src.simulation.listmode import (
    sample_events_from_sinogram,
    get_lor_endpoints,
)
from src.representation.pointcloud import localize_events
from src.tda.persistence import compute_frame_pcfs
from src.tda.distances import compute_pcf_distance_matrix
from src.utils.visualization import (
    plot_phantom_frame,
    plot_distance_matrix,
)

# --- Config ---
device = get_device()
num_frames = 20
num_events_per_frame = 35000
gating_method = 'masspcf'     # 'masspcf' (batched), 'witness', or 'ripser'
xcat_path = '../../../data/respiratory_only.npy'

# --- Phantom ---

print(f"Loading XCAT phantom from {xcat_path}")
full = load_xcat(xcat_path, device=device)
phantom = full[:num_frames, 300:300 + 109, :, :]
proj = get_mct_projector(device=device, img_shape=tuple(phantom.shape[1:]), tof=True)


print(f"Phantom shape: {phantom.shape}")
plot_phantom_frame(phantom, frame=0)

# --- Per-frame forward-projection + event sampling ---
print(f"Sampling {num_events_per_frame} TOF events per frame ({num_frames} frames) ...")
event_frames = []
point_clouds = []
for f in range(num_frames):
    sino = proj(phantom[f])
    indices = sample_events_from_sinogram(sino, num_events_per_frame)
    p1, p2, tof_bins = get_lor_endpoints(indices, proj)
    points = localize_events(p1, p2, tof_bins, proj)
    event_frames.append(indices)
    point_clouds.append(points)
    print(f"  frame {f + 1}/{num_frames}: {points.shape[0]} events")

# --- Inter-frame Betti-curve distance matrix ---
print(f"Computing per-frame Betti curve PCFs (method={gating_method!r}) ...")
pcf_tensors = compute_frame_pcfs(
    event_frames,
    proj,
    method=gating_method,
    max_dim=1,
    subsample='voxel',
    subsample_kwargs={'voxel_size': 10.0},
)

"""pcf_tensors = compute_frame_pcfs(
    event_frames,
    proj,
    method=gating_method,
    max_dim=1,
    method_kwargs={},
    subsample="farthest",              # Switch away from voxel
    subsample_kwargs={"n": 2500},      # Keep an exact, stable number of landmarks
)"""

print("Computing inter-frame L2 distance matrix ...")
dist_matrix = compute_pcf_distance_matrix(pcf_tensors)
dist_matrix = np.asarray(dist_matrix)
print(f"Distance matrix shape: {dist_matrix.shape}  range: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]")

plot_distance_matrix(
    dist_matrix,
    title=f"Inter-frame Betti curve L2 distance ({gating_method})",
    labels=list(range(num_frames)),
    cbar_label="L2 distance",
)
print("Saving plot image to distance_matrix_10mm.png...")
plt.savefig("distance_matrix_10mm.png", dpi=300, bbox_inches="tight")

# --- Cyclic correlation ---
n = dist_matrix.shape[0]
phase_dist = np.array([
    [min(abs(i - j), n - abs(i - j)) for j in range(n)]
    for i in range(n)
], dtype=float)

upper = np.triu_indices(n, k=1)
rho, pval = spearmanr(dist_matrix[upper], phase_dist[upper])
print(f"Cyclic Spearman ρ = {rho:.4f}  (p = {pval:.2e})")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(phase_dist[upper], dist_matrix[upper], s=6, alpha=0.6)
ax.set_xlabel("Ground-truth phase distance (frames)")
ax.set_ylabel("Betti-curve L2 distance")
ax.set_title(f"Cyclic correlation  ρ = {rho:.3f}")
plt.tight_layout()
plt.savefig("cyclic_correlation.png", dpi=300, bbox_inches="tight")

print("Done!")
