"""TDA directly on TOF event point clouds (no reconstruction).

Phantom -> forward-project -> sample TOF events -> localise to TOF-bin
centres -> point cloud -> witness/Vietoris-Rips persistence ->
Betti and Euler-characteristic curves -> inter-frame Betti-curve L2
distance matrix.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils.device import get_device
from src.phantom.generator import generate_moving_sphere, load_xcat
from src.simulation.scanner import get_mini_projector, get_mct_projector
from src.simulation.listmode import (
    sample_events_from_sinogram,
    get_lor_endpoints,
)
from src.representation.pointcloud import localize_events
from src.tda.persistence import compute_persistence, compute_frame_pcfs
from src.tda.vectorization import (
    diagrams_to_betti_curves,
    euler_characteristic_curve,
)
from src.tda.distances import compute_pcf_distance_matrix
from src.utils.visualization import (
    plot_phantom_frame,
    plot_pointcloud_3d,
    plot_persistence_diagram,
    plot_betti_curves,
    plot_euler_characteristic,
    plot_distance_matrix,
)

# --- Config ---
device = get_device()
num_frames = 10
num_events_per_frame = 25000
gating_method = 'masspcf'     # 'masspcf' (batched), 'witness', or 'ripser'
xcat_path = '../data/respiratory_only.npy'

# --- Phantom ---
if os.path.exists(xcat_path):
    print(f"Loading XCAT phantom from {xcat_path}")
    full = load_xcat(xcat_path, device=device)
    phantom = full[:num_frames, 300:300 + 109, :, :]
    proj = get_mct_projector(device=device, img_shape=tuple(phantom.shape[1:]), tof=True)
else:
    print(f"XCAT phantom not found at {xcat_path}; using synthetic moving sphere")
    img_shape = (20, 80, 80)
    phantom = generate_moving_sphere(
        num_phases=num_frames,
        num_cycles=1,
        img_shape=img_shape,
        device=device,
    )
    proj = get_mini_projector(device=device, img_shape=img_shape, tof=True)

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

# --- Inspect the first frame ---
plot_pointcloud_3d(point_clouds[0])

# --- Single-frame persistence via witness complex ---
print("Computing witness-complex persistence on frame 0 ...")
diagrams = compute_persistence(
    point_clouds[0],
    subsample='voxel',
    subsample_kwargs={'voxel_size': 5.0},
    method='witness',
    method_kwargs={
        'landmark_ratio': 0.10,
        'landmark_method': 'farthest',
        'max_dim': 1,
    },
)

# --- Vectorise and plot ---
betti_curves = diagrams_to_betti_curves(diagrams)
t_ecc, chi = euler_characteristic_curve(diagrams)

plot_persistence_diagram(diagrams, title="Persistence diagram - frame 0 (TOF)")
plot_betti_curves(betti_curves, title="Betti curves - frame 0 (TOF)")
plot_euler_characteristic(t_ecc, chi, title="Euler characteristic - frame 0 (TOF)")

# --- Inter-frame Betti-curve distance matrix ---
print(f"Computing per-frame Betti curve PCFs (method={gating_method!r}) ...")
pcf_tensors = compute_frame_pcfs(
    event_frames,
    proj,
    method=gating_method,
    max_dim=1,
    subsample='voxel',
    subsample_kwargs={'voxel_size': 5.0},
)

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
print("Done!")
