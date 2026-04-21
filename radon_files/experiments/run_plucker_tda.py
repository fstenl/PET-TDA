"""TDA on Plücker coordinate representation of LORs.

Phantom -> forward-project -> sample events -> extract LOR endpoints ->
Plücker coordinates + hybrid distance matrix -> Vietoris-Rips persistence ->
Betti and Euler-characteristic curves -> inter-frame Wasserstein distance matrix.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils.device import get_device
from src.phantom.generator import load_xcat
from src.simulation.scanner import get_mct_projector
from src.simulation.listmode import (
    sample_events_from_sinogram,
    get_lor_endpoints,
    mash_sinogram_indices,
)
from src.tda.persistence import compute_persistence_plucker, compute_frame_diagrams_plucker
from src.tda.distances import compute_distance_matrix
from src.tda.vectorization import diagrams_to_betti_curves, euler_characteristic_curve
from src.utils.visualization import (
    plot_phantom_frame,
    plot_persistence_diagram,
    plot_betti_curves,
    plot_euler_characteristic,
    plot_distance_matrix,
)

# --- Config ---
device = get_device()
num_frames = 10
num_events_per_frame = 35000
alpha = 1.0
n_samples = 10000
hom_dim = 1
# Sinogram-index mashing: (radial, view, plane[, tof]) integer factors.
# Collapses adjacent LOR bins, then one representative LOR per coarse cell
# is passed to the Plücker pipeline. Set all to 1 to disable.
mash = (4, 8, 2)
xcat_path = '../data/respiratory_only.npy'

print(f"Loading XCAT phantom from {xcat_path}")
full = load_xcat(xcat_path, device=device)
phantom = full[:num_frames, 300:300 + 109, :, :]
proj = get_mct_projector(device=device, img_shape=tuple(phantom.shape[1:]), tof=False)
beta = 1.0 / proj.lor_descriptor.scanner.__getattribute__('radius')

print(f"Phantom shape: {phantom.shape}")

# --- Per-frame event sampling ---
print(f"Sampling {num_events_per_frame} events per frame ({num_frames} frames) ...")
event_frames = []
lor_endpoints = []
for f in range(num_frames):
    sino = proj(phantom[f])
    indices = sample_events_from_sinogram(sino, num_events_per_frame)
    n_raw = indices.shape[0]
    if any(m > 1 for m in mash):
        indices = mash_sinogram_indices(
            indices, proj,
            mash_radial=mash[0], mash_view=mash[1], mash_plane=mash[2],
        )
    p1, p2, _ = get_lor_endpoints(indices, proj)
    event_frames.append(indices)
    lor_endpoints.append((p1, p2))
    print(f"  frame {f + 1}/{num_frames}: {n_raw} events -> {p1.shape[0]} unique coarse LORs")


# --- Inter-frame Wasserstein distance matrix ---
print("Computing Plücker persistence diagrams for all frames ...")
all_diagrams = compute_frame_diagrams_plucker(
    event_frames, proj, alpha=alpha, beta=beta, max_dim=hom_dim, n_samples=n_samples
)

print("Computing inter-frame Wasserstein distance matrix ...")
dist_matrix = compute_distance_matrix(all_diagrams, method='wasserstein', hom_dim=hom_dim)
print(f"Distance matrix shape: {dist_matrix.shape}  range: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]")

plot_distance_matrix(
    dist_matrix,
    title=f"Inter-frame Wasserstein distance (Plücker, H{hom_dim})",
    labels=list(range(num_frames)),
    cbar_label="Wasserstein distance",
)
print("Saving plot image to distance_matrix_10mm.png...")
plt.savefig("distance_matrix_10mm.png", dpi=300, bbox_inches="tight")
    
print("Done!")
