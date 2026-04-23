"""TDA on Plücker coordinate representation of LORs.

Phantom -> forward-project -> sample events -> extract LOR endpoints ->
Poisson-disk subsampling -> Plücker coordinates + hybrid distance matrix -> 
Vietoris-Rips persistence -> Betti and Euler-characteristic curves -> 
inter-frame Wasserstein distance matrix.

"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from src.utils.device import get_device
from src.phantom.generator import load_xcat
from src.simulation.scanner import get_mct_projector
from src.simulation.listmode import (
    sample_events_from_sinogram,
    get_lor_endpoints,
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

# Import your subsampling function
# (Adjust this import path if subsampling.py is in a different directory)
from src.representation.subsampling import poisson_disk 

# --- Config ---
device = get_device()
num_frames = 20
num_events_per_frame = 2000
alpha = 1.0
n_samples = 10000
hom_dim = 1

# Poisson-disk thinning configuration
# This is the minimum Euclidean distance between LORs in the 6D (p1, p2) space.
# Increase this value to heavily reduce the point cloud size.
min_distance = 30.0 
xcat_path = '../../../data/respiratory_only.npy'

print(f"Loading XCAT phantom from {xcat_path}")
full = load_xcat(xcat_path, device=device)
phantom = full[:num_frames, 300:300 + 109, :, :]
proj = get_mct_projector(device=device, img_shape=tuple(phantom.shape[1:]), tof=False)
beta = 1.0 / proj.lor_descriptor.scanner.__getattribute__('radius')

print(f"Phantom shape: {phantom.shape}")

# --- Per-frame event sampling & Subsampling ---
print(f"Sampling {num_events_per_frame} events per frame ({num_frames} frames) ...")
event_frames = []
lor_endpoints = []

for f in range(num_frames):
    sino = proj(phantom[f])
    
    # Sample dense events
    indices = sample_events_from_sinogram(sino, num_events_per_frame)
    n_raw = indices.shape[0]
    
    # Extract geometry to allow spatial subsampling
    p1, p2, _ = get_lor_endpoints(indices, proj)
    
    # Concatenate p1 and p2 into 6D arrays for geometric distance 
    points_6d = torch.cat([p1, p2], dim=1).cpu().numpy()
    
    # Apply Poisson-disk subsampling
    #thinned_points = poisson_disk(points_6d, min_distance=min_distance)
    thinned_points = points_6d  
    
    # Map the thinned coordinates back to their original tensor indices
    tree = cKDTree(points_6d)
    _, kept_rows = tree.query(thinned_points, k=1)
    
    # Filter the indices and endpoints using the kept rows
    indices_thinned = indices[kept_rows]
    p1_thinned = p1[kept_rows]
    p2_thinned = p2[kept_rows]
    
    event_frames.append(indices_thinned)
    lor_endpoints.append((p1_thinned, p2_thinned))
    print(f"  frame {f + 1}/{num_frames}: {n_raw} events -> {p1_thinned.shape[0]} thinned LORs")

# --- Inter-frame Wasserstein distance matrix ---
print("Computing Plücker persistence diagrams for all frames ...")
all_diagrams = compute_frame_diagrams_plucker(
    event_frames, proj, alpha=alpha, beta=beta, max_dim=hom_dim
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

print("Saving plot image to distance_matrix_plucker.png...")
plt.savefig("distance_matrix_plucker.png", dpi=300, bbox_inches="tight")

print("Done!")