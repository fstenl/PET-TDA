# %%
%matplotlib widget

import array_api_compat.cupy as xp
import cupy
import numpy as np
import matplotlib.pyplot as plt
import parallelproj

from scanner import get_mCT_scanner, sample_events_from_sinogram, get_event_coords
from visualization import (
    visualize_image,
    visualize_image_jupyter,
    visualize_sinogram_tof,
    plot_3d,
)
from persistence import (
    compute_persistence,
    diagrams_to_betti_curves,
    euler_characteristic_curve,
)

# %%
dev = cupy.cuda.Device(0)
print(f"Using device: {dev}")

proj = get_mCT_scanner(xp, dev)

xcat = np.load("data/respiratory_only.npy")
print(f"Image shape: {xcat.shape}")
visualize_image_jupyter(xcat)
image_to_project = xp.asarray(xcat[0, 300:300+109, :, :])  # Take the first frame for projection

print(f"In shape: {proj.in_shape}")
print(f"Out Shape: {proj.out_shape}")

print("Number of LORs:", proj.lor_descriptor.num_rad * proj.lor_descriptor.num_views * proj.lor_descriptor.num_planes)
print("Number of tof bins:", proj.tof_parameters)
forward_proj = proj(image_to_project)
visualize_sinogram_tof(forward_proj, show=True)

events = sample_events_from_sinogram(forward_proj, 500)

dists = get_event_coords(events, proj)

plot_3d(dists, 500)
print("Event coordinates shape:", dists.shape)

# %% Persistent homology — pluggable pipeline (subsample → backend → vectorise)
diagrams = compute_persistence(
    dists,
    # subsample="voxel", subsample_kwargs={"voxel_size": 3.0},   # try me
    method="witness",
    method_kwargs={
        "landmark_ratio": 0.10,
        "landmark_method": "farthest",
        "max_dim": 1,
    },
)

# Betti curves — β_k(t) counts features alive at filtration value t (mm)
betti_curves = diagrams_to_betti_curves(diagrams)
t_ecc, chi = euler_characteristic_curve(diagrams)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for dim in range(len(diagrams)):
    t, beta = betti_curves[dim]
    axes[dim].plot(t, beta)
    axes[dim].set_title(f"Betti curve β_{dim}(t)")
    axes[dim].set_xlabel("Filtration (mm)")
    axes[dim].set_ylabel(f"β_{dim}")

axes[2].plot(t_ecc, chi)
axes[2].set_title("Euler characteristic χ(t)")
axes[2].set_xlabel("Filtration (mm)")
axes[2].set_ylabel("χ")
fig.tight_layout()
plt.show()

