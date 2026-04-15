# %%
%matplotlib widget

import array_api_compat.torch as xp
import numpy as np
import matplotlib.pyplot as plt
import parallelproj

from radon_files.scanner import get_mCT_scanner, sample_events_2, get_event_coords
from radon_files.visualization import (
    visualize_image,
    visualize_image_jupyter,
    visualize_sinogram_tof,
    plot_3d,
)
from radon_files.reconstruction import histogram_events, run_mlem
from radon_files.persistence import witness_persistence, diagrams_to_betti_curves, euler_characteristic_curve

# %%
dev = "cpu"
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

events = sample_events_2(image_to_project, proj, 35000)

dists = get_event_coords(events, proj)

plot_3d(dists, 10000)
print("Event coordinates shape:", dists.shape)

# %% Persistent homology via witness complex
diagrams = witness_persistence(
    dists,
    landmark_ratio=0.10,   # 10% of points as landmarks
    landmark_method="maxmin",  # farthest-point sampling for even spatial coverage
    max_dim=1,             # H0 (connected components) + H1 (loops)
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

# %%
measured_sino = histogram_events(events, proj)
print("Measured sinogram stats:", measured_sino.shape, measured_sino.sum().item())

reconstruction = run_mlem(proj, measured_sino, num_iter=3)
recon_np = parallelproj.to_numpy_array(reconstruction.unsqueeze(0))
visualize_image(recon_np)
print("MLEM reconstruction min/max:", reconstruction.min().item(), reconstruction.max().item())

# %%
