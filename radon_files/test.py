# %%
%matplotlib widget

import array_api_compat.torch as xp
import numpy as np
import matplotlib.pyplot as plt
import parallelproj
from gudhi.subsampling import choose_n_farthest_points
from radon_files.scanner import get_mCT_scanner, sample_events_2, get_event_coords
from radon_files.visualization import (
    visualize_image,
    visualize_image_jupyter,
    visualize_sinogram_tof,
    plot_3d,
)
from radon_files.reconstruction import histogram_events, run_mlem
from radon_files.persistence import (
    witness_persistence,
    alpha_persistence,          
    bootstrapped_persistence,
    partitioned_persistence,   
    diagrams_to_betti_curves,
    euler_characteristic_curve,
)

# %%
dev = "cpu"
print(f"Using device: {dev}")

proj = get_mCT_scanner(xp, dev)

xcat = np.load("/home/mamostudent3/data/respiratory_only.npy")
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
    axes[dim].set_title(f"Witness Complex - Betti curve β_{dim}(t)")
    axes[dim].set_xlabel("Filtration (mm)")
    axes[dim].set_ylabel(f"β_{dim}")

axes[2].plot(t_ecc, chi)
axes[2].set_title("Witness Complex - Euler characteristic χ(t)")
axes[2].set_xlabel("Filtration (mm)")
axes[2].set_ylabel("χ")
fig.tight_layout()
plt.show()

# %% Pure Alpha — ground truth on full point cloud
print("Computing pure Alpha complex on full point cloud...")
diag_pure = alpha_persistence(dists.numpy(), max_dim=1)
betti_pure = diagrams_to_betti_curves(diag_pure)
t_ecc_pure, chi_pure = euler_characteristic_curve(diag_pure)

# MaxMin + Alpha averaged runs — stability check

# Uses MaxMin to pick 4,000 landmark points from the full 35,000
# Runs Alpha complex on just those 4,000
print("Computing MaxMin + Alpha averaged runs...")
N_runs = 10
all_diagrams = []
for i in range(N_runs):
    print(f"Run {i+1}/{N_runs}")
    pts_sub = np.array(
        choose_n_farthest_points(
            points=dists.numpy().tolist(),
            nb_points=4000
        )
    )
    diag = alpha_persistence(pts_sub, max_dim=1)
    all_diagrams.append(diag)

all_betti = [diagrams_to_betti_curves(diag) for diag in all_diagrams]

# %% Plot both together for comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for dim in range(2):  # H0 and H1
    t = all_betti[0][dim][0]

    # averaged runs + std band
    curves = np.array([b[dim][1] for b in all_betti])
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)

    axes[dim].plot(t, mean_curve, label="MaxMin+Alpha avg", color="steelblue")
    axes[dim].fill_between(t, mean_curve - std_curve, mean_curve + std_curve,
                           alpha=0.3, color="steelblue", label="±1 std")

    # pure alpha ground truth — interpolate onto same t axis
    t_pure, beta_pure = betti_pure[dim]
    beta_interp = np.interp(t, t_pure, beta_pure)
    axes[dim].plot(t, beta_interp, label="Pure Alpha (ground truth)",
                   color="crimson", linestyle="--", linewidth=1.5)

    axes[dim].set_title(f"β_{dim}(t) — Alpha comparison")
    axes[dim].set_xlabel("Filtration (mm)")
    axes[dim].set_ylabel(f"β_{dim}")
    axes[dim].legend(fontsize=8)

# Euler characteristic panel
t = all_betti[0][0][0]
chi_avg = np.zeros_like(t, dtype=float)
for dim in range(len(all_betti[0])):
    mean_curve = np.array([b[dim][1] for b in all_betti]).mean(axis=0)
    chi_avg += ((-1) ** dim) * mean_curve

# interpolate pure alpha chi onto same t axis
chi_pure_interp = np.interp(t, t_ecc_pure, chi_pure)

axes[2].plot(t, chi_avg, label="MaxMin+Alpha avg", color="steelblue")
axes[2].plot(t, chi_pure_interp, label="Pure Alpha (ground truth)",
             color="crimson", linestyle="--", linewidth=1.5)
axes[2].set_title("Euler characteristic χ(t)")
axes[2].set_xlabel("Filtration (mm)")
axes[2].set_ylabel("χ")
axes[2].legend(fontsize=8)

fig.tight_layout()
plt.savefig("alpha_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# %% Bootstrap average approach (stable across subsamples)
print("Computing persistent homology via bootstrapped subsampling...")
diagrams_boot = bootstrapped_persistence(dists.numpy(), n_samples=5, subsample_size=2000, max_dim=1)
betti_curves_boot = diagrams_to_betti_curves(diagrams_boot)
t_ecc_boot, chi_boot = euler_characteristic_curve(diagrams_boot)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(betti_curves_boot[0][0], betti_curves_boot[0][1])
axes[0].set_title("Bootstrap - β₀ (components)")
axes[0].set_xlabel("Filtration (mm)")
axes[0].set_ylabel("β₀")

axes[1].plot(betti_curves_boot[1][0], betti_curves_boot[1][1])
axes[1].set_title("Bootstrap - β₁ (loops)")
axes[1].set_xlabel("Filtration (mm)")
axes[1].set_ylabel("β₁")

axes[2].plot(t_ecc_boot, chi_boot)
axes[2].set_title("Bootstrap - Euler characteristic χ(t)")
axes[2].set_xlabel("Filtration (mm)")
axes[2].set_ylabel("χ")

fig.tight_layout()
plt.savefig("bootstrap_betti_curves.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Spatial partition approach (preserves local structure)
print("Computing persistent homology via spatial partitioning...")
# This method divides the point cloud into spatial regions, computes persistence in each, and then aggregates results.
diagrams = partitioned_persistence(dists, n_partitions=4, max_dim=1)

# Betti curves — β_k(t) counts features alive at filtration value t (mm)
betti_curves = diagrams_to_betti_curves(diagrams)
t_ecc, chi = euler_characteristic_curve(diagrams)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for dim in range(len(diagrams)):
    t, beta = betti_curves[dim]
    axes[dim].plot(t, beta)
    axes[dim].set_title(f"Spatial Partition - Betti curve β_{dim}(t)")
    axes[dim].set_xlabel("Filtration (mm)")
    axes[dim].set_ylabel(f"β_{dim}")

axes[2].plot(t_ecc, chi)
axes[2].set_title("Spatial Partition - Euler characteristic χ(t)")
axes[2].set_xlabel("Filtration (mm)")
axes[2].set_ylabel("χ")
fig.tight_layout()
plt.savefig("partitioned_betti_curves.png", dpi=300, bbox_inches="tight")

# %%
measured_sino = histogram_events(events, proj)
print("Measured sinogram stats:", measured_sino.shape, measured_sino.sum().item())
                                                                                                                                                                                                                                                                                                            
reconstruction = run_mlem(proj, measured_sino, num_iter=3)
recon_np = parallelproj.to_numpy_array(reconstruction.unsqueeze(0))
visualize_image(recon_np)
print("MLEM reconstruction min/max:", reconstruction.min().item(), reconstruction.max().item())

# %%
