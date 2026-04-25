"""Study of how MLEM iteration count affects TDA clustering of reconstructed volumes."""

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.phantom.generator import generate_pulsing_sphere
from src.simulation.scanner import get_mini_projector
from src.simulation.listmode import sample_events, indices_to_sinogram
from src.representation.mlem import reconstruct_mlem
from src.tda.persistence import compute_persistence_volume
from src.tda.distances import compute_distance_matrix
from src.tda.clustering import cluster_distance_matrix, dice_score
from src.utils.visualization2 import plot_distance_matrix

device = 'cpu'
img_shape = (8, 40, 40)
num_phases = 10
num_cycles = 2
num_events = 25000
min_persistence = 0.003
iterations_to_test = [3, 5, 10, 20]

# --- Ground truth ---
ground_truth = np.array(list(range(num_phases)) * num_cycles)
print(f"Ground truth: {ground_truth}")

# --- Phantom ---
phantom = generate_pulsing_sphere(
    num_phases=num_phases,
    num_cycles=num_cycles,
    img_shape=img_shape,
    device=device,
)
print(f"Phantom shape: {phantom.shape}")

# --- Scanner ---
proj = get_mini_projector(device=device, img_shape=img_shape, tof=True)

# --- Sample sinograms once for all frames ---
print("Sampling sinograms...")
sinograms = []
for frame_idx in range(phantom.shape[0]):
    print(f"  Frame {frame_idx + 1} / {phantom.shape[0]}", end="\r")
    indices = sample_events(phantom[frame_idx], proj, num_events=num_events)
    sinograms.append(indices_to_sinogram(indices, proj))
print()

# --- Main experiment loop ---
dice_scores = []

for num_iterations in iterations_to_test:
    print(f"Running MLEM with {num_iterations} iterations...")

    # Reconstruct all frames
    reconstructions = []
    for frame_idx in range(phantom.shape[0]):
        print(f"  Reconstructing frame {frame_idx + 1} / {phantom.shape[0]}", end="\r")
        image = reconstruct_mlem(
            sinograms[frame_idx], proj,
            num_iterations=num_iterations,
            verbose=False,
        )
        reconstructions.append(image)
    print()

    # Compute persistence diagrams
    print("  Computing persistence diagrams...")
    diagrams = []
    for image in reconstructions:
        dgm = compute_persistence_volume(image, max_dim=1, min_persistence=min_persistence)
        diagrams.append(dgm)

    # Compute distance matrix
    print("  Computing distance matrix...")
    dist_matrix = compute_distance_matrix(diagrams, method='wasserstein', hom_dim=1)
    plot_distance_matrix(dist_matrix, title=f"Wasserstein distance matrix ({num_iterations} iterations)")

    # Cluster and evaluate
    labels = cluster_distance_matrix(dist_matrix, num_clusters=num_phases)
    score = dice_score(labels, ground_truth, num_clusters=num_phases)
    dice_scores.append(score)
    print(f"  Dice score: {score:.3f}")

# --- Plot results ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(iterations_to_test, dice_scores, marker='o')
ax.set_xlabel("MLEM iterations")
ax.set_ylabel("Dice score")
ax.set_title("TDA clustering quality vs MLEM iterations")
ax.set_ylim(0, 1)
ax.grid(True)
plt.tight_layout()
plt.show()