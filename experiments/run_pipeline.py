import torch
import matplotlib.pyplot as plt

from src.phantom.generator import generate_moving_sphere
from src.simulation.scanner import get_mini_projector
from src.simulation.listmode import sample_events, indices_to_sinogram
from src.representation.mlem import reconstruct_mlem
from src.tda.persistence import compute_persistence_volume
from src.tda.distances import compute_distance_matrix
from src.utils.visualization import (
    plot_phantom_frame,
    plot_sinogram,
    plot_reconstruction,
    plot_persistence_diagram,
    plot_distance_matrix,
)

device = 'cpu'
img_shape = (20, 80, 80)
num_phases = 5
num_cycles = 2
num_events = 25000
num_iterations = 4

# --- Phantom ---
phantom = generate_moving_sphere(
    num_phases=num_phases,
    num_cycles=num_cycles,
    img_shape=img_shape,
    device=device,
)
print(f"Phantom shape: {phantom.shape}")
plot_phantom_frame(phantom, frame=0)

# --- Scanner ---
proj = get_mini_projector(device=device, img_shape=img_shape, tof=True)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
proj.show_geometry(ax)
plt.show()

# --- Reconstruct all frames ---
reconstructions = []

for frame_idx in range(phantom.shape[0]):
    print(f"Processing frame {frame_idx + 1} / {phantom.shape[0]}")

    frame = phantom[frame_idx]
    indices = sample_events(frame, proj, num_events=num_events)
    sinogram = indices_to_sinogram(indices, proj)
    image = reconstruct_mlem(sinogram, proj, num_iterations=num_iterations, verbose=False)
    reconstructions.append(image)

reconstructions = torch.stack(reconstructions)
print(f"Reconstructions shape: {reconstructions.shape}")

# --- Visualize a few reconstructions ---
plot_sinogram(sinogram)
plot_reconstruction(reconstructions[0])
plot_reconstruction(reconstructions[num_phases // 2])

# --- TDA on reconstructed volumes ---
print("Computing persistence diagrams...")
diagrams = []
for i, image in enumerate(reconstructions):
    print(f"Persistence {i + 1} / {len(reconstructions)}", end="\r")
    dgm = compute_persistence_volume(image, max_dim=1)
    diagrams.append(dgm)
print()

plot_persistence_diagram(diagrams[0], title="Persistence diagram - frame 0")
plot_persistence_diagram(diagrams[num_phases // 2], title=f"Persistence diagram - frame {num_phases // 2}")

# --- Distance matrix ---
print("Computing distance matrix...")
dist_matrix = compute_distance_matrix(diagrams, method='wasserstein', hom_dim=1)
plot_distance_matrix(dist_matrix, title="Wasserstein distance matrix (H1)")
print("Done!")