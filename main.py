import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import matplotlib.pyplot as plt
import array_api_compat.torch as xp
# import array_api_compat.numpy as xp
# import array_api_compat.cupy as xp
from phantoms import primitives, trajectories, generator, noise
from scanner import scanner
from lines import representations, metrics
from tda import diagrams, distances, vectorization
import parallelproj


FIG_STYLE = {
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
}


def get_device():
    """Detect the best available compute device based on the array backend:
      - numpy  → 'cpu'
      - cupy   → cupy cuda device
      - torch  → 'cuda' if available, else 'cpu'
    """
    if "numpy" in xp.__name__:
        return "cpu"
    elif "cupy" in xp.__name__:
        import cupy
        return cupy.cuda.Device(0)
    elif "torch" in xp.__name__:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cpu"


def _compute_diagram_for_frame(frame, projector, num_lors, scanner_radius,
                               collect_sinogram=False, collect_adjoint=False):
    """Compute a persistence diagram from a single LOR sample of one frame.

    Core pipeline: LOR sampling -> Plucker coords -> distance matrix -> persistence.
    Designed for use with ThreadPoolExecutor.

    """
    result = scanner.generate_lors_from_image(
        frame, projector, num_lors=num_lors, show=False,
        return_sinogram=collect_sinogram, return_adjoint=collect_adjoint,
    )

    # Unpack: first two elements are always (p1, p2); extras follow in
    # the order [sinogram_img], [adjoint_img] depending on the flags.
    idx = 0
    p1, p2 = result[0], result[1]
    idx = 2
    sino_img = None
    adj_img = None
    if collect_sinogram:
        sino_img = result[idx]; idx += 1
    if collect_adjoint:
        adj_img = result[idx]; idx += 1

    plucker_coords = representations.to_canonical_plucker(p1, p2)
    dist_matrix = metrics.compute_hybrid_weighted_distance(
        plucker_coords, alpha=1, beta=1 / scanner_radius
    )
    dgm = diagrams.compute_persistence(dist_matrix, is_distance_matrix=True)

    extras = ()
    if collect_sinogram:
        extras += (sino_img,)
    if collect_adjoint:
        extras += (adj_img,)

    if extras:
        return (dgm,) + extras
    return dgm


def _plot_sinograms(sinogram_images: list[np.ndarray], title: str = "Sinograms", save_path: str | None = None) -> None:
    """Plot collected sinogram images on the main thread."""
    n = len(sinogram_images)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
        for i, img in enumerate(sinogram_images):
            ax = axes[i // cols][i % cols]
            ax.imshow(img, cmap="Greys_r", vmin=0)
            ax.set_title(f"Frame {i}")
            ax.tick_params(which='both', direction='in', top=True, right=True)
        for i in range(n, rows * cols):
            axes[i // cols][i % cols].axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


def _plot_adjoint_images(adjoint_images: list[np.ndarray], title: str = "Adjoint images", save_path: str | None = None) -> None:
    """Plot collected adjoint images on the main thread."""
    n = len(adjoint_images)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
        for i, img in enumerate(adjoint_images):
            ax = axes[i // cols][i % cols]
            ax.imshow(img, cmap="Greys_r", vmin=0)
            ax.set_title(f"Frame {i}")
            ax.tick_params(which='both', direction='in', top=True, right=True)
        for i in range(n, rows * cols):
            axes[i // cols][i % cols].axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


def _compute_avg_pairwise_distance_matrix(
    all_frame_diagrams: list[list],
    method: str = "wasserstein",
    hom_dim: int = 1,
) -> np.ndarray:
    """Compute an inter-frame distance matrix averaged over all sample pairs.

    For frames *i* and *j*, the entry ``D[i, j]`` is the mean of all pairwise
    distances between every sample of frame *i* and every sample of frame *j*.

    Args:
        all_frame_diagrams: ``all_frame_diagrams[i][s]`` is the persistence
            diagram for the *s*-th LOR sample of frame *i*.
        method: ``'wasserstein'`` or ``'bottleneck'``.
        hom_dim: Homology dimension used for the distance computation.

    Returns:
        A symmetric ``(num_frames, num_frames)`` numpy array of mean distances.
    """
    dist_func = (
        distances.compute_wasserstein_distance
        if method == "wasserstein"
        else distances.compute_bottleneck_distance
    )

    num_frames = len(all_frame_diagrams)

    # Collect all unique (i, j) pairs with i <= j
    jobs: list[tuple[int, int, object, object]] = []
    for i in range(num_frames):
        for j in range(i, num_frames):
            for dgm_a in all_frame_diagrams[i]:
                for dgm_b in all_frame_diagrams[j]:
                    jobs.append((i, j, dgm_a, dgm_b))

    # Run all distance computations in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(dist_func, dgm_a, dgm_b, hom_dim=hom_dim)
            for _, _, dgm_a, dgm_b in jobs
        ]

        # Accumulate results
        accum: dict[tuple[int, int], list[float]] = {}
        for (i, j, _, _), future in zip(jobs, futures):
            accum.setdefault((i, j), []).append(future.result())

    # Build symmetric matrix
    mat = np.zeros((num_frames, num_frames))
    for (i, j), vals in accum.items():
        mean_val = float(np.mean(vals))
        mat[i, j] = mean_val
        mat[j, i] = mean_val

    return mat


def _save_or_show(fig, save_path: str | None = None) -> None:
    """Save figure to *save_path* (closing it) or show it interactively."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def linear_motion_test(num_steps: int = 5, num_lors: int = 2000, save_dir: str | None = None) -> None:
    print("=" * 60)
    print("Running: Motion Analysis (Linear Trajectory)")
    print("=" * 60)

    # TDA configuration parameters
    hom_dim = 1  # Use 0 for clusters, 1 for loops
    method = 'wasserstein'  # Use 'wasserstein' or 'bottleneck'

    # PET scanner geometry and projection settings
    device = get_device()
    projector = scanner.get_mini_scanner(xp, device)
    grid_shape = projector.in_shape
    phantom_radius = grid_shape[1] // 8  # Set radius to an eighth of the Y dimension for good visibility
    scanner_radius = projector.lor_descriptor.scanner.__getattribute__('radius')
    print(f"Using device: {device} | scanner image shape: {grid_shape} | radius: {scanner_radius}")

    # Generate a linear trajectory path for the phantom movement
    path = trajectories.linear_trajectory(
        start=(-grid_shape[0]//2+phantom_radius, -grid_shape[1]//2+phantom_radius, 0),
        end=(grid_shape[0]//2-phantom_radius, grid_shape[1]//2-phantom_radius, 0),
        steps=num_steps
    )

    print("Generating moving sphere phantom frames...")
    # Produce the sequence of 3D image frames based on the trajectory
    frames = generator.generate_frames(
        primitives.create_sphere_phantom,
        trajectory=path,
        shape=grid_shape,
        radius=phantom_radius,
        device=device
    )
    print("Frames generated. Visualizing the phantom sequence...")
    generator.plot_frame_sequence(frames, save_path=os.path.join(save_dir, "linear_frames.png") if save_dir else None)

    # Parallel frame processing
    print(f"Processing {num_steps} frames in parallel...")
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    all_diagrams = [None] * num_steps
    sinogram_images = [None] * num_steps
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_idx = {
            executor.submit(
                _compute_diagram_for_frame, frame, projector, num_lors, scanner_radius,
                collect_sinogram=True
            ): i
            for i, frame in enumerate(frames)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            dgm, sino_img = future.result()
            all_diagrams[i] = dgm
            sinogram_images[i] = sino_img
            print(f"Frame {i + 1}/{num_steps} processed.")

    torch.set_num_threads(prev_threads)

    # Plot sinograms on the main thread
    _plot_sinograms(sinogram_images, title="Sinograms – Linear motion",
                    save_path=os.path.join(save_dir, "linear_sinograms.png") if save_dir else None)

    print("All frames processed. Computing inter-frame distances...")
    # Compute a distance matrix comparing every frame to every other frame
    dist_heatmap = distances.compute_all_pairs_distances(
        all_diagrams,
        method=method,
        hom_dim=hom_dim
    )

    print("Distance matrix computed. Visualizing heatmap...")
    method_name = method.capitalize()
    hom_name = f"H$_{{{hom_dim}}}$"
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(dist_heatmap, cmap='viridis', origin='lower')
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{method_name} ({hom_name}) – Linear motion")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Frame index")
        for i in range(num_steps):
            for j in range(num_steps):
                ax.text(j, i, f"{dist_heatmap[i, j]:.2f}",
                        ha="center", va="center", color="w", fontsize=8)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.tight_layout()
        _save_or_show(fig, os.path.join(save_dir, "linear_heatmap.png") if save_dir else None)


def deformation_test(num_steps: int = 5, num_lors: int = 2000, num_samples: int = 1, save_dir: str | None = None) -> None:
    """Run TDA on a sequence of phantoms deforming from a sphere to a cube.
    This test evaluates whether TDA can detect gradual shape changes in a phantom sequence, independent of motion. 
    The phantom smoothly transitions from a sphere to a cube.
    
    Each frame is processed in parallel to compute its persistence diagram from LOR samples, 
    and then inter-frame distances are computed and visualized as a heatmap.

    Args:
        num_samples: Number of independent LOR samples per frame.  When > 1
            the heatmap entry for frames *i* vs *j* is the mean over all
            pairwise sample distances.
    """
    print("=" * 60)
    print("Running: Deformation Test (Sphere → Cube)")
    print("=" * 60)

    # TDA configuration parameters
    hom_dim = 1  # Use 0 for clusters, 1 for loops
    method = 'wasserstein'  # Use 'wasserstein' or 'bottleneck'

    device = get_device()
    projector = scanner.get_mini_scanner(xp, device, show=False)
    grid_shape = projector.in_shape
    phantom_radius = grid_shape[2] // 2 # Set radius to half of the Z dimension for good visibility of deformation
    print(f"Using device: {device} | scanner image shape: {grid_shape} | radius: {projector.lor_descriptor.scanner.__getattribute__('radius')}")

    # Create a static trajectory to isolate shape changes from motion
    path = trajectories.static_trajectory(
        center=(0, 0, 0),
        steps=num_steps
    )

    # Define the deformation function: 0.0 (sphere) -> 1.0 (cube)
    def deformation_modifier(step):
        return {'morph_factor': step / (num_steps - 1)}

    print("Generating deformed phantom frames...")
    # Generate frames using the deformed primitive
    frames = generator.generate_frames(
        phantom_func=primitives.create_morphed_phantom,
        trajectory=path,
        shape_func=deformation_modifier,
        shape=grid_shape,
        radius=phantom_radius,
        device=device
    )

    print("Frames generated. Visualizing the phantom sequence...")
    # Visualize the phantom sequence to verify the deformation
    generator.plot_frame_sequence(frames, save_path=os.path.join(save_dir, "deformation_frames.png") if save_dir else None)
    generator.plot_frame_sequence_3d(frames, save_path=os.path.join(save_dir, "deformation_frames_3d.png") if save_dir else None)

    # Parallel frame processing
    scanner_radius = projector.lor_descriptor.scanner.__getattribute__('radius')
    total_jobs = num_steps * num_samples
    print(f"Processing {total_jobs} persistence diagrams ({num_samples} samples × {num_steps} frames) in parallel...")
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # Avoid thread oversubscription

    all_frame_diagrams: list[list] = [[] for _ in range(num_steps)]
    sinogram_images = [None] * num_steps
    adjoint_images = [None] * num_steps

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_idx = {}
        for i, frame in enumerate(frames):
            for s in range(num_samples):
                # Collect sinogram/adjoint only for the first sample of each frame
                collect = s == 0
                future = executor.submit(
                    _compute_diagram_for_frame, frame, projector, num_lors, scanner_radius,
                    collect_sinogram=collect, collect_adjoint=collect
                )
                future_to_idx[future] = (i, s)

        done_count = 0
        for future in as_completed(future_to_idx):
            i, s = future_to_idx[future]
            result = future.result()
            if s == 0:
                dgm, sino_img, adjoint_img = result
                sinogram_images[i] = sino_img
                adjoint_images[i] = adjoint_img
            else:
                dgm = result
            all_frame_diagrams[i].append(dgm)
            done_count += 1
            if done_count % num_samples == 0:
                print(f"  Progress: {done_count}/{total_jobs} diagrams computed.")

    torch.set_num_threads(prev_threads)


    print("All frames processed. Computing inter-frame distances...")
    # Compute averaged pairwise distance matrix
    dist_heatmap = _compute_avg_pairwise_distance_matrix(
        all_frame_diagrams, method=method, hom_dim=hom_dim
    )

    print("Distance matrix computed. Visualizing heatmap...")
    method_name = method.capitalize()
    hom_name = f"H$_{{{hom_dim}}}$"
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(dist_heatmap, cmap='viridis', origin='lower')
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{method_name} ({hom_name}) – Deformation")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Frame index")
        for i in range(num_steps):
            for j in range(num_steps):
                ax.text(j, i, f"{dist_heatmap[i, j]:.2f}",
                        ha="center", va="center", color="w", fontsize=8)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.tight_layout()
        _save_or_show(fig, os.path.join(save_dir, "deformation_heatmap.png") if save_dir else None)

def intra_variability_deformation_test(num_steps: int = 5, num_lors: int = 2000, num_samples: int = 10, ref_frame: int = 0, save_dir: str | None = None) -> None:
    print("=" * 60)
    print("Running: Intra-Frame Variability Deformation Test")
    print("=" * 60)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # TDA configuration parameters
    hom_dim = 1  # Use 0 for clusters, 1 for loops
    method = 'wasserstein'  # Use 'wasserstein' or 'bottleneck'
    dist_func = distances.compute_wasserstein_distance if method == 'wasserstein' else distances.compute_bottleneck_distance

    device = get_device()
    projector = scanner.get_mini_scanner(xp, device)
    grid_shape = projector.in_shape
    phantom_radius = grid_shape[2] // 2 #  Set radius to half of the Z dimension for good visibility of deformation
    scanner_radius = projector.lor_descriptor.scanner.__getattribute__('radius')
    print(f"Using device: {device} | scanner image shape: {grid_shape} | radius: {scanner_radius}")

    # Create a static trajectory to isolate shape changes from motion
    path = trajectories.static_trajectory(
        center=(0, 0, 0),
        steps=num_steps
    )

    # Define the deformation function: 0.0 (sphere) -> 1.0 (cube)
    def deformation_modifier(step):
        return {'morph_factor': step / (num_steps - 1)}

    print("Generating deformed phantom frames...")
    # Generate frames using the deformed primitive
    frames = generator.generate_frames(
        primitives.create_morphed_phantom,
        trajectory=path,
        shape_func=deformation_modifier,
        shape=grid_shape,
        radius=phantom_radius,
        device=device
    )

    print("Frames generated. Visualizing the phantom sequence...")
    generator.plot_frame_sequence_3d(frames, save_path=os.path.join(save_dir, "intra_deformation_frames_3d.png") if save_dir else None)


    # Multi-sample persistence computation
    # For each frame, draw `num_samples` independent LOR sets and compute
    # a persistence diagram for each.
    print(f"Computing {num_steps * num_samples} persistence diagrams in parallel...")
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # Avoid thread oversubscription

    all_frame_diagrams: list[list] = [[] for _ in range(num_steps)]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_idx = {}
        for i, frame in enumerate(frames):
            for s in range(num_samples):
                future = executor.submit(
                    _compute_diagram_for_frame, frame, projector, num_lors, scanner_radius
                )
                future_to_idx[future] = i

        done_count = 0
        total = num_steps * num_samples
        for future in as_completed(future_to_idx):
            frame_idx = future_to_idx[future]
            all_frame_diagrams[frame_idx].append(future.result())
            done_count += 1
            if done_count % num_samples == 0:
                print(f"  Progress: {done_count}/{total} samples computed.")

    torch.set_num_threads(prev_threads)

    # Intra- and inter-frame distances
    # All pairwise distance computations are independent to the thread pool in a single batch.
    print("Computing intra- and inter-frame distances in parallel...")

    # Collect all jobs: (job_key, dgm_a, dgm_b)
    intra_jobs = []  # (frame_idx, dgm_a, dgm_b)
    inter_jobs = []  # (frame_idx, dgm_a, dgm_b)

    for i in range(num_steps):
        for s1 in range(num_samples):
            for s2 in range(s1 + 1, num_samples):
                intra_jobs.append((i, all_frame_diagrams[i][s1], all_frame_diagrams[i][s2]))

    for i in range(num_steps):
        for s0 in range(num_samples):
            for si in range(num_samples):
                if i == ref_frame and s0 == si:
                    continue  # skip self-comparisons to avoid zero-inflating the reference frame
                inter_jobs.append((i, all_frame_diagrams[ref_frame][s0], all_frame_diagrams[i][si]))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        intra_futures = [
            executor.submit(dist_func, dgm_a, dgm_b, hom_dim=hom_dim)
            for _, dgm_a, dgm_b in intra_jobs
        ]
        inter_futures = [
            executor.submit(dist_func, dgm_a, dgm_b, hom_dim=hom_dim)
            for _, dgm_a, dgm_b in inter_jobs
        ]

        # Gather intra-frame results keyed by frame index
        intra_results: dict[int, list[float]] = {i: [] for i in range(num_steps)}
        for job, future in zip(intra_jobs, intra_futures):
            intra_results[job[0]].append(future.result())

        # Gather inter-frame results keyed by frame index
        inter_results: dict[int, list[float]] = {i: [] for i in range(num_steps)}
        for job, future in zip(inter_jobs, inter_futures):
            inter_results[job[0]].append(future.result())

    intra_frame_means = np.array([np.mean(intra_results[i]) for i in range(num_steps)])
    inter_frame_means = np.array([np.mean(inter_results[i]) for i in range(num_steps)])


    #Visualization
    print("Distances computed. Visualizing results...")
    method_name = method.capitalize()
    hom_name = f"H$_{{{hom_dim}}}$"
    frame_indices = np.arange(num_steps)

    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.errorbar(
            frame_indices, inter_frame_means,
            yerr=intra_frame_means,
            fmt='s', markersize=6, capsize=4, capthick=1.2,
            color='#1f77b4', ecolor='#d62728', elinewidth=1.2,
            label=r'Mean $\pm$ intra-frame variability',
        )

        ax.set_xlabel('Frame index')
        ax.set_ylabel(f'{method_name} distance ({hom_name})')
        ax.set_title(f'{method_name} ({hom_name}) from frame {ref_frame} – Deformation')
        ax.set_xticks(frame_indices)
        ax.legend(frameon=True, edgecolor='black', fancybox=False, loc='upper left')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        fig.tight_layout()
        _save_or_show(fig, os.path.join(save_dir, "intra_deformation_errorbar.png") if save_dir else None)



def sinusoidal_motion_test(num_steps: int = 5, num_lors: int = 2000, num_samples: int = 1, save_dir: str | None = None) -> None:
    """Run TDA on a sinusoidal motion phantom simulation.

    Args:
        num_samples: Number of independent LOR samples per frame.  When > 1
            the heatmap entry for frames *i* vs *j* is the mean over all
            pairwise sample distances.
    """
    print("=" * 60)
    print("Running: Sinusoidal Motion Simulation")
    print("=" * 60)

    # TDA configuration parameters
    hom_dim = 1  # Use 0 for clusters, 1 for loops
    method = 'wasserstein'  # Use 'wasserstein' or 'bottleneck'

    # PET scanner geometry and projection settings
    device = get_device()
    projector = scanner.get_mini_scanner(xp, device)
    grid_shape = projector.in_shape
    phantom_radius = grid_shape[1] // 8  # Set radius to an eighth of the Y dimension for good visibility
    scanner_radius = projector.lor_descriptor.scanner.__getattribute__('radius')
    print(f"Using device: {device} | scanner image shape: {grid_shape} | radius: {scanner_radius}")

    # Generate a periodic trajectory to simulate sinusoidal motion (1 full cycle)
    path = trajectories.periodic_trajectory(
        center=(0, 0, 0),
        amplitude=(0, 20 - phantom_radius, 0),
        steps=num_steps,
        frequency=1.0
    )

    print("Generating sinusoidal motion phantom frames...")
    # Produce the sequence of 3D image frames
    frames = generator.generate_frames(
        primitives.create_sphere_phantom,
        trajectory=path,
        shape=grid_shape,
        radius=phantom_radius,
        device=device
    )

    print("Frames generated. Visualizing the phantom sequence...")
    # Visualize the ground truth phantom sequence
    generator.plot_frame_sequence_3d(frames, save_path=os.path.join(save_dir, "sinusoidal_frames_3d.png") if save_dir else None)

    # Parallel frame processing
    scanner_radius = projector.lor_descriptor.scanner.__getattribute__('radius')
    total_jobs = num_steps * num_samples
    print(f"Processing {total_jobs} persistence diagrams ({num_samples} samples × {num_steps} frames) in parallel...")
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # Avoid thread oversubscription

    all_frame_diagrams: list[list] = [[] for _ in range(num_steps)]
    sinogram_images = [None] * num_steps
    adjoint_images = [None] * num_steps

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_idx = {}
        for i, frame in enumerate(frames):
            for s in range(num_samples):
                collect = s == 0
                future = executor.submit(
                    _compute_diagram_for_frame, frame, projector, num_lors, scanner_radius,
                    collect_sinogram=collect, collect_adjoint=collect
                )
                future_to_idx[future] = (i, s)

        done_count = 0
        for future in as_completed(future_to_idx):
            i, s = future_to_idx[future]
            result = future.result()
            if s == 0:
                dgm, sino_img, adjoint_img = result
                sinogram_images[i] = sino_img
                adjoint_images[i] = adjoint_img
            else:
                dgm = result
            all_frame_diagrams[i].append(dgm)
            done_count += 1
            if done_count % num_samples == 0:
                print(f"  Progress: {done_count}/{total_jobs} diagrams computed.")

    torch.set_num_threads(prev_threads)

    print("All frames processed. Computing inter-frame distances...")
    # Compute averaged pairwise distance matrix
    dist_heatmap = _compute_avg_pairwise_distance_matrix(
        all_frame_diagrams, method=method, hom_dim=hom_dim
    )

    print("Distance matrix computed. Visualizing heatmap...")
    method_name = method.capitalize()
    hom_name = f"H$_{{{hom_dim}}}$"
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(dist_heatmap, cmap='viridis', origin='lower')
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{method_name} ({hom_name}) – Sinusoidal Motion")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Frame index")
        for i in range(num_steps):
            for j in range(num_steps):
                ax.text(j, i, f"{dist_heatmap[i, j]:.2f}",
                        ha="center", va="center", color="w", fontsize=8)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.tight_layout()
        _save_or_show(fig, os.path.join(save_dir, "sinusoidal_heatmap.png") if save_dir else None)


def intra_variability_sinusoidal__motion_test(
    num_steps: int = 5,
    num_lors: int = 2000,
    num_samples: int = 10,
    method: str = "wasserstein",
    hom_dim: int = 1,
    ref_frame: int = 0,
    save_dir: str | None = None,
) -> None:
    """Intra-frame variability test for the sinusoidal motion simulation.

    Same structure as intra_variability_deformation_test but the signal comes
    from a periodic (sinusoidal) trajectory instead of a shape deformation.
    For each frame, *num_samples* independent LOR sets are drawn so that
    the intra-frame sampling noise can be compared to the inter-frame
    topological signal caused by motion.
    """
    print("=" * 60)
    print("Running: Intra-Frame Variability Sinusoidal Motion Test")
    print("=" * 60)

    dist_func = (
        distances.compute_wasserstein_distance
        if method == "wasserstein"
        else distances.compute_bottleneck_distance
    )

    device = get_device()
    projector = scanner.get_mini_scanner(xp, device)
    grid_shape = projector.in_shape
    phantom_radius = grid_shape[1] // 8
    scanner_radius = projector.lor_descriptor.scanner.__getattribute__("radius")
    print(f"Using device: {device} | scanner image shape: {grid_shape} | radius: {scanner_radius}")

    # Generate a periodic trajectory to simulate sinusoidal motion (1 full cycle)
    path = trajectories.periodic_trajectory(
        center=(0, 0, 0),
        amplitude=(0, 20 - phantom_radius, 0),
        steps=num_steps,
        frequency=1.0
    )

    print("Generating sinusoidal motion phantom frames...")
    # Produce the sequence of 3D image frames
    frames = generator.generate_frames(
        primitives.create_sphere_phantom,
        trajectory=path,
        shape=grid_shape,
        radius=phantom_radius,
        device=device
    )

    print("Frames generated. Visualizing the phantom sequence...")
    generator.plot_frame_sequence(frames, title="Sinusoidal Motion – Ground truth",
                                 save_path=os.path.join(save_dir, "intra_sinusoidal_frames.png") if save_dir else None)

    # Multi-sample persistence computation (parallel)
    print(f"Computing {num_steps * num_samples} persistence diagrams in parallel...")
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    all_frame_diagrams: list[list] = [[] for _ in range(num_steps)]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_idx = {}
        for i, frame in enumerate(frames):
            for s in range(num_samples):
                future = executor.submit(
                    _compute_diagram_for_frame, frame, projector, num_lors, scanner_radius
                )
                future_to_idx[future] = i

        done_count = 0
        total = num_steps * num_samples
        for future in as_completed(future_to_idx):
            frame_idx = future_to_idx[future]
            all_frame_diagrams[frame_idx].append(future.result())
            done_count += 1
            if done_count % num_samples == 0:
                print(f"  Progress: {done_count}/{total} samples computed.")

    torch.set_num_threads(prev_threads)

    # Intra- and inter-frame distances
    print("Computing intra- and inter-frame distances in parallel...")

    intra_jobs = []
    inter_jobs = []

    for i in range(num_steps):
        for s1 in range(num_samples):
            for s2 in range(s1 + 1, num_samples):
                intra_jobs.append((i, all_frame_diagrams[i][s1], all_frame_diagrams[i][s2]))

    for i in range(num_steps):
        for s0 in range(num_samples):
            for si in range(num_samples):
                if i == ref_frame and s0 == si:
                    continue
                inter_jobs.append((i, all_frame_diagrams[ref_frame][s0], all_frame_diagrams[i][si]))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        intra_futures = [
            executor.submit(dist_func, dgm_a, dgm_b, hom_dim=hom_dim)
            for _, dgm_a, dgm_b in intra_jobs
        ]
        inter_futures = [
            executor.submit(dist_func, dgm_a, dgm_b, hom_dim=hom_dim)
            for _, dgm_a, dgm_b in inter_jobs
        ]

        intra_results: dict[int, list[float]] = {i: [] for i in range(num_steps)}
        for job, future in zip(intra_jobs, intra_futures):
            intra_results[job[0]].append(future.result())

        inter_results: dict[int, list[float]] = {i: [] for i in range(num_steps)}
        for job, future in zip(inter_jobs, inter_futures):
            inter_results[job[0]].append(future.result())

    intra_frame_means = np.array([np.mean(intra_results[i]) for i in range(num_steps)])
    inter_frame_means = np.array([np.mean(inter_results[i]) for i in range(num_steps)])

    # Visualization
    print("Distances computed. Visualizing results...")
    method_name = method.capitalize()
    hom_name = f"H$_{{{hom_dim}}}$"
    frame_indices = np.arange(num_steps)

    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.errorbar(
            frame_indices, inter_frame_means,
            yerr=intra_frame_means,
            fmt='s', markersize=6, capsize=4, capthick=1.2,
            color='#1f77b4', ecolor='#d62728', elinewidth=1.2,
            label=r'Mean $\pm$ intra-frame variability',
        )

        ax.set_xlabel('Frame index')
        ax.set_ylabel(f'{method_name} distance ({hom_name})')
        ax.set_title(f'{method_name} ({hom_name}) from frame {ref_frame} – Sinusoidal Motion')
        ax.set_xticks(frame_indices)
        ax.legend(frameon=True, edgecolor='black', fancybox=False, loc='upper left')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        fig.tight_layout()
        _save_or_show(fig, os.path.join(save_dir, "intra_sinusoidal_errorbar.png") if save_dir else None)

def _up_down_scale(step: int, num_steps: int, amplitude: float) -> float:
    """Linear increase then decrease: 1.0 -> (1.0+amplitude) -> 1.0."""
    if num_steps < 2:
        return 1.0
    mid = (num_steps - 1) / 2.0
    if step <= mid:
        phase = step / mid if mid != 0 else 0.0
    else:
        phase = (num_steps - 1 - step) / mid if mid != 0 else 0.0
    return 1.0 + amplitude * phase


def size_test(
    phantom_fn,
    num_steps: int = 5,
    num_lors: int = 2000,
    num_samples: int = 1,
    method: str = "wasserstein",
    hom_dim: int = 1,
    amplitude: float = 1.0,
    save_dir: str | None = None,
):
    """Run TDA on a sequence of phantoms that grow then shrink in size.
    Args:
        phantom_fn: A function that generates a phantom image given size parameters.
            Supported functions:
                - primitives.create_sphere_phantom
                - primitives.create_box_phantom
                - primitives.create_ellipsoid_phantom
        num_steps: Number of frames in the sequence.
        num_lors: Number of LORs to sample per frame.
        num_samples: Number of independent LOR samples per frame.  When > 1
            the heatmap entry is the mean over all pairwise sample distances.
        method: Distance method for comparing diagrams ('wasserstein' or 'bottleneck').
        hom_dim: Homology dimension to analyze (0 for clusters, 1 for loops).
        amplitude: Maximum relative size increase (e.g., 1.0 means up to 2x size at peak).
    """
    print("=" * 60)
    print(f"Running: Size Test ({phantom_fn.__name__})")
    print("=" * 60)

    device = get_device()
    projector = scanner.get_mini_scanner(xp, device)
    grid_shape = projector.in_shape
    max_radius = grid_shape[2] // 2  # peak radius at the midpoint of the sequence, scaled to fit in the grid
    base_radius = max(1, round(max_radius / (1 + amplitude)))
    print(f"Base radius: {base_radius} | Max radius: {max_radius} | Amplitude: {amplitude}")
    scanner_radius = projector.lor_descriptor.scanner.__getattribute__("radius")
    print(f"Using device: {device} | scanner image shape: {grid_shape} | radius: {scanner_radius}")

    path = trajectories.static_trajectory(center=(0, 0, 0), steps=num_steps)

    print(f"Generating size-varying phantom frames ({phantom_fn.__name__})...")
    # build frames with systematic up->down size variation according to the specified phantom function
    if phantom_fn is primitives.create_sphere_phantom:
        def size_modifier(step: int):
            scale = _up_down_scale(step, num_steps, amplitude)
            return {"radius": max(1, round(base_radius * scale))}

        frames = generator.generate_frames(
            phantom_fn, trajectory=path, shape_func=size_modifier,
            shape=grid_shape, radius=base_radius
        )

    elif phantom_fn is primitives.create_box_phantom:
        base_side = float(base_radius * 1.8)

        def size_modifier(step: int):
            scale = _up_down_scale(step, num_steps, amplitude)
            L = base_side * scale
            return {"side_lengths": (L, L, L)}

        frames = generator.generate_frames(
            phantom_fn, trajectory=path, shape_func=size_modifier,
            shape=grid_shape, side_lengths=(base_side, base_side, base_side)
        )

    elif phantom_fn is primitives.create_ellipsoid_phantom:
        base_radii = (base_radius, max(1, round(base_radius * 0.7)), max(1, round(base_radius * 0.7)))

        def size_modifier(step: int):
            scale = _up_down_scale(step, num_steps, amplitude)
            radii = (max(1, round(base_radii[0] * scale)), base_radii[1], base_radii[2])
            return {"radii": radii}

        frames = generator.generate_frames(
            phantom_fn, trajectory=path, shape_func=size_modifier,
            shape=grid_shape, radii=base_radii
        )

    else:
        raise ValueError(
            "Unsupported phantom_fn. Use one of:\n"
            "  primitives.create_sphere_phantom\n"
            "  primitives.create_box_phantom\n"
            "  primitives.create_ellipsoid_phantom"
        )

    print("Frames generated. Visualizing the phantom sequence...")
    generator.plot_frame_sequence_3d(frames, save_path=os.path.join(save_dir, f"size_{phantom_fn.__name__}_frames_3d.png") if save_dir else None)


    # Parallel frame processing
    total_jobs = num_steps * num_samples
    print(f"Processing {total_jobs} persistence diagrams ({num_samples} samples × {num_steps} frames) in parallel...")
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # Avoid thread oversubscription

    all_frame_diagrams: list[list] = [[] for _ in range(num_steps)]
    sinogram_images = [None] * num_steps

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_idx = {}
        for i, frame in enumerate(frames):
            for s in range(num_samples):
                collect = s == 0
                future = executor.submit(
                    _compute_diagram_for_frame, frame, projector, num_lors, scanner_radius,
                    collect_sinogram=collect
                )
                future_to_idx[future] = (i, s)

        done_count = 0
        for future in as_completed(future_to_idx):
            i, s = future_to_idx[future]
            result = future.result()
            if s == 0:
                dgm, sino_img = result
                sinogram_images[i] = sino_img
            else:
                dgm = result
            all_frame_diagrams[i].append(dgm)
            done_count += 1
            if done_count % num_samples == 0:
                print(f"  Progress: {done_count}/{total_jobs} diagrams computed.")

    torch.set_num_threads(prev_threads)


    print("All frames processed. Computing inter-frame distances...")
    # Compute averaged pairwise distance matrix
    dist_heatmap = _compute_avg_pairwise_distance_matrix(
        all_frame_diagrams, method=method, hom_dim=hom_dim
    )

    print("Distance matrix computed. Visualizing heatmap...")
    method_name = method.capitalize()
    hom_name = f"H$_{{{hom_dim}}}$"
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(dist_heatmap, cmap='viridis', origin='lower')
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{method_name} ({hom_name}) – Size ({phantom_fn.__name__})")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Frame index")
        for i in range(num_steps):
            for j in range(num_steps):
                ax.text(j, i, f"{dist_heatmap[i, j]:.2f}",
                        ha="center", va="center", color="w", fontsize=8)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.tight_layout()
        _save_or_show(fig, os.path.join(save_dir, f"size_{phantom_fn.__name__}_heatmap.png") if save_dir else None)


def intra_variability_size_test(phantom_fn=None, num_steps: int = 5, num_lors: int = 2000, num_samples: int = 10, method: str = "wasserstein", hom_dim: int = 1, amplitude: float = 1, ref_frame: int = 0, save_dir: str | None = None) -> None:
    """Intra-frame variability test for the size up-then-down simulation.

    For each frame, *num_samples* independent LOR sets are drawn so
    that the intra-frame sampling noise can be compared to the inter-frame
    topological signal.
    """
    if phantom_fn is None:
        phantom_fn = primitives.create_sphere_phantom

    print("=" * 60)
    print(f"Running: Intra-Frame Variability Size Test ({phantom_fn.__name__})")
    print("=" * 60)

    dist_func = (
        distances.compute_wasserstein_distance
        if method == "wasserstein"
        else distances.compute_bottleneck_distance
    )

    device = get_device()
    projector = scanner.get_mini_scanner(xp, device)
    grid_shape = projector.in_shape
    max_radius = grid_shape[2] // 2  # peak radius at the midpoint of the schedule
    base_radius = max(1, round(max_radius / (1 + amplitude)))
    scanner_radius = projector.lor_descriptor.scanner.__getattribute__("radius")
    print(f"Using device: {device} | scanner image shape: {grid_shape} | radius: {scanner_radius}")

    path = trajectories.static_trajectory(center=(0, 0, 0), steps=num_steps)

    print(f"Generating size-varying phantom frames ({phantom_fn.__name__})...")
    # Build frames with systematic up->down size schedule
    if phantom_fn is primitives.create_sphere_phantom:
        def size_modifier(step: int):
            scale = _up_down_scale(step, num_steps, amplitude)
            return {"radius": max(1, round(base_radius * scale))}

        frames = generator.generate_frames(
            phantom_fn, trajectory=path, shape_func=size_modifier,
            shape=grid_shape, radius=base_radius, device=device
        )

    elif phantom_fn is primitives.create_box_phantom:
        base_side = float(base_radius * 1.8)

        def size_modifier(step: int):
            scale = _up_down_scale(step, num_steps, amplitude)
            L = base_side * scale
            return {"side_lengths": (L, L, L)}

        frames = generator.generate_frames(
            phantom_fn, trajectory=path, shape_func=size_modifier,
            shape=grid_shape, side_lengths=(base_side, base_side, base_side), device=device
        )

    elif phantom_fn is primitives.create_ellipsoid_phantom:
        base_radii = (base_radius, max(1, round(base_radius * 0.7)), max(1, round(base_radius * 0.7)))

        def size_modifier(step: int):
            scale = _up_down_scale(step, num_steps, amplitude)
            radii = (max(1, round(base_radii[0] * scale)), base_radii[1], base_radii[2])
            return {"radii": radii}

        frames = generator.generate_frames(
            phantom_fn, trajectory=path, shape_func=size_modifier,
            shape=grid_shape, radii=base_radii, device=device
        )

    else:
        raise ValueError(
            "Unsupported phantom_fn. Use one of:\n"
            "  primitives.create_sphere_phantom\n"
            "  primitives.create_box_phantom\n"
            "  primitives.create_ellipsoid_phantom"
        )

    print("Frames generated. Visualizing the phantom sequence...")
    generator.plot_frame_sequence(frames, title=f"Size – {phantom_fn.__name__}",
                                 save_path=os.path.join(save_dir, f"intra_size_{phantom_fn.__name__}_frames.png") if save_dir else None)

    # Multi-sample persistence computation (parallel)
    print(f"Computing {num_steps * num_samples} persistence diagrams in parallel...")
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    all_frame_diagrams: list[list] = [[] for _ in range(num_steps)]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_idx = {}
        for i, frame in enumerate(frames):
            for s in range(num_samples):
                future = executor.submit(
                    _compute_diagram_for_frame, frame, projector, num_lors, scanner_radius
                )
                future_to_idx[future] = i

        done_count = 0
        total = num_steps * num_samples
        for future in as_completed(future_to_idx):
            frame_idx = future_to_idx[future]
            all_frame_diagrams[frame_idx].append(future.result())
            done_count += 1
            if done_count % num_samples == 0:
                print(f"  Progress: {done_count}/{total} samples computed.")

    torch.set_num_threads(prev_threads)

    # Intra- and inter-frame distances (parallel)
    print("Computing intra- and inter-frame distances in parallel...")

    intra_jobs = []
    inter_jobs = []

    for i in range(num_steps):
        for s1 in range(num_samples):
            for s2 in range(s1 + 1, num_samples):
                intra_jobs.append((i, all_frame_diagrams[i][s1], all_frame_diagrams[i][s2]))

    for i in range(num_steps):
        for s0 in range(num_samples):
            for si in range(num_samples):
                if i == ref_frame and s0 == si:
                    continue
                inter_jobs.append((i, all_frame_diagrams[ref_frame][s0], all_frame_diagrams[i][si]))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        intra_futures = [
            executor.submit(dist_func, dgm_a, dgm_b, hom_dim=hom_dim)
            for _, dgm_a, dgm_b in intra_jobs
        ]
        inter_futures = [
            executor.submit(dist_func, dgm_a, dgm_b, hom_dim=hom_dim)
            for _, dgm_a, dgm_b in inter_jobs
        ]

        intra_results: dict[int, list[float]] = {i: [] for i in range(num_steps)}
        for job, future in zip(intra_jobs, intra_futures):
            intra_results[job[0]].append(future.result())

        inter_results: dict[int, list[float]] = {i: [] for i in range(num_steps)}
        for job, future in zip(inter_jobs, inter_futures):
            inter_results[job[0]].append(future.result())

    intra_frame_means = np.array([np.mean(intra_results[i]) for i in range(num_steps)])
    inter_frame_means = np.array([np.mean(inter_results[i]) for i in range(num_steps)])

    # Visualization
    print("Distances computed. Visualizing results...")
    method_name = method.capitalize()
    hom_name = f"H$_{{{hom_dim}}}$"
    frame_indices = np.arange(num_steps)

    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.errorbar(
            frame_indices, inter_frame_means,
            yerr=intra_frame_means,
            fmt='s', markersize=6, capsize=4, capthick=1.2,
            color='#1f77b4', ecolor='#d62728', elinewidth=1.2,
            label=r'Mean $\pm$ intra-frame variability',
        )

        ax.set_xlabel('Frame index')
        ax.set_ylabel(f'{method_name} distance ({hom_name})')
        ax.set_title(f'{method_name} ({hom_name}) from frame {ref_frame} – Size ({phantom_fn.__name__})')
        ax.set_xticks(frame_indices)
        ax.legend(frameon=True, edgecolor='black', fancybox=False, loc='upper left')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        fig.tight_layout()
        _save_or_show(fig, os.path.join(save_dir, f"intra_size_{phantom_fn.__name__}_errorbar.png") if save_dir else None)


def visualize_2d_phantom(shape_type: str = 'triangle', num_lors: int = 1000, noise_type: str | None = None, noise_level: float | None = None, save_dir: str | None = None) -> None:
    """Generate a 2D phantom, project it, compute LORs and visualize the Plücker point cloud.

    Args:
        shape_type (str): One of 'triangle', 'disk', 'ellips', 'square'.
        num_lors (int): Number of LORs to sample per iteration.
        noise_type (str, optional): Noise to apply before projection.
            One of 'gaussian', 'poisson', 'salt_and_pepper'.
            None means no noise.
        noise_level (float, optional): Intensity parameter for the chosen noise type.
            Required when noise_type is set.
    """
    print("=" * 60)
    print(f"Running: 2D Phantom Point Cloud Visualization (shape={shape_type})")
    print("=" * 60)

    import numpy as np
    import plotly.graph_objects as go

    # Set up a 2D projector
    device = get_device()
    projector = scanner.get_mini_scanner(xp, device)
    _, _, projector = scanner.to_2D(projector)
    grid_shape = projector.in_shape  # (X, Y, 1)
    phantom_radius = grid_shape[1] // 4

    # Build the 2D phantom based on the chosen shape type
    shape_2d = grid_shape[:2]
    center_2d = (0, 0)

    if shape_type == 'triangle':
        image_2d = primitives.create_simplex_phantom(
            shape=shape_2d, radius=phantom_radius, center=center_2d, device=device
        )
    elif shape_type == 'disk':
        image_2d = primitives.create_sphere_phantom(
            shape=shape_2d, radius=phantom_radius, center=center_2d, device=device
        )
    elif shape_type == 'ellips':
        radii = (phantom_radius, phantom_radius * 0.6)
        image_2d = primitives.create_ellipsoid_phantom(
            shape=shape_2d, radii=radii, center=center_2d, device=device
        )
    elif shape_type == 'square':
        side_lengths = (phantom_radius * 1.8, phantom_radius * 1.8)
        image_2d = primitives.create_box_phantom(
            shape=shape_2d, side_lengths=side_lengths, center=center_2d, device=device
        )
    else:
        raise ValueError(
            f"Unknown shape_type '{shape_type}'. "
            "Choose from 'triangle', 'disk', 'ellips', 'square'."
        )
    

    # Apply noise if requested
    if noise_type is not None:
        image_2d = noise.apply_noise(image_2d, noise_type, noise_level)
        actual_level = noise_level if noise_level is not None else noise._NOISE_TYPES[noise_type][2]
        print(f"Applied '{noise_type}' noise with level={actual_level}")

    # Expand to (X, Y, 1) for the 2D projector
    image = image_2d.unsqueeze(2).float()
    print(f"Generated 2D '{shape_type}' phantom with shape {image.shape}")

    # Visualise the phantom
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(image[:, :, 0].cpu(), cmap='gray', origin='lower')
        ax.set_title(f"Phantom – {shape_type}")
        fig.colorbar(im, ax=ax)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.tight_layout()
        _save_or_show(fig, os.path.join(save_dir, f"2d_phantom_{shape_type}.png") if save_dir else None)

    # Forward-project (sinogram)
    forward = projector(image)
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(parallelproj.to_numpy_array(forward[:, :, 0].T),
                  cmap='Greys_r', vmin=0)
        ax.set_title(f"Sinogram – {shape_type}")
        ax.tick_params(which='both', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.tight_layout()
        _save_or_show(fig, os.path.join(save_dir, f"2d_sinogram_{shape_type}.png") if save_dir else None)

    # Generate LORs and Plücker coordinates
    
    p1, p2 = scanner.generate_lors_from_image(
        image, projector, num_lors=num_lors, show=False
    )
    plucker_coords = representations.to_canonical_plucker(p1, p2)
    print(f"Plücker coordinates shape: {plucker_coords.shape}")

    # Check for duplicate LOR endpoint pairs
    # Concatenate p1 and p2 into a single row per LOR and look for repeats
    lor_pairs = torch.cat([p1, p2], dim=1)  # (num_lors, 6)
    unique_pairs, counts = torch.unique(
        lor_pairs, dim=0, return_inverse=False, return_counts=True
    )
    num_duplicated = (counts > 1).sum().item()
    total_excess = (counts[counts > 1] - 1).sum().item()
    print(f"LOR duplicate check: {unique_pairs.shape[0]} unique out of {lor_pairs.shape[0]} "
          f"({num_duplicated} LORs sampled more than once, {total_excess} extra copies)")

    # 3D scatter of Plücker point cloud
    fig = go.Figure(data=go.Scatter3d(
        x=plucker_coords[:, 0], y=plucker_coords[:, 1], z=plucker_coords[:, -1],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.8)
    ))
    fig.update_layout(
        title=f'Plücker Point Cloud – {shape_type}',
        scene=dict(
            xaxis_title='lx', yaxis_title='ly', zaxis_title='mz',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.write_image(os.path.join(save_dir, f"2d_plucker_{shape_type}.png"))
    else:
        fig.show()


def visualize_sphere_sinogram(save_dir: str | None = None) -> None:
    """Create a sphere phantom and display its full forward-projected sinogram."""
    dev = get_device()
    proj = scanner.get_mini_scanner(xp, dev)

    # Create a sphere phantom that fits the scanner image grid (40x40x16)
    phantom = primitives.create_sphere_phantom(
        shape=(40, 40, 16), radius=8, center=(5, -3, 0), intensity=1.0, device=dev,
    )

    # Forward project to get the full sinogram
    sinogram = proj(phantom)
    sino_np = parallelproj.to_numpy_array(sinogram)

    # sinogram shape is (num_radial, num_views, num_planes) with RVP order
    print(f"Phantom shape : {phantom.shape}")
    print(f"Sinogram shape: {sino_np.shape}")

    phantom_np = parallelproj.to_numpy_array(phantom)
    mid_plane = sino_np.shape[2] // 2

    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(6, 9))

        axes[0].imshow(phantom_np[:, :, phantom.shape[2] // 2].T, cmap="Greys_r", origin="lower")
        axes[0].set_title("Phantom (mid-Z Slice)")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

        axes[1].imshow(sino_np.sum(axis=2).T, cmap="Greys_r", aspect="auto", origin="lower")
        axes[1].set_title("Sinogram")
        axes[1].set_xlabel("Radial bin")
        axes[1].set_ylabel("View angle")

        for ax in axes:
            ax.tick_params(which='both', direction='in', top=True, right=True)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

        fig.suptitle("Sphere – Phantom & sinogram")
        fig.tight_layout()
        _save_or_show(fig, os.path.join(save_dir, "sphere_sinogram.png") if save_dir else None)


if __name__ == "__main__":
    #visualize_sphere_sinogram()
    #deformation_test()
    #intra_variability_deformation_test(ref_frame=1,save_dir="figures/intra_variability_deformation")
    #linear_motion_test()
    #sinusoidal_motion_test()
    #intra_variability_sinusoidal__motion_test()
    size_test(primitives.create_sphere_phantom)
    #intra_variability_size_test()
    #visualize_sphere_sinogram()
    """phantoms =  ['triangle', 'disk', 'ellips', 'square']
    for shape in phantoms:
        visualize_2d_phantom(shape_type=shape, num_lors=1000)"""

