import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_phantom_frame(phantom, frame=0):
    """Plot z, y, and x center slices of a phantom frame.

    Args:
        phantom (torch.Tensor): Phase series of shape (num_frames, Nz, Ny, Nx).
        frame (int): Frame index to visualize.
    """
    vol = phantom[frame].cpu()
    nz, ny, nx = vol.shape

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Phantom frame {frame}")

    axes[0].imshow(vol[nz // 2, :, :], cmap='hot')
    axes[0].set_title("z slice")

    axes[1].imshow(vol[:, ny // 2, :], cmap='hot')
    axes[1].set_title("y slice")

    axes[2].imshow(vol[:, :, nx // 2], cmap='hot')
    axes[2].set_title("x slice")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_sinogram(sinogram, max_planes=10, max_tofbins=9):
    """Plot sinogram as a grid of planes, with TOF bins as columns if present.

    For non-TOF sinograms, each subplot shows one plane. For TOF sinograms,
    rows are planes and columns are TOF bins centered around bin 0.

    Args:
        sinogram (torch.Tensor): Sinogram of shape (num_rad, num_angles, num_planes)
            or (num_rad, num_angles, num_planes, num_tofbins).
        max_planes (int): Maximum number of planes to show.
        max_tofbins (int): Maximum number of TOF bins to show (TOF only).
    """
    sino = sinogram.cpu().float()
    tof = sino.ndim == 4

    num_planes = sino.shape[2]
    num_tofbins = sino.shape[3] if tof else 1

    if tof:
        rows = min(max_planes, num_planes)
        cols = min(max_tofbins, num_tofbins)
    else:
        rows = 4
        cols = min(5, -(-num_planes // 4))

    vmax = float(sino.max())
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.5 * rows),
                             sharex=True, sharey=True)
    axes = axes.reshape(rows, cols) if rows > 1 or cols > 1 else [[axes]]

    plane_step = max(1, num_planes // rows)

    for i in range(rows):
        if tof:
            plane_idx = i * plane_step
            center_bin = num_tofbins // 2
            bin_offset = center_bin - (cols // 2)

            for j in range(cols):
                tof_idx = bin_offset + j
                ax = axes[i][j]

                if 0 <= tof_idx < num_tofbins:
                    ax.imshow(sino[:, :, plane_idx, tof_idx],
                              cmap='Greys_r', vmin=0, vmax=vmax, aspect='auto')

                if i == 0:
                    ax.set_title(f"bin {tof_idx - center_bin}", fontsize='small')
                if j == 0:
                    ax.set_ylabel(f"plane {plane_idx}", fontsize='small')

                ax.set_xticks([])
                ax.set_yticks([])
        else:
            for j in range(cols):
                plane_idx = i * cols + j
                ax = axes[i][j]

                if plane_idx < num_planes:
                    ax.imshow(sino[:, :, plane_idx],
                              cmap='Greys_r', vmin=0, vmax=vmax, aspect='auto')
                    ax.set_title(f"plane {plane_idx}", fontsize='small')
                else:
                    ax.set_axis_off()

                ax.set_xticks([])
                ax.set_yticks([])

    fig.suptitle("Sinogram" + (" (TOF)" if tof else ""))
    plt.tight_layout()
    plt.show()


def plot_pointcloud(points, max_points=50_000):
    """Plot a 3D scatter of TOF bin center coordinates.

    Args:
        points (torch.Tensor): TOF bin center coordinates of shape (N, 3).
        max_points (int): Maximum number of points to plot for performance.
    """
    pts = points.cpu()

    if pts.shape[0] > max_points:
        indices = torch.randperm(pts.shape[0])[:max_points]
        pts = pts[indices]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 2], pts[:, 1], pts[:, 0], s=0.1, alpha=0.3, c='steelblue')
    ax.set_title("TOF point cloud")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    plt.show()


def plot_reconstruction(image):
    """Plot z, y, and x center slices of a reconstructed image.

    Args:
        image (torch.Tensor): Reconstructed image of shape (Nz, Ny, Nx).
    """
    vol = image.cpu()
    nz, ny, nx = vol.shape

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("MLEM reconstruction")

    axes[0].imshow(vol[nz // 2, :, :], cmap='hot')
    axes[0].set_title("z slice")

    axes[1].imshow(vol[:, ny // 2, :], cmap='hot')
    axes[1].set_title("y slice")

    axes[2].imshow(vol[:, :, nx // 2], cmap='hot')
    axes[2].set_title("x slice")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_persistence_diagram(diagrams, title="Persistence diagram"):
    """Plot persistence diagrams for all homology dimensions.

    Args:
        diagrams (list[np.ndarray]): Persistence diagrams, one per dimension.
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ['steelblue', 'tomato', 'seagreen']
    for dim, dgm in enumerate(diagrams):
        if len(dgm) > 0:
            ax.scatter(dgm[:, 0], dgm[:, 1], s=10,
                       label=f"H{dim}", color=colors[dim % len(colors)])

    all_vals = np.concatenate([dgm.flatten() for dgm in diagrams if len(dgm) > 0])
    vmin, vmax = all_vals.min(), all_vals.max()
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=0.8)

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_distance_matrix(dist_matrix, title="Distance matrix"):
    """Plot a pairwise distance matrix as a heatmap.

    Args:
        dist_matrix (np.ndarray): Symmetric distance matrix of shape (N, N).
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")

    plt.tight_layout()
    plt.show()