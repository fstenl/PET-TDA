"""Visualization utilities for PET phantom, sinogram, reconstruction, and TDA data."""

import torch
import matplotlib.pyplot as plt
import numpy as np


def save_or_show(fig, path=None):
    """Save figure to file or show interactively.

    Args:
        fig (matplotlib.figure.Figure): Figure to save or show.
        path (str | None): File path to save to. If None, shows interactively.
    """
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_volume(frames, slice_idx=None, max_frames=None, title="", path=None):
    """Plot center z-slices of a sequence of 3D volumes in a grid.

    Args:
        frames (torch.Tensor): Volume series of shape (num_frames, Nz, Ny, Nx).
        slice_idx (int | None): z-slice index to show. Defaults to center slice.
        max_frames (int | None): Maximum number of frames to show. If None, shows all.
        title (str): Overall plot title.
        path (str | None): File path to save to. If None, shows interactively.
    """
    frames = frames.cpu()
    num_frames = frames.shape[0]

    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
        frames = frames[:num_frames]

    if slice_idx is None:
        slice_idx = frames.shape[1] // 2

    cols = min(5, num_frames)
    rows = -(-num_frames // cols)  # ceiling division

    vmax = float(frames.max())

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows):
        for j in range(cols):
            frame_idx = i * cols + j
            ax = axes[i][j]

            if frame_idx < num_frames:
                ax.imshow(frames[frame_idx, slice_idx, :, :],
                          cmap='Greys_r', vmin=0, vmax=vmax)
                ax.set_title(f"frame {frame_idx}", fontsize='small')
            else:
                ax.set_axis_off()

            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    save_or_show(fig, path)


def plot_sinogram(sinogram, max_planes=10, max_tofbins=9, path=None):
    """Plot sinogram as a grid of planes, with TOF bins as columns if present.

    For non-TOF sinograms, each subplot shows one plane. For TOF sinograms,
    rows are planes and columns are TOF bins centered around bin 0.

    Args:
        sinogram (torch.Tensor): Sinogram of shape (num_rad, num_angles, num_planes)
            or (num_rad, num_angles, num_planes, num_tofbins).
        max_planes (int): Maximum number of planes to show.
        max_tofbins (int): Maximum number of TOF bins to show (TOF only).
        path (str | None): File path to save to. If None, shows interactively.
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
    save_or_show(fig, path)


def plot_pointcloud(points, max_points=50_000, path=None):
    """Plot a 3D scatter of point coordinates.

    Args:
        points (torch.Tensor): Point coordinates of shape (N, 3).
        max_points (int): Maximum number of points to plot for performance.
        path (str | None): File path to save to. If None, shows interactively.
    """
    pts = points.cpu()

    if pts.shape[0] > max_points:
        indices = torch.randperm(pts.shape[0])[:max_points]
        pts = pts[indices]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 2], pts[:, 1], pts[:, 0], s=0.1, alpha=0.3, c='steelblue')
    ax.set_title("Point cloud")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    save_or_show(fig, path)


def plot_persistence_diagram(diagrams, title="Persistence diagram", path=None):
    """Plot persistence diagrams for all homology dimensions.

    Args:
        diagrams (list[np.ndarray]): Persistence diagrams, one per dimension.
        title (str): Plot title.
        path (str | None): File path to save to. If None, shows interactively.
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
    save_or_show(fig, path)


def plot_distance_matrix(dist_matrix, title="Distance matrix", path=None):
    """Plot a pairwise distance matrix as a heatmap.

    Args:
        dist_matrix (np.ndarray): Symmetric distance matrix of shape (N, N).
        title (str): Plot title.
        path (str | None): File path to save to. If None, shows interactively.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")

    plt.tight_layout()
    save_or_show(fig, path)