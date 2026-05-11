import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from collections.abc import Callable
from scipy.ndimage import binary_erosion

def generate_frames(phantom_func: Callable, trajectory: list[tuple], shape_func: Callable | None = None, **base_kwargs) -> list[torch.Tensor]:
    """Generates a sequence of phantom frames based on a trajectory and optional shape changes.

    Args:
        phantom_func (callable): Function from primitives.py (e.g., create_sphere_phantom).
        trajectory (list): List of center tuples from trajectories.py.
        shape_func (callable, optional): Function returning a dict of shape updates per step.
        **base_kwargs: Constant arguments for the phantom function (shape, intensity, etc.).

    Returns:
        list: A list of torch.Tensors, each representing one frame/volume.
    """
    frames = []

    for i, center in enumerate(trajectory):
        # Prepare arguments for this specific frame
        current_kwargs = base_kwargs.copy()

        # Apply dynamic shape changes if a modifier function is provided
        if shape_func is not None:
            current_kwargs.update(shape_func(i))

        # Render the phantom at the current center
        frame = phantom_func(center=center, **current_kwargs)

        frames.append(frame)

    return frames


def plot_frame_sequence(frames: list[torch.Tensor], axis: int = 2, title: str = "Phantom Sequence", save_path: str | None = None) -> None:
    """Visualizes central slices of the generated frames to verify motion.

    Args:
        frames (list): List of 3D torch.Tensors representing the volumes.
        axis (int): The axis along which to take the central slice.
            With (X, Y, Z) convention: 0=X (sagittal), 1=Y (coronal), 2=Z (axial).
        title (str): The main title for the plot montage.
        save_path (str, optional): Path to save the figure. If provided, the figure
            is saved and not shown in the GUI.
    """
    n = len(frames)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.suptitle(f"{title}", fontsize=16)

    # Ensure axes is an array even for a single frame
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, frame in enumerate(frames):
        # Extract the middle slice along the specified axis
        mid_idx = frame.shape[axis] // 2

        if axis == 0:
            slice_img = frame[mid_idx, :, :].T
        elif axis == 1:
            slice_img = frame[:, mid_idx, :].T
        else:
            slice_img = frame[:, :, mid_idx].T

        # Plotting the 2D slice
        im = axes_flat[i].imshow(slice_img.cpu().numpy(), cmap='hot')
        axes_flat[i].set_title(f"Frame {i}")
        axes_flat[i].axis('off')

    # Hide unused subplot slots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_frame_sequence_3d(
    frames: list[torch.Tensor],
    threshold: float = 0.5,
    title: str = "Phantom Dataset",
    alpha: float = 1,
    cmap: str = "Greys_r",
    save_path: str | None = None,
) -> None:
    """Visualizes phantom frames as 3D voxel plots using matplotlib.

    Non-zero voxels above *threshold* are rendered with colours mapped to
    their intensity.  Each frame is shown in its own 3D subplot.

    Args:
        frames (list): List of 3D torch.Tensors representing the volumes.
        threshold (float): Minimum voxel value (relative to the frame max)
            to display.  0.5 means only voxels above 50 % of the peak are
            shown.  Lower values show more voxels but render more slowly.
        title (str): The main title for the plot montage.
        alpha (float): Opacity of the rendered voxels (0–1).
        cmap (str): Matplotlib colormap name used to colour voxels by
            intensity.
        save_path (str, optional): Path to save the figure. If provided, the figure
            is saved and not shown in the GUI.
    """
    

    n = len(frames)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    fig.suptitle(title, fontsize=16)

    colormap = plt.colormaps[cmap]

    # Pre-compute consistent axis limits from the frame dimensions
    sx, sy, sz = frames[0].shape
    aspect = (sx, sy, sz) 

    for i, frame in enumerate(frames):
        data = frame.cpu().numpy().astype(float)

        # Normalise to [0, 1] so the threshold and colormap work consistently
        vmax = data.max()
        if vmax > 0:
            data_norm = data / vmax
        else:
            data_norm = data

        # Boolean mask of voxels above the threshold
        filled = data_norm >= threshold

        # Keep only the surface shell
        interior = binary_erosion(filled)
        shell = filled & ~interior

        # Map normalised intensity to RGBA colours via the chosen colormap
        colours = colormap(data_norm)
        colours[..., 3] = np.where(shell, alpha, 0.0)

        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.voxels(shell, facecolors=colours, edgecolor="k", linewidth=0.1)
        ax.set_title(f"Frame {i}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Synchronise axis limits so all frames share the same scale
        ax.set_xlim(0, sx)
        ax.set_ylim(0, sy)
        ax.set_zlim(0, sz)

        # Set box aspect ratio to match the true grid proportions
        ax.set_box_aspect(aspect)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

