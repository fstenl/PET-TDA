import matplotlib.pyplot as plt

def generate_frames(phantom_func, trajectory, shape_func=None, **base_kwargs):
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


def plot_frame_sequence(frames, axis=2, title="Phantom Movement Sequence"):
    """Visualizes central slices of the generated frames to verify motion.

    Args:
        frames (list): List of 3D torch.Tensors representing the volumes.
        axis (int): The axis along which to take the central slice.
            With (X, Y, Z) convention: 0=X (sagittal), 1=Y (coronal), 2=Z (axial).
        title (str): The main title for the plot montage.
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
        axes_flat[i].set_title(f"Step {i}")
        axes_flat[i].axis('off')

    # Hide unused subplot slots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()