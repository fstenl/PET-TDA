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


def plot_distance_matrix(dist_matrix, title="Distance matrix",
                         labels=None, cbar_label=None, cmap='viridis'):
    """Plot a pairwise distance matrix as a heatmap.

    Args:
        dist_matrix (np.ndarray): Symmetric distance matrix of shape (N, N).
        title (str): Plot title.
        labels (list | None): Per-row/column tick labels. Defaults to indices.
        cbar_label (str | None): Colorbar label. Defaults to no label.
        cmap (str): Matplotlib colormap name.
    """
    dist_matrix = np.asarray(dist_matrix)
    n = dist_matrix.shape[0]
    if labels is None:
        labels = list(range(n))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist_matrix, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(l) for l in labels], fontsize=8)
    ax.set_yticklabels([str(l) for l in labels], fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_volume_interactive(image):
    """Interactive 3D phantom slicer with Matplotlib sliders.

    Displays axial, coronal, and sagittal planes and exposes sliders for the
    frame index (time) and each spatial slice. Use from a regular Python
    session; for Jupyter prefer plot_volume_jupyter.

    Args:
        image (torch.Tensor): Phase series of shape (num_frames, Nz, Ny, Nx).
    """
    from matplotlib.widgets import Slider

    vol = image.detach().cpu().numpy() if hasattr(image, 'detach') else np.asarray(image)
    num_frames, dim_z, dim_y, dim_x = vol.shape
    init_f, init_z, init_y, init_x = 0, dim_z // 2, dim_y // 2, dim_x // 2

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax_axial = fig.add_subplot(gs[0, 0])
    ax_coronal = fig.add_subplot(gs[:, 1])
    ax_sagittal = fig.add_subplot(gs[:, 2])
    plt.subplots_adjust(bottom=0.25)

    im_axial = ax_axial.imshow(vol[init_f, init_z, :, :], cmap='gray',
                               origin='lower', aspect='auto')
    ax_axial.set_title(f"Axial (Z={init_z})")
    ax_axial.axis('off')

    im_coronal = ax_coronal.imshow(vol[init_f, :, init_y, :], cmap='gray',
                                   aspect='auto', origin='lower')
    ax_coronal.set_title(f"Coronal (Y={init_y})")
    ax_coronal.axis('off')

    im_sagittal = ax_sagittal.imshow(vol[init_f, :, :, init_x], cmap='gray',
                                     aspect='auto', origin='lower')
    ax_sagittal.set_title(f"Sagittal (X={init_x})")
    ax_sagittal.axis('off')

    ax_slider_f = fig.add_axes([0.2, 0.20, 0.60, 0.03])
    ax_slider_z = fig.add_axes([0.2, 0.15, 0.60, 0.03])
    ax_slider_y = fig.add_axes([0.2, 0.10, 0.60, 0.03])
    ax_slider_x = fig.add_axes([0.2, 0.05, 0.60, 0.03])

    slider_f = Slider(ax=ax_slider_f, label='Frame', valmin=0,
                      valmax=num_frames - 1, valinit=init_f, valstep=1)
    slider_z = Slider(ax=ax_slider_z, label='Axial (Z)', valmin=0,
                      valmax=dim_z - 1, valinit=init_z, valstep=1)
    slider_y = Slider(ax=ax_slider_y, label='Coronal (Y)', valmin=0,
                      valmax=dim_y - 1, valinit=init_y, valstep=1)
    slider_x = Slider(ax=ax_slider_x, label='Sagittal (X)', valmin=0,
                      valmax=dim_x - 1, valinit=init_x, valstep=1)

    def update(val):
        f = int(slider_f.val)
        z = int(slider_z.val)
        y = int(slider_y.val)
        x = int(slider_x.val)

        im_axial.set_data(vol[f, z, :, :])
        im_coronal.set_data(vol[f, :, y, :])
        im_sagittal.set_data(vol[f, :, :, x])

        im_axial.set_clim(vmin=vol[f, z, :, :].min(), vmax=vol[f, z, :, :].max())
        im_coronal.set_clim(vmin=vol[f, :, y, :].min(), vmax=vol[f, :, y, :].max())
        im_sagittal.set_clim(vmin=vol[f, :, :, x].min(), vmax=vol[f, :, :, x].max())

        ax_axial.set_title(f"Axial (Z={z})")
        ax_coronal.set_title(f"Coronal (Y={y})")
        ax_sagittal.set_title(f"Sagittal (X={x})")
        fig.canvas.draw()

    slider_f.on_changed(update)
    slider_z.on_changed(update)
    slider_y.on_changed(update)
    slider_x.on_changed(update)

    plt.show()


def plot_volume_jupyter(image):
    """Jupyter-friendly phantom slicer using ipywidgets.

    Args:
        image (torch.Tensor): Phase series of shape (num_frames, Nz, Ny, Nx).
    """
    import ipywidgets as widgets

    vol = image.detach().cpu().numpy() if hasattr(image, 'detach') else np.asarray(image)
    num_frames, dim_z, dim_y, dim_x = vol.shape

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax_axial = fig.add_subplot(gs[0, 0])
    ax_coronal = fig.add_subplot(gs[:, 1])
    ax_sagittal = fig.add_subplot(gs[:, 2])

    im_axial = ax_axial.imshow(vol[0, dim_z // 2, :, :], cmap='gray',
                               origin='lower', aspect='auto')
    im_coronal = ax_coronal.imshow(vol[0, :, dim_y // 2, :], cmap='gray',
                                   aspect='auto', origin='lower')
    im_sagittal = ax_sagittal.imshow(vol[0, :, :, dim_x // 2], cmap='gray',
                                     aspect='auto', origin='lower')

    plt.show()

    def update_plot(f, z, y, x):
        im_axial.set_data(vol[f, z, :, :])
        im_coronal.set_data(vol[f, :, y, :])
        im_sagittal.set_data(vol[f, :, :, x])
        im_axial.set_clim(vmin=vol[f, z, :, :].min(), vmax=vol[f, z, :, :].max())
        im_coronal.set_clim(vmin=vol[f, :, y, :].min(), vmax=vol[f, :, y, :].max())
        im_sagittal.set_clim(vmin=vol[f, :, :, x].min(), vmax=vol[f, :, :, x].max())
        ax_axial.set_title(f"Axial (Z={z})")
        ax_coronal.set_title(f"Coronal (Y={y})")
        ax_sagittal.set_title(f"Sagittal (X={x})")
        fig.canvas.draw_idle()

    widgets.interact(
        update_plot,
        f=widgets.IntSlider(min=0, max=num_frames - 1, value=0, description='Frame'),
        z=widgets.IntSlider(min=0, max=dim_z - 1, value=dim_z // 2, description='Axial'),
        y=widgets.IntSlider(min=0, max=dim_y - 1, value=dim_y // 2, description='Coronal'),
        x=widgets.IntSlider(min=0, max=dim_x - 1, value=dim_x // 2, description='Sagittal'),
    )


def plot_pointcloud_3d(points, max_points=5000):
    """3D scatter of a TOF event point cloud coloured by z.

    Args:
        points (torch.Tensor): TOF point coordinates of shape (N, 3).
        max_points (int): Maximum number of points to plot for performance.
    """
    pts = points.detach().cpu().numpy() if hasattr(points, 'detach') else np.asarray(points)

    if pts.shape[0] > max_points:
        import torch
        idx = torch.randperm(pts.shape[0])[:max_points].numpy()
        pts = pts[idx]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.5,
               c=pts[:, 2], cmap='viridis')
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(f"TOF point cloud ({pts.shape[0]} events)")

    plt.tight_layout()
    plt.show()


def plot_betti_curves(betti_curves, title="Betti curves"):
    """Plot Betti curves for each homology dimension on shared axes.

    Args:
        betti_curves (dict[int, tuple[np.ndarray, np.ndarray]]): Map from
            dimension to (t, beta) as returned by diagrams_to_betti_curves.
        title (str): Plot title.
    """
    dims = sorted(betti_curves.keys())

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4),
                             squeeze=False)
    for i, dim in enumerate(dims):
        t, beta = betti_curves[dim]
        axes[0, i].plot(t, beta)
        axes[0, i].set_title(f"Betti curve B_{dim}(t)")
        axes[0, i].set_xlabel("Filtration (mm)")
        axes[0, i].set_ylabel(f"B_{dim}")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_euler_characteristic(t, chi, title="Euler characteristic"):
    """Plot the Euler characteristic curve chi(t).

    Args:
        t (np.ndarray): Filtration values of shape (n_steps,).
        chi (np.ndarray): Euler characteristic values of shape (n_steps,).
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, chi)
    ax.set_title(title)
    ax.set_xlabel("Filtration (mm)")
    ax.set_ylabel("chi(t)")

    plt.tight_layout()
    plt.show()