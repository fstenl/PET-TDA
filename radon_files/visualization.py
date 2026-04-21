import array_api_compat.torch as xp
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import parallelproj
from IPython.display import display
import ipywidgets as widgets


def visualize_image(image, show: bool = True):
    """Create interactive visualization of 3D medical image with orthogonal planes and sliders."""
    num_frames, dim_z, dim_y, dim_x = image.shape

    # Initial indices
    init_f = 0
    init_z = dim_z // 2
    init_y = dim_y // 2
    init_x = dim_x // 2

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Axial in top-left
    ax_axial = fig.add_subplot(gs[0, 0])
    # Coronal and Sagittal span full height on the right
    ax_coronal = fig.add_subplot(gs[:, 1])
    ax_sagittal = fig.add_subplot(gs[:, 2])

    # Leave room at the bottom for the 4 sliders
    plt.subplots_adjust(bottom=0.25)

    # Display initial slices
    # Axial Plane (XY)
    im_axial = ax_axial.imshow(image[init_f, init_z, :, :], cmap='gray', origin='lower', aspect='auto')
    ax_axial.set_title(f"Axial (Z={init_z})")
    ax_axial.axis('off')

    # Coronal Plane (XZ)
    im_coronal = ax_coronal.imshow(image[init_f, :, init_y, :], cmap='gray', aspect='auto', origin='lower')
    ax_coronal.set_title(f"Coronal (Y={init_y})")
    ax_coronal.axis('off')

    # Sagittal Plane (YZ)
    im_sagittal = ax_sagittal.imshow(image[init_f, :, :, init_x], cmap='gray', aspect='auto', origin='lower')
    ax_sagittal.set_title(f"Sagittal (X={init_x})")
    ax_sagittal.axis('off')

    # Define slider axes
    ax_slider_f = fig.add_axes([0.2, 0.20, 0.60, 0.03])
    ax_slider_z = fig.add_axes([0.2, 0.15, 0.60, 0.03])
    ax_slider_y = fig.add_axes([0.2, 0.10, 0.60, 0.03])
    ax_slider_x = fig.add_axes([0.2, 0.05, 0.60, 0.03])

    # Create sliders
    slider_f = Slider(ax=ax_slider_f, label='Frame (Time)', valmin=0, valmax=num_frames - 1, valinit=init_f, valstep=1)
    slider_z = Slider(ax=ax_slider_z, label='Axial Slice (Z)', valmin=0, valmax=dim_z - 1, valinit=init_z, valstep=1)
    slider_y = Slider(ax=ax_slider_y, label='Coronal Slice (Y)', valmin=0, valmax=dim_y - 1, valinit=init_y, valstep=1)
    slider_x = Slider(ax=ax_slider_x, label='Sagittal Slice (X)', valmin=0, valmax=dim_x - 1, valinit=init_x, valstep=1)

    # Update function called when any slider is moved
    def update(val):
        f = int(slider_f.val)
        z = int(slider_z.val)
        y = int(slider_y.val)
        x = int(slider_x.val)

        # Update image data for each plane
        im_axial.set_data(image[f, z, :, :])
        im_coronal.set_data(image[f, :, y, :])
        im_sagittal.set_data(image[f, :, :, x])

        # Adjust contrast dynamically for each plane
        im_axial.set_clim(vmin=image[f, z, :, :].min(), vmax=image[f, z, :, :].max())
        im_coronal.set_clim(vmin=image[f, :, y, :].min(), vmax=image[f, :, y, :].max())
        im_sagittal.set_clim(vmin=image[f, :, :, x].min(), vmax=image[f, :, :, x].max())

        # Update titles
        ax_axial.set_title(f"Axial (Z={z})")
        ax_coronal.set_title(f"Coronal (Y={y})")
        ax_sagittal.set_title(f"Sagittal (X={x})")

        fig.canvas.draw()

    # Register the update function
    slider_f.on_changed(update)
    slider_z.on_changed(update)
    slider_y.on_changed(update)
    slider_x.on_changed(update)
    if show:
        display(fig.canvas)


def visualize_image_jupyter(image):
    """Create interactive Jupyter visualization of 3D medical image using ipywidgets."""
    num_frames, dim_z, dim_y, dim_x = image.shape

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax_axial = fig.add_subplot(gs[0, 0])
    ax_coronal = fig.add_subplot(gs[:, 1])
    ax_sagittal = fig.add_subplot(gs[:, 2])

    im_axial = ax_axial.imshow(image[0, dim_z//2, :, :], cmap='gray', origin='lower', aspect='auto')
    im_coronal = ax_coronal.imshow(image[0, :, dim_y//2, :], cmap='gray', aspect='auto', origin='lower')
    im_sagittal = ax_sagittal.imshow(image[0, :, :, dim_x//2], cmap='gray', aspect='auto', origin='lower')

    plt.show()

    # Create the interactive function
    def update_plot(f, z, y, x):
        im_axial.set_data(image[f, z, :, :])
        im_coronal.set_data(image[f, :, y, :])
        im_sagittal.set_data(image[f, :, :, x])

        im_axial.set_clim(vmin=image[f, z, :, :].min(), vmax=image[f, z, :, :].max())
        im_coronal.set_clim(vmin=image[f, :, y, :].min(), vmax=image[f, :, y, :].max())
        im_sagittal.set_clim(vmin=image[f, :, :, x].min(), vmax=image[f, :, :, x].max())

        ax_axial.set_title(f"Axial (Z={z})")
        ax_coronal.set_title(f"Coronal (Y={y})")
        ax_sagittal.set_title(f"Sagittal (X={x})")

        fig.canvas.draw_idle()

    # Generate the sliders using ipywidgets
    widgets.interact(update_plot,
                     f=widgets.IntSlider(min=0, max=num_frames-1, value=0, description='Frame'),
                     z=widgets.IntSlider(min=0, max=dim_z-1, value=dim_z//2, description='Axial'),
                     y=widgets.IntSlider(min=0, max=dim_y-1, value=dim_y//2, description='Coronal'),
                     x=widgets.IntSlider(min=0, max=dim_x-1, value=dim_x//2, description='Sagittal'))


def visualise_sinogram(x_fwd, proj, show: bool = True):
    """Visualizes the sampled sinogram."""
    print(f"Sinogram shape: {x_fwd.shape}")
    print(f"Number of planes: {proj.lor_descriptor.num_planes}")

    # Debug specific planes
    for i in [8, 9, 10, 11]:
        plane_data = x_fwd[:, :, i]
        print(f"Plane {i}: min={parallelproj.to_numpy_array(xp.min(plane_data)):.4f}, max={parallelproj.to_numpy_array(xp.max(plane_data)):.4f}, mean={parallelproj.to_numpy_array(xp.mean(plane_data)):.4f}")

    fig, ax = plt.subplots(4, 5, figsize=(2 * 5, 2 * 4))
    vmax = float(xp.max(x_fwd))
    for i in range(20):
        axx = ax.ravel()[i]
        if i < proj.lor_descriptor.num_planes:
            axx.imshow(
                parallelproj.to_numpy_array(x_fwd[:, :, i].T),
                cmap="Greys",
                vmin=0,
                vmax=vmax,
            )
            axx.set_title(f"sino plane {i}", fontsize="medium")
        else:
            axx.set_axis_off()
    fig.tight_layout()
    if show:
        plt.show()


def visualize_sinogram_tof(sino_data, title='TOF Sinogram', show=False):
    """Helper for 4D TOF sinograms."""
    sino_data = parallelproj.to_numpy_array(sino_data)

    n_planes = sino_data.shape[2]
    n_tofbins = sino_data.shape[3]

    rows_to_show = min(7, n_planes)
    cols_to_show = min(9, n_tofbins)

    fig, ax = plt.subplots(rows_to_show, cols_to_show,
                            figsize=(1.4 * cols_to_show, 1.2 * rows_to_show),
                            sharex=True, sharey=True)
    vmax = float(sino_data.max())

    # Handle single row/col edge cases
    if rows_to_show == 1 and cols_to_show == 1: ax = np.array([[ax]])
    elif rows_to_show == 1: ax = ax[np.newaxis, :]
    elif cols_to_show == 1: ax = ax[:, np.newaxis]

    step = max(1, n_planes // rows_to_show)

    for i in range(rows_to_show):
        plane_idx = i * step

        for j in range(cols_to_show):
            center_bin = n_tofbins // 2
            start_bin_offset = center_bin - (cols_to_show // 2)
            tof_bin_idx = start_bin_offset + j

            if 0 <= tof_bin_idx < n_tofbins:
                ax[i, j].imshow(
                    sino_data[:, :, plane_idx, tof_bin_idx],
                    cmap="Greys_r", vmin=0, vmax=vmax, origin="lower"
                )

                if i == 0:
                    ax[i, j].set_title(f"Bin {tof_bin_idx - center_bin}", fontsize="small")

            if j == 0:
                ax[i, j].set_ylabel(f"Slice {plane_idx}", fontsize="small")

            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show()


def plot_3d(points, num_display=5000):
    """3D scatter plot of event point cloud."""
    pts = points.cpu().numpy()
    if pts.shape[0] > num_display:
        indices = torch.randperm(pts.shape[0])[:num_display]
        pts = pts[indices]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.5, c=pts[:, 2], cmap='viridis')
    ax.set_xlabel('Axial (mm)')
    ax.set_ylabel('Coronal (mm)')
    ax.set_zlabel('Sagittal (mm)')
    ax.set_title(f'TOF Point Cloud ({num_display} events)')
    plt.show()
