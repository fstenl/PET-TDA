from scanner.scanner import get_mCT_scanner, get_mini_scanner
import array_api_compat.torch as xp
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import parallelproj
from phantoms.primitives import create_sphere_phantom

 
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
        
        fig.canvas.draw_idle()

    # Register the update function
    slider_f.on_changed(update)
    slider_z.on_changed(update)
    slider_y.on_changed(update)
    slider_x.on_changed(update)
    if show:
        plt.show()

def visualise_sinogram(x_fwd, show: bool = True):
    """Visualizes the sampled sinogram """
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

    
if __name__ == "__main__":
    dev = get_device()
    print(f"Using device: {dev}")

    proj = get_mCT_scanner(xp, dev, show=False)
   
    xcat = np.load("data/respiratory_only.npy")
    print(f"Image shape: {xcat.shape}")
    visualize_image(xcat, show = False)
    image_to_project = xp.asarray(xcat[0,30:85,:,:])  # Take the first frame for projection

    forward_proj = proj(image_to_project)
    visualise_sinogram(forward_proj, show= True)





