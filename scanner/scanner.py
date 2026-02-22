import torch
import parallelproj
import matplotlib.pyplot as plt
from parallelproj import RegularPolygonPETScannerGeometry, RegularPolygonPETLORDescriptor, RegularPolygonPETProjector, SinogramSpatialAxisOrder

import numpy as np
def generate_lors_from_image(image, projector , num_lors, show=False, device='cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """Generates LOR endpoints by sampling from a projected activity image.

    Args:
        image (torch.Tensor): 3D tensor representing the activity distribution.
        projector (RegularPolygonPETProjector): Scanner projector for forward projection.
        num_lors (int): Number of LORs to sample.
        show (bool, optional): Whether to visualize the sinogram and sampled LORs. Defaults to False.
        device (str): Compute device. Defaults to 'cpu'.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Sampled p1 and p2 coordinates (num_lors, 3).
    """

    # Forward project image to create sinogram
    forward = projector(image)
    sino_poisson = torch.poisson(forward)

    # Create sampling weights from the sinogram
    sampling_weights = torch.clamp(sino_poisson.flatten(), min=0)
    if sampling_weights.sum() <= 0:
        raise ValueError("Sinogram is empty. Ensure phantom is within scanner FOV.")

    # Sample LOR indices
    sampling_weights /= sampling_weights.sum()
    lor_indices = torch.multinomial(sampling_weights, num_lors, replacement=True)

    # For visualization: create a binary image of sampled LORs in the sinogram space
    sampled_sinogram = torch.zeros_like(sino_poisson.flatten())
    lor_indices_unique, counts = torch.unique(lor_indices, return_counts=True)
    sampled_sinogram[lor_indices_unique] = counts.float()
    sampled_sinogram = sampled_sinogram.reshape(projector.out_shape)
    
    if show:        
        #fig, ax = plt.subplots(1, 3, figsize=(5, 5))

        slice_idx = projector.out_shape[2] // 2

        """img1 = parallelproj.to_numpy_array(forward[:, :, slice_idx].T)
        ax[0].imshow(img1, cmap="Greys_r", vmin=0)
        ax[0].set_title("Forward Projection")

        img2 = parallelproj.to_numpy_array(sino_poisson[:, :, slice_idx].T)
        ax[1].imshow(img2, cmap="Greys_r", vmin=0)
        ax[1].set_title("Sinogram Poisson Sampled")

        img3 = parallelproj.to_numpy_array(sampled_sinogram[:, :, slice_idx].T)
        ax[2].imshow(img3, cmap="Greys_r", vmin=0)
        ax[2].set_title("Sampled LORs in Sinogram Space")

        plt.tight_layout()
        plt.show()"""
        img3 = parallelproj.to_numpy_array(sampled_sinogram[:, :, slice_idx].T)
        plt.figure(figsize=(5, 5))
        plt.imshow(img3, cmap="Greys_r", vmin=0)
        #plt.title("Sampled LORs in Sinogram Space")
        plt.show()

    # Map indices to 3D endpoint coordinates
    p1_all, p2_all = projector.lor_descriptor.get_lor_coordinates()
    p1_flat = p1_all.reshape(-1, 3)
    p2_flat = p2_all.reshape(-1, 3)

    return p1_flat[lor_indices], p2_flat[lor_indices]

def get_scanner(xp, dev, show = False) -> RegularPolygonPETProjector:
    """Initializes the PET scanner geometry and projector.
    Args:
        xp: Array module (e.g., torch or numpy).
        dev: Compute device (e.g., 'cpu' or 'cuda').
        show (bool): Whether to visualize the scanner geometry.
    Returns:
        proj (RegularPolygonPETProjector): Configured scanner projector."""
    
    num_rings = 5
    scanner = RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=12,
        num_lor_endpoints_per_side=15,
        lor_spacing=2.3,
        ring_positions=xp.linspace(-10, 10, num_rings),
        symmetry_axis=2,
    )

    lor_desc = RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=10,
        sinogram_order=SinogramSpatialAxisOrder.RVP,
        max_ring_difference=2
    )

    proj = RegularPolygonPETProjector(
        lor_descriptor=lor_desc,
        img_shape=(40, 40, 8),
        voxel_size=(2.0, 2.0, 2.0)
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    proj.show_geometry(ax=ax)
    plt.show()

    return proj

def to_2D(proj):
    dev = proj._dev
    scanner = RegularPolygonPETScannerGeometry(
    proj.xp,
    dev,
    radius=proj.lor_descriptor.scanner.__getattribute__('radius'),
    num_sides=proj.lor_descriptor.scanner.__getattribute__('num_sides'),
    num_lor_endpoints_per_side= proj.lor_descriptor.scanner.__getattribute__('num_lor_endpoints_per_side'),
    lor_spacing=proj.lor_descriptor.scanner.__getattribute__('lor_spacing'),
    ring_positions=proj.xp.linspace(0,0,1),
    symmetry_axis=proj.lor_descriptor.scanner.__getattribute__('symmetry_axis'),
    )

    lor_desc = RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=proj.lor_descriptor.__getattribute__('radial_trim'),
        max_ring_difference=proj.lor_descriptor.__getattribute__('max_ring_difference'),
        sinogram_order=proj.lor_descriptor.__getattribute__('sinogram_order'),
    )
    
    proj = RegularPolygonPETProjector(
        lor_desc, img_shape=(proj.in_shape[0], proj.in_shape[0], 1), voxel_size=proj.__getattribute__('voxel_size')
    )
    return scanner, lor_desc, proj