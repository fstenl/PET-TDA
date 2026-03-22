import os
import torch
import parallelproj
import matplotlib.pyplot as plt
from parallelproj import RegularPolygonPETScannerGeometry, RegularPolygonPETLORDescriptor, RegularPolygonPETProjector, SinogramSpatialAxisOrder

import numpy as np
def generate_lors_from_image(
    image: torch.Tensor,
    projector: RegularPolygonPETProjector,
    num_lors: int,
    show: bool = False,
    return_sinogram: bool = False,
    return_adjoint: bool = False,
    save_path: str | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, np.ndarray]
    | tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
):
    """Generates LOR endpoints by sampling from a projected activity image.

    Args:
        image (torch.Tensor): 3D tensor representing the activity distribution.
        projector (RegularPolygonPETProjector): Scanner projector for forward projection.
        num_lors (int): Number of LORs to sample.
        show (bool, optional): Whether to visualize the sinogram and sampled LORs.
            Not thread-safe — use ``return_sinogram=True`` and plot on the main
            thread when running in parallel. Defaults to False.
        return_sinogram (bool, optional): If True, return the sampled sinogram
            image (numpy array) as an additional element so calling code can
            plot it later on the main thread. Defaults to False.
        return_adjoint (bool, optional): If True, return the adjoint
            (back-projected) image of the sampled sinogram as a numpy array.
            Useful for visualising which voxels are covered by the sampled
            LORs. Defaults to False.

    Returns:
        (p1, p2) when both ``return_sinogram`` and ``return_adjoint`` are False.
        (p1, p2, sinogram_image) when only ``return_sinogram`` is True.
        (p1, p2, sinogram_image, adjoint_image) when both are True.
        (p1, p2, adjoint_image) when only ``return_adjoint`` is True.
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

    # Pick the plane with the most counts so the visualisation always shows
    plane_sums = sampled_sinogram.sum(dim=(0, 1))
    slice_idx = int(plane_sums.argmax())
    sinogram_img = parallelproj.to_numpy_array(sampled_sinogram[:, :, slice_idx].T)

    # Compute the adjoint (back-projection) of the sampled sinogram
    adjoint_img = None
    if return_adjoint:
        adjoint_vol = projector.adjoint(sampled_sinogram)
        mid_slice = adjoint_vol.shape[2] // 2
        adjoint_img = parallelproj.to_numpy_array(adjoint_vol[:, :, mid_slice].T)

    if show:
        fig, ax = plt.subplots()
        ax.imshow(sinogram_img, cmap="Greys_r", vmin=0)
        ax.set_title("Sampled LORs")
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    # Map indices to 3D endpoint coordinates
    p1_all, p2_all = projector.lor_descriptor.get_lor_coordinates()
    p1_flat = p1_all.reshape(-1, 3)
    p2_flat = p2_all.reshape(-1, 3)

    result = (p1_flat[lor_indices], p2_flat[lor_indices])
    if return_sinogram:
        result = result + (sinogram_img,)
    if return_adjoint:
        result = result + (adjoint_img,)

    return result

def get_mini_scanner(xp, dev, show: bool = False, save_path: str | None = None) -> RegularPolygonPETProjector:
    """Initializes the PET scanner geometry and projector.
    Args:
        xp: Array module (e.g., torch or numpy).
        dev: Compute device (e.g., 'cpu' or 'cuda').
        show (bool): Whether to visualize the scanner geometry.
    Returns:
        proj (RegularPolygonPETProjector): Configured scanner projector."""
    
    num_rings = 10
    scanner = RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=12,
        num_lor_endpoints_per_side=15,
        lor_spacing=2.3,
        ring_positions=xp.linspace(-20, 20, num_rings),
        symmetry_axis=2,
    )

    lor_desc = RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=10,
        sinogram_order=SinogramSpatialAxisOrder.RVP,
        max_ring_difference=8
    )

    proj = RegularPolygonPETProjector(
        lor_descriptor=lor_desc,
        img_shape=(40, 40, 16),
        voxel_size=(2.0, 2.0, 2.0)
    )

    if show:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        proj.show_geometry(ax=ax)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    return proj

def get_mCT_scanner(xp, dev, show: bool = False, save_path: str | None = None) -> RegularPolygonPETProjector:
    """Initializes the mCT PET scanner geometry and projector.

    Args:
        xp: Array module (e.g., torch or numpy).
        dev: Compute device (e.g., 'cpu' or 'cuda').
        show (bool): Whether to visualize the scanner geometry.
    Returns:
        proj (RegularPolygonPETProjector): Configured scanner projector."""
    
    num_rings = 55 
    radius_mm = (4 * 13* 48 + 48*3)/(2*torch.pi) # Approximate mCT scanner radius in mm
    axial_fov_half = 109.0 # Approximate half axial FOV in mm

    scanner = RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=radius_mm,
        num_sides=48,
        num_lor_endpoints_per_side=13,
        lor_spacing=4.0,
        ring_positions=xp.linspace(-axial_fov_half, axial_fov_half, num_rings),
        symmetry_axis=2,
    )

    lor_desc = RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=15,
        sinogram_order=SinogramSpatialAxisOrder.RVP,
        max_ring_difference=49
    )

    proj = RegularPolygonPETProjector(
        lor_descriptor=lor_desc,
        img_shape=(128, 128, 55),
        voxel_size=(4.0, 4.0, 4.0)
    )

    if show:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        proj.show_geometry(ax=ax)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
    return proj




def to_2D(proj: RegularPolygonPETProjector) -> tuple[RegularPolygonPETScannerGeometry, RegularPolygonPETLORDescriptor, RegularPolygonPETProjector]:
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