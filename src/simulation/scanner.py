import torch
import array_api_compat.torch as xp
from parallelproj import (
    RegularPolygonPETScannerGeometry,
    RegularPolygonPETLORDescriptor,
    RegularPolygonPETProjector,
    TOFParameters
)


def get_mini_projector(device='cpu', img_shape=(20, 80, 80), voxel_size=(1.0, 1.0, 1.0), tof=True):
    """Build a small polygonal PET projector.

    Args:
        device (str): Torch device, e.g. "cpu" or "cuda".
        img_shape (tuple[int, int, int]): Image grid dimensions (Nz, Ny, Nx).
        voxel_size (tuple[float, float, float]): Voxel size in mm (dz, dy, dx).
        tof (bool): If True, attach TOF parameters.

    Returns:
        RegularPolygonPETProjector: Configured projector.
    """
    scanner = RegularPolygonPETScannerGeometry(
        xp,
        device,
        radius=65.0,
        num_sides=12,
        num_lor_endpoints_per_side=15,
        lor_spacing=2.3,
        ring_positions=torch.linspace(-20, 20, 10, device=device),
        symmetry_axis=0
    )

    lor_desc = RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=1,
        max_ring_difference=12,
    )

    proj = RegularPolygonPETProjector(
        lor_desc,
        img_shape=img_shape,
        voxel_size=voxel_size
    )

    if tof:
        proj.tof_parameters = TOFParameters(
            num_tofbins=13,
            tofbin_width=10.0,
            sigma_tof=5.0,
        )

    return proj


def get_mct_projector(device='cpu', img_shape=(109, 200, 200), voxel_size=(2.0, 4.0, 4.0), tof=True):
    """Build a projector for the Siemens Biograph mCT.

    Args:
        device (str): Torch device, e.g. "cpu" or "cuda".
        img_shape (tuple[int, int, int]): Image grid dimensions (Nz, Ny, Nx).
        voxel_size (tuple[float, float, float]): Voxel size in mm (dz, dy, dx).
        tof (bool): If True, attach TOF parameters.

    Returns:
        RegularPolygonPETProjector: Configured projector.
    """
    scanner = RegularPolygonPETScannerGeometry(
        xp,
        device,
        radius=421.0,
        num_sides=48,
        num_lor_endpoints_per_side=13,
        lor_spacing=4.0,
        ring_positions=torch.linspace(-109, 109, 55, device=device),  # 4 * 13 + 3
        symmetry_axis=0
    )

    lor_desc = RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=151,
        max_ring_difference=49,
    )

    proj = RegularPolygonPETProjector(
        lor_desc,
        img_shape=img_shape,
        voxel_size=voxel_size,
    )

    if tof:
        proj.tof_parameters = TOFParameters(
            num_tofbins=13,
            tofbin_width=46.8,
            sigma_tof=33.6,
        )

    return proj