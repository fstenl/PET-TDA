import os
import parallelproj
import matplotlib.pyplot as plt
from parallelproj import (
    RegularPolygonPETScannerGeometry,
    RegularPolygonPETLORDescriptor,
    RegularPolygonPETProjector,
    SinogramSpatialAxisOrder,
)
import numpy as np
from array_api_compat import array_namespace


def sample_events_2(image, proj, n_events):
    """Sample ``n_events`` list-mode event indices from a phantom image (CDF method).

    Backend-agnostic: works with whatever array namespace ``proj`` is using
    (torch, cupy, numpy) — no torch-specific calls.
    """
    sinogram = proj(image)
    return sample_events_from_sinogram(sinogram, n_events)


def sample_events_from_sinogram(sinogram, n_events):
    """Sample ``n_events`` list-mode event indices from a precomputed sinogram.

    PET sinograms carry large structurally-zero regions (outside the axial
    FOV, low-sensitivity TOF bins).  We build the CDF over the nonzero bins
    only, which shrinks the cumsum — typically the hot path — by several-fold,
    then remap the sampled indices back to the original flat layout.
    """
    xp = array_namespace(sinogram)

    flat = xp.reshape(sinogram, (-1,))
    weights = xp.clip(flat, 0.0, None)

    nz_idx = xp.nonzero(weights)[0]
    if nz_idx.shape[0] == 0:
        raise ValueError("Sinogram is empty. Check phantom position.")

    cdf = xp.cumsum(weights[nz_idx], axis=0)
    total = cdf[-1]

    # Uniform draws in [0, total).  array-api doesn't spec random, so go via
    # numpy and move the result into xp's namespace.
    r_np = np.random.default_rng().random(n_events) * float(total)
    r = xp.asarray(r_np, dtype=cdf.dtype, device=_device_of(cdf))

    sampled = xp.searchsorted(cdf, r)
    return nz_idx[sampled]


def _device_of(arr):
    """Best-effort device extraction that works across torch / cupy / numpy."""
    # array-api ``.device`` exists on torch >= 2.1 and cupy; numpy lacks it.
    return getattr(arr, "device", None)


def get_event_coords(indices, proj):
    """Map event indices to 3D coordinates (TOF midpoints or LOR endpoints).

    Backend-agnostic: all ops go through ``proj.xp``, so this runs under torch
    or cupy without change.  ``indices`` may be a plain numpy array or an ``xp``
    array; it is coerced into ``proj.xp``.
    """
    xp = proj.xp

    indices = xp.asarray(indices)

    lor_desc = proj.lor_descriptor
    p1, p2 = lor_desc.get_lor_coordinates()
    p1 = xp.reshape(p1, (-1, 3))
    p2 = xp.reshape(p2, (-1, 3))

    if proj.tof_parameters is not None:
        n_tof = proj.tof_parameters.num_tofbins
        w_tof = proj.tof_parameters.tofbin_width

        lor_idx = indices // n_tof
        tof_idx = indices % n_tof

        x1, x2 = p1[lor_idx], p2[lor_idx]
        dist = (xp.astype(tof_idx, p1.dtype) - (n_tof - 1) / 2.0) * w_tof

        vec = x2 - x1
        unit = vec / xp.linalg.vector_norm(vec, axis=1, keepdims=True)
        return (x1 + x2) / 2.0 + unit * dist[:, None]

    return p1[indices], p2[indices]


def get_mini_scanner(xp, dev, show: bool = False, save_path: str | None = None) -> RegularPolygonPETProjector:
    """Initializes a small PET scanner geometry and projector."""
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
    """Initializes the mCT PET scanner geometry and projector."""
    num_rings = 55
    radius_mm = (4 * 13 * 48 + 48 * 3) / (2 * np.pi)   # plain float, backend-agnostic
    axial_fov_half = 109.0

    scanner = RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=radius_mm,
        num_sides=48,
        num_lor_endpoints_per_side=13,
        lor_spacing=4.0,
        ring_positions=xp.linspace(-axial_fov_half, axial_fov_half, num_rings),
        symmetry_axis=0,
    )

    lor_desc = RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=125,
        sinogram_order=SinogramSpatialAxisOrder.RVP,
        max_ring_difference=49
    )

    proj = RegularPolygonPETProjector(
        lor_descriptor=lor_desc,
        img_shape=(109, 128, 128),
        voxel_size=(2.0, 4.0, 4.0)
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

    proj.tof_parameters = parallelproj.TOFParameters(num_tofbins=13, tofbin_width=46.8, sigma_tof=33.58)
    return proj


def to_2D(proj: RegularPolygonPETProjector) -> tuple[RegularPolygonPETScannerGeometry, RegularPolygonPETLORDescriptor, RegularPolygonPETProjector]:
    dev = proj._dev
    scanner = RegularPolygonPETScannerGeometry(
        proj.xp,
        dev,
        radius=proj.lor_descriptor.scanner.__getattribute__('radius'),
        num_sides=proj.lor_descriptor.scanner.__getattribute__('num_sides'),
        num_lor_endpoints_per_side=proj.lor_descriptor.scanner.__getattribute__('num_lor_endpoints_per_side'),
        lor_spacing=proj.lor_descriptor.scanner.__getattribute__('lor_spacing'),
        ring_positions=proj.xp.linspace(0, 0, 1),
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
