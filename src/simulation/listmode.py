import torch


def sample_events_from_sinogram(sinogram, num_events):
    """Sample list-mode event indices from a precomputed sinogram via inverse-CDF.

    PET sinograms carry large structurally-zero regions (outside the axial FOV,
    low-sensitivity TOF bins), so the CDF is built over the nonzero bins only
    and the sampled indices are remapped back to the original flat layout.

    Args:
        sinogram (torch.Tensor): Sinogram of shape matching proj.out_shape.
        num_events (int): Number of events to sample.

    Returns:
        torch.Tensor: Flat sinogram indices of shape (num_events,).
    """
    flat = sinogram.flatten().to(dtype=torch.float32)
    weights = torch.clamp(flat, min=0)

    nz_idx = torch.nonzero(weights, as_tuple=False).squeeze(-1)
    if nz_idx.numel() == 0:
        raise ValueError("Sinogram is empty. Check phantom position.")

    cdf = torch.cumsum(weights[nz_idx], dim=0)
    total = cdf[-1]
    if total <= 0:
        raise ValueError("Sinogram is empty. Check phantom position.")

    r = torch.rand(num_events, device=weights.device, dtype=weights.dtype) * total
    sampled = torch.searchsorted(cdf, r)
    return nz_idx[sampled]


def sample_events(image, proj, num_events):
    """Sample list-mode event indices from an activity image via forward projection.

    Args:
        image (torch.Tensor): 3D activity image of shape (Nz, Ny, Nx).
        proj (RegularPolygonPETProjector): Configured parallelproj projector.
        num_events (int): Number of events to sample.

    Returns:
        torch.Tensor: Flat sinogram indices of shape (num_events,).
    """
    return sample_events_from_sinogram(proj(image), num_events)


def get_lor_endpoints(indices, proj):
    """Get detector coordinates and TOF bin indices for sampled events.

    Args:
        indices (torch.Tensor): LOR indices of shape (N,).
        proj (RegularPolygonPETProjector): Configured parallelproj projector.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: Start and end
            detector coordinates each of shape (N, 3), and TOF bin indices of
            shape (N,) or None if non-TOF.
    """
    p1, p2 = proj.lor_descriptor.get_lor_coordinates()
    p1, p2 = p1.reshape(-1, 3), p2.reshape(-1, 3)

    if proj.tof_parameters is not None:
        num_tofbins = proj.tof_parameters.num_tofbins
        tof_bins = (indices % num_tofbins) - (num_tofbins // 2)
        lor_indices = torch.div(indices, num_tofbins, rounding_mode="floor")
        return p1[lor_indices], p2[lor_indices], tof_bins

    return p1[indices], p2[indices], None


def mash_sinogram_indices(indices, proj, mash_radial=1, mash_view=1,
                          mash_plane=1, mash_tof=1):
    """Collapse sinogram bins by integer factors; keep one LOR per coarse cell.

    Decomposes each flat index into (radial, view, plane[, tof]) using the
    projector's C-order layout, integer-divides each axis by its mash factor,
    deduplicates the coarse cells, and re-encodes each surviving cell to the
    flat index of its cell-centre fine bin. Physically equivalent to reducing
    detector pitch / TOF resolution prior to list-mode binning — standard PET
    "mashing". The returned indices are still valid inputs to
    ``get_lor_endpoints``.
    """
    out_shape = proj.out_shape
    has_tof = proj.tof_parameters is not None
    if has_tof:
        R, V, P, T = out_shape
    else:
        R, V, P = out_shape
        T = 1
        mash_tof = 1

    if min(mash_radial, mash_view, mash_plane, mash_tof) < 1:
        raise ValueError('mash factors must be >= 1')

    flat = indices.to(torch.int64)
    if has_tof:
        t = flat % T
        rest = torch.div(flat, T, rounding_mode='floor')
    else:
        t = torch.zeros_like(flat)
        rest = flat
    p = rest % P
    rest = torch.div(rest, P, rounding_mode='floor')
    v = rest % V
    r = torch.div(rest, V, rounding_mode='floor')

    cr = torch.div(r, mash_radial, rounding_mode='floor')
    cv = torch.div(v, mash_view, rounding_mode='floor')
    cp = torch.div(p, mash_plane, rounding_mode='floor')
    ct = torch.div(t, mash_tof, rounding_mode='floor')

    n_cv = (V + mash_view - 1) // mash_view
    n_cp = (P + mash_plane - 1) // mash_plane
    n_ct = (T + mash_tof - 1) // mash_tof

    coarse_key = ((cr * n_cv + cv) * n_cp + cp) * n_ct + ct
    unique_keys = torch.unique(coarse_key)

    u_ct = unique_keys % n_ct
    rest = torch.div(unique_keys, n_ct, rounding_mode='floor')
    u_cp = rest % n_cp
    rest = torch.div(rest, n_cp, rounding_mode='floor')
    u_cv = rest % n_cv
    u_cr = torch.div(rest, n_cv, rounding_mode='floor')

    f_r = torch.clamp(u_cr * mash_radial + mash_radial // 2, max=R - 1)
    f_v = torch.clamp(u_cv * mash_view + mash_view // 2, max=V - 1)
    f_p = torch.clamp(u_cp * mash_plane + mash_plane // 2, max=P - 1)
    f_t = torch.clamp(u_ct * mash_tof + mash_tof // 2, max=T - 1)

    if has_tof:
        fine_flat = ((f_r * V + f_v) * P + f_p) * T + f_t
    else:
        fine_flat = (f_r * V + f_v) * P + f_p

    return fine_flat.to(indices.dtype)


def indices_to_sinogram(indices, proj):
    """Convert flat LOR indices to a sinogram by counting events per bin.

    Args:
        indices (torch.Tensor): Flat sinogram indices of shape (N,).
        proj (RegularPolygonPETProjector): Configured parallelproj projector.

    Returns:
        torch.Tensor: Sinogram of shape matching proj.out_shape.
    """
    sinogram = torch.zeros(proj.out_shape, dtype=torch.float32, device=indices.device)
    sinogram.flatten().scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
    return sinogram
