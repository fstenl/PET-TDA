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
