import torch


def sample_events(image, proj, num_events):
    """Sample list-mode event indices from an activity image via forward projection.

    Args:
        image (torch.Tensor): 3D activity image of shape (Nz, Ny, Nx).
        proj (RegularPolygonPETProjector): Configured parallelproj projector.
        num_events (int): Number of events to sample.

    Returns:
        torch.Tensor: Flat sinogram indices of shape (num_events,).
    """
    sinogram = proj(image)
    weights = sinogram.flatten().to(dtype=torch.float32)
    weights = torch.clamp(weights, min=0)

    cdf = torch.cumsum(weights, dim=0)

    if cdf[-1] <= 0:
        raise ValueError("Sinogram is empty. Check phantom position.")

    r = torch.rand(num_events, device=weights.device, dtype=weights.dtype) * cdf[-1]
    return torch.searchsorted(cdf, r)


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