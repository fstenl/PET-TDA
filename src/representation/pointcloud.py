import torch


def localize_events(p1, p2, tof_bins, proj):
    """Compute 3D spatial coordinates of TOF bin centers for sampled events.

    Each event is localized to the center of its TOF bin along the LOR,
    measured from the midpoint of the LOR in physical units.

    Args:
        p1 (torch.Tensor): LOR start coordinates of shape (N, 3).
        p2 (torch.Tensor): LOR end coordinates of shape (N, 3).
        tof_bins (torch.Tensor): TOF bin indices of shape (N,), signed integers
            where 0 is the bin centered at the LOR midpoint.
        proj (RegularPolygonPETProjector): Configured parallelproj projector
            with tof_parameters set.

    Returns:
        torch.Tensor: TOF bin center coordinates of shape (N, 3).
    """
    tofbin_width = proj.tof_parameters.tofbin_width

    midpoints = 0.5 * (p1 + p2)
    directions = p2 - p1
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True)

    offsets = tof_bins.to(dtype=torch.float32).unsqueeze(1) * tofbin_width * directions
    return midpoints + offsets