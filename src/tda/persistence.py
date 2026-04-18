import numpy as np
import torch
import gudhi
from ripser import ripser


def compute_persistence_pointcloud(points, max_dim=1):
    """Compute persistence diagrams from a point cloud using Vietoris-Rips.

    Args:
        points (torch.Tensor or np.ndarray): Point coordinates of shape (N, D).
        max_dim (int): Maximum homology dimension to compute.

    Returns:
        list[np.ndarray]: Persistence diagrams, one array per dimension.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    result = ripser(points, maxdim=max_dim)
    return result['dgms']


def compute_persistence_volume(volume, max_dim=2):
    """Compute persistence diagrams from an image volume using cubical homology.

    Sublevel-set filtration is applied, meaning topological features are tracked
    as the intensity threshold increases from min to max voxel value. Infinite
    death times are removed before returning.

    Args:
        volume (torch.Tensor or np.ndarray): Image volume of shape (N0, N1, ...).
        max_dim (int): Maximum homology dimension to compute.

    Returns:
        list[np.ndarray]: Persistence diagrams, one array per dimension.
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    cubical = gudhi.CubicalComplex(top_dimensional_cells=volume)
    cubical.compute_persistence()

    diagrams = []
    for dim in range(max_dim + 1):
        pairs = cubical.persistence_intervals_in_dimension(dim)
        if len(pairs) > 0:
            pairs = np.array(pairs)
            pairs = pairs[np.isfinite(pairs).all(axis=1)]
            diagrams.append(pairs)
        else:
            diagrams.append(np.empty((0, 2)))

    return diagrams