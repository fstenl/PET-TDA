import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from persim import bottleneck, wasserstein


def compute_bottleneck_distance(dgm1: list, dgm2: list, hom_dim: int = 1) -> float:
    """
    Computes the Bottleneck distance between two persistence diagrams.
    Measures the maximum distance any single point has to move.

    Args:
        dgm1, dgm2: Persistence diagrams (from ripser).
        hom_dim (int): Homology dimension to compare (0=H0, 1=H1).

    Returns:
        float: The bottleneck distance.
    """
    # Extract the specific dimension (e.g., H1 for loops)
    return bottleneck(dgm1[hom_dim], dgm2[hom_dim])


def compute_wasserstein_distance(dgm1: list, dgm2: list, hom_dim: int = 1) -> float:
    """
    Computes the Wasserstein distance between two persistence diagrams.
    Measures the 'total cost' of moving all points from one diagram to the other.

    Args:
        dgm1, dgm2: Persistence diagrams.
        hom_dim (int): Homology dimension to compare.

    Returns:
        float: The wasserstein distance.
    """
    return wasserstein(dgm1[hom_dim], dgm2[hom_dim])


def compute_trajectory_distances(diagram_list: list, method: str = 'wasserstein', hom_dim: int = 1) -> list[float]:
    """
    Computes the topological distance between consecutive frames in a sequence.

    Args:
        diagram_list (list): List of diagrams generated from the frame sequence.
        method (str): 'bottleneck' or 'wasserstein'.
        hom_dim (int): Homology dimension (usually 1 for loops).

    Returns:
        list: Distances between frame t and t+1.
    """
    distances = []
    dist_func = wasserstein if method == 'wasserstein' else bottleneck

    for i in range(len(diagram_list) - 1):
        d = dist_func(diagram_list[i][hom_dim], diagram_list[i + 1][hom_dim])
        distances.append(d)

    return distances


def _compute_pair_distance(diagram_list, i, j, dist_func, hom_dim):
    """Compute the distance between diagrams of frame i and frame j."""
    dgm1 = diagram_list[i][hom_dim]
    dgm2 = diagram_list[j][hom_dim]

    dgm1_finite = dgm1[np.isfinite(dgm1[:, 1])]
    dgm2_finite = dgm2[np.isfinite(dgm2[:, 1])]

    if len(dgm1_finite) > 0 and len(dgm2_finite) > 0:
        return i, j, dist_func(dgm1_finite, dgm2_finite)
    return i, j, 0.0


def compute_all_pairs_distances(diagram_list: list, method: str = 'wasserstein', hom_dim: int = 1, max_workers: int | None = None) -> np.ndarray:
    """
    Computes a distance matrix comparing every frame's diagram to every other frame,
    handling infinite death times commonly found in H0.
    Pairs are computed in parallel using concurrent.futures.

    Args:
        diagram_list (list): List of diagrams for each frame.
        method (str): 'wasserstein' or 'bottleneck'.
        hom_dim (int): Homology dimension (usually 1 for loops).
        max_workers (int | None): Number of threads (None = os.cpu_count()).

    Returns:
        np.ndarray: A square symmetric matrix of distances.
    """
    num_frames = len(diagram_list)
    dist_matrix = np.zeros((num_frames, num_frames))
    dist_func = wasserstein if method == 'wasserstein' else bottleneck

    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        future_to_pair = {
            executor.submit(
                _compute_pair_distance, diagram_list, i, j, dist_func, hom_dim
            ): (i, j)
            for i in range(num_frames)
            for j in range(i + 1, num_frames)
        }
        for future in as_completed(future_to_pair):
            i, j, d = future.result()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix