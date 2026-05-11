import numpy as np
from persim import bottleneck, wasserstein


def compute_bottleneck(dgm1, dgm2, hom_dim=1):
    """Compute the bottleneck distance between two persistence diagrams.

    Args:
        dgm1 (list[np.ndarray]): Persistence diagrams from first input.
        dgm2 (list[np.ndarray]): Persistence diagrams from second input.
        hom_dim (int): Homology dimension to compare.

    Returns:
        float: Bottleneck distance.
    """
    d1, d2 = dgm1[hom_dim], dgm2[hom_dim]
    if len(d1) == 0 or len(d2) == 0:
        return 0.0
    return bottleneck(d1, d2)


def compute_wasserstein(dgm1, dgm2, hom_dim=1):
    """Compute the Wasserstein distance between two persistence diagrams.

    Args:
        dgm1 (list[np.ndarray]): Persistence diagrams from first input.
        dgm2 (list[np.ndarray]): Persistence diagrams from second input.
        hom_dim (int): Homology dimension to compare.

    Returns:
        float: Wasserstein distance.
    """
    d1, d2 = dgm1[hom_dim], dgm2[hom_dim]
    if len(d1) == 0 or len(d2) == 0:
        return 0.0
    return wasserstein(d1, d2)


def compute_distance_matrix(diagrams, method='wasserstein', hom_dim=1):
    """Compute a pairwise distance matrix between a list of persistence diagrams.

    Args:
        diagrams (list[list[np.ndarray]]): Persistence diagrams, one per frame.
        method (str): Distance metric, either 'wasserstein' or 'bottleneck'.
        hom_dim (int): Homology dimension to compare.

    Returns:
        np.ndarray: Symmetric distance matrix of shape (num_frames, num_frames).
    """
    dist_func = wasserstein if method == 'wasserstein' else bottleneck
    num_frames = len(diagrams)
    dist_matrix = np.zeros((num_frames, num_frames))

    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            dgm1 = diagrams[i][hom_dim]
            dgm2 = diagrams[j][hom_dim]

            if len(dgm1) > 0 and len(dgm2) > 0:
                d = dist_func(dgm1, dgm2)
            else:
                d = 0.0

            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix