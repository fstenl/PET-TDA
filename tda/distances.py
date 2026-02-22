import numpy as np
from persim import bottleneck, wasserstein


def compute_bottleneck_distance(dgm1, dgm2, hom_dim=1):
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


def compute_wasserstein_distance(dgm1, dgm2, hom_dim=1):
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


def compute_trajectory_distances(diagram_list, method='wasserstein', hom_dim=1):
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


def compute_all_pairs_distances(diagram_list, method='wasserstein', hom_dim=1):
    """
    Computes a distance matrix comparing every frame's diagram to every other frame,
    handling infinite death times commonly found in H0.

    Args:
        diagram_list (list): List of diagrams for each frame.
        method (str): 'wasserstein' or 'bottleneck'.
        hom_dim (int): Homology dimension (usually 1 for loops).

    Returns:
        np.ndarray: A square symmetric matrix of distances.
    """
    num_frames = len(diagram_list)
    dist_matrix = np.zeros((num_frames, num_frames))
    dist_func = wasserstein if method == 'wasserstein' else bottleneck

    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            # Extract diagrams for the specific dimension
            dgm1 = diagram_list[i][hom_dim]
            dgm2 = diagram_list[j][hom_dim]

            # Filter out points with infinite death times (np.inf) to avoid warnings
            dgm1_finite = dgm1[np.isfinite(dgm1[:, 1])]
            dgm2_finite = dgm2[np.isfinite(dgm2[:, 1])]

            # Compute distance between frame i and frame j using finite points
            if len(dgm1_finite) > 0 and len(dgm2_finite) > 0:
                d = dist_func(dgm1_finite, dgm2_finite)
            else:
                # Handle cases where diagrams might be empty after filtering
                d = 0.0

            dist_matrix[i, j] = d
            dist_matrix[j, i] = d  # Matrix is symmetric

    return dist_matrix