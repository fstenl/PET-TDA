import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering


def cluster_distance_matrix(dist_matrix, num_clusters):
    """Cluster frames using spectral clustering on a distance matrix.

    Args:
        dist_matrix (np.ndarray): Pairwise distance matrix of shape (N, N).
        num_clusters (int): Number of clusters to find.

    Returns:
        np.ndarray: Cluster labels of shape (N,).
    """
    similarity = np.exp(-dist_matrix)
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        random_state=0,
    )
    return clustering.fit_predict(similarity)


def match_labels(labels, ground_truth, num_clusters):
    """Find the optimal mapping between cluster labels and ground truth labels.

    Uses the Hungarian algorithm to find the permutation of cluster labels
    that maximizes overlap with ground truth.

    Args:
        labels (np.ndarray): Cluster labels of shape (N,).
        ground_truth (np.ndarray): Ground truth labels of shape (N,).
        num_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Remapped cluster labels of shape (N,).
    """
    cost_matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(num_clusters):
            cost_matrix[i, j] = -np.sum((labels == i) & (ground_truth == j))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = np.zeros(num_clusters, dtype=int)
    mapping[row_ind] = col_ind

    return mapping[labels]


def dice_score(labels, ground_truth, num_clusters):
    """Compute the mean Dice score between cluster labels and ground truth.

    Args:
        labels (np.ndarray): Cluster labels of shape (N,).
        ground_truth (np.ndarray): Ground truth labels of shape (N,).
        num_clusters (int): Number of clusters.

    Returns:
        float: Mean Dice score across all clusters, between 0 and 1.
    """
    matched = match_labels(labels, ground_truth, num_clusters)

    scores = []
    for k in range(num_clusters):
        a = matched == k
        b = ground_truth == k
        intersection = np.sum(a & b)
        score = 2 * intersection / (np.sum(a) + np.sum(b))
        scores.append(score)

    return float(np.mean(scores))