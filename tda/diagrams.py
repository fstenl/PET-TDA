import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import torch


def compute_persistence(data, is_distance_matrix=False, max_dim=1):
    """Computes persistence diagrams from point clouds or distance matrices.

    Args:
        data (torch.Tensor or np.ndarray): The input data.
            Either (N, D) coordinates or (N, N) distance matrix.
        is_distance_matrix (bool): Set to True if data is a distance matrix.
        max_dim (int): Maximum homology dimension (0=H0, 1=H1, etc.).

    Returns:
        list: A list of persistence diagrams (one for each dimension).
    """
    # Convert torch tensor to numpy as ripser is a C++ wrapper for numpy
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # distance_matrix=True tells ripser the input is already a distance matrix
    result = ripser(data, distance_matrix=is_distance_matrix, maxdim=max_dim)

    return result['dgms']

def plot_persistence_diagram(dgms, title="Persistence Diagram", save_path=None):
    """Visualizes the H0 and H1 persistence diagrams.

    Args:
        dgms (list): Persistence diagrams from compute_persistence.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the image file.
    """
    plt.figure(figsize=(6, 6))
    plot_diagrams(dgms, show=False)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()