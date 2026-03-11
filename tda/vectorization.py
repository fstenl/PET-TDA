import numpy as np
from persim import PersistenceImager

def get_persistence_images(diagram_list: list, hom_dim: int = 1, resolution: tuple = (20, 20)) -> np.ndarray:
    """Converts a list of persistence diagrams into vector-ready images.

    Args:
        diagram_list (list): List of persistence diagrams for each frame.
        hom_dim (int): Homology dimension to vectorize (0 or 1).
        resolution (tuple): Grid size of the resulting image (rows, cols).

    Returns:
        np.ndarray: A 3D array of shape (num_frames, rows, cols).
    """
    # Extract the requested homology dimension and remove points with infinite death times
    dgms = [d[hom_dim] for d in diagram_list]
    dgms_finite = [d[np.isfinite(d[:, 1])] for d in dgms]

    # Initialize the imager and manually set the pixel resolution
    imager = PersistenceImager()
    imager.n_pixels = resolution

    # Fit the imager to the data to establish global birth and persistence ranges
    imager.fit(dgms_finite)

    # Transform the collection of diagrams into a sequence of smoothed images
    imgs = imager.transform(dgms_finite)

    return np.array(imgs)