import numpy as np

from src.representation.subsampling import subsample


def _apply_min_persistence(diagrams, min_persistence):
    """Drop features with lifetime below min_persistence.

    Args:
        diagrams (list[np.ndarray]): Persistence diagrams, one per dimension.
        min_persistence (float): Minimum lifetime to keep.

    Returns:
        list[np.ndarray]: Filtered diagrams.
    """
    if min_persistence <= 0:
        return diagrams
    out = []
    for dgm in diagrams:
        if len(dgm) == 0:
            out.append(dgm)
            continue
        keep = (dgm[:, 1] - dgm[:, 0]) >= min_persistence
        out.append(dgm[keep])
    return out


def _report(name, diagrams):
    """Print a one-line-per-dimension summary of finite/infinite features."""
    for dim, dgm in enumerate(diagrams):
        n_inf = int(np.sum(~np.isfinite(dgm[:, 1]))) if len(dgm) else 0
        n_fin = len(dgm) - n_inf
        print(f"  [{name}] H{dim}: {len(dgm)} features "
              f"({n_fin} finite, {n_inf} infinite)")


def witness(points, n_landmarks=None, landmark_ratio=0.10,
            landmark_method='farthest', landmark_kwargs=None,
            max_alpha_square=float('inf'), max_dim=1, min_persistence=0.0):
    """GUDHI Euclidean strong witness complex persistence.

    The full point cloud acts as the witness set. Landmarks are drawn from the
    cloud via the subsampling module and become the vertices of the simplicial
    complex. GUDHI filtration values are squared distances, so sqrt is applied
    before returning, giving diagrams in the same units as the input.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        n_landmarks (int | None): Number of landmarks. If None, uses
            max(int(N * landmark_ratio), max_dim + 2).
        landmark_ratio (float): Fraction of points used as landmarks when
            n_landmarks is None.
        landmark_method (str): Subsampling strategy name (see
            src.representation.subsampling.subsample).
        landmark_kwargs (dict | None): Extra kwargs for the subsampler.
        max_alpha_square (float): Maximum squared filtration value.
        max_dim (int): Maximum homology dimension to compute.
        min_persistence (float): Minimum lifetime to keep.

    Returns:
        list[np.ndarray]: Persistence diagrams, one (n_k, 2) array per dimension.
    """
    import gudhi

    points = np.asarray(points, dtype=np.float64)
    n_points = len(points)
    if n_landmarks is None:
        n_landmarks = max(int(n_points * landmark_ratio), max_dim + 2)

    landmarks = subsample(
        points,
        method=landmark_method,
        n=n_landmarks,
        **(landmark_kwargs or {}),
    )

    print(f"[witness] {n_points} points -> {len(landmarks)} landmarks "
          f"({landmark_method}), max_dim={max_dim}")

    wc = gudhi.EuclideanStrongWitnessComplex(
        landmarks=np.asarray(landmarks).tolist(),
        witnesses=points.tolist(),
    )
    simplex_tree = wc.create_simplex_tree(
        max_alpha_square=max_alpha_square,
        limit_dimension=max_dim + 1,
    )
    print(f"[witness] simplex tree: {simplex_tree.num_simplices()} simplices, "
          f"dimension {simplex_tree.dimension()}")

    simplex_tree.persistence(
        homology_coeff_field=2,
        min_persistence=min_persistence,
    )

    diagrams = []
    for dim in range(max_dim + 1):
        pairs = simplex_tree.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            diagrams.append(np.empty((0, 2), dtype=np.float64))
        else:
            pairs = np.asarray(pairs, dtype=np.float64)
            diagrams.append(np.sqrt(np.clip(pairs, 0, None)))

    _report('witness', diagrams)
    return diagrams


def ripser_vr(points, max_dim=1, thresh=float('inf'), min_persistence=0.0):
    """Vietoris-Rips persistence via the ripser package.

    Ripser returns Euclidean-distance filtration values directly, so no
    rescaling is needed.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        max_dim (int): Maximum homology dimension to compute.
        thresh (float): Maximum filtration value; features beyond are ignored.
        min_persistence (float): Minimum lifetime to keep.

    Returns:
        list[np.ndarray]: Persistence diagrams, one (n_k, 2) array per dimension.
    """
    from ripser import ripser as _ripser

    points = np.asarray(points, dtype=np.float64)
    print(f"[ripser] {len(points)} points, max_dim={max_dim}, thresh={thresh}")

    result = _ripser(points, maxdim=max_dim, thresh=thresh)
    diagrams = [np.asarray(d, dtype=np.float64) for d in result['dgms']]
    diagrams = _apply_min_persistence(diagrams, min_persistence)

    _report('ripser', diagrams)
    return diagrams


def masspcf_vr(points, max_dim=1, min_persistence=0.0):
    """Batched Vietoris-Rips persistence via masspcf (single-cloud wrapper).

    Wraps masspcf.persistence.compute_persistent_homology for a single cloud so
    it fits the same interface as witness and ripser_vr. For many clouds at
    once, prefer the batched helpers in src.tda.persistence.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        max_dim (int): Maximum homology dimension to compute.
        min_persistence (float): Minimum lifetime to keep.

    Returns:
        list[np.ndarray]: Persistence diagrams, one (n_k, 2) array per dimension.
    """
    import masspcf as mpcf
    from masspcf import persistence as mpers

    points = np.asarray(points, dtype=np.float64)
    print(f"[masspcf] {len(points)} points, max_dim={max_dim}")

    pclouds = mpcf.zeros((1,), dtype=mpcf.pcloud64)
    pclouds[0] = points
    bcs = mpers.compute_persistent_homology(pclouds, max_dim=max_dim)

    diagrams = []
    for dim in range(max_dim + 1):
        arr = np.asarray(bcs[0, dim], dtype=np.float64)
        if arr.ndim == 1 and arr.size == 0:
            arr = np.empty((0, 2), dtype=np.float64)
        elif arr.ndim != 2 or arr.shape[-1] != 2:
            arr = arr.reshape(-1, 2)
        diagrams.append(arr)

    diagrams = _apply_min_persistence(diagrams, min_persistence)
    _report('masspcf', diagrams)
    return diagrams
