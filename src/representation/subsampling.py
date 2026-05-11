import numpy as np


def _as_float64(points):
    return np.asarray(points, dtype=np.float64)


def random_uniform(points, n, rng=None):
    """Pick n points uniformly at random without replacement.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        n (int): Target number of points to keep.
        rng (np.random.Generator | int | None): Random generator or seed.

    Returns:
        np.ndarray: Subsampled points of shape (min(n, N), D).
    """
    points = _as_float64(points)
    if n >= len(points):
        return points
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


def farthest_point(points, n, seed_idx=None):
    """Greedy farthest-point (maxmin) subsampling via GUDHI.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        n (int): Target number of landmarks to select.
        seed_idx (int | None): Index of the starting point. If None, GUDHI picks.

    Returns:
        np.ndarray: Selected landmarks of shape (<=n, D).
    """
    from gudhi.subsampling import choose_n_farthest_points

    points = _as_float64(points)
    if n >= len(points):
        return points

    kwargs = {'points': points, 'nb_points': n}
    if seed_idx is not None:
        kwargs['starting_point'] = int(seed_idx)
    return np.asarray(choose_n_farthest_points(**kwargs), dtype=np.float64)


def voxel_grid(points, voxel_size):
    """Snap points onto a regular grid and keep one representative per cell.

    The representative is the centroid of the points that fell into the cell.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        voxel_size (float): Grid cell size in the same units as points.

    Returns:
        np.ndarray: One centroid per occupied cell, shape (M, D) with M <= N.
    """
    points = _as_float64(points)
    if voxel_size <= 0:
        raise ValueError('voxel_size must be positive')

    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_cells = inv.max() + 1

    sums = np.zeros((n_cells, points.shape[1]), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    np.add.at(sums, inv, points)
    np.add.at(counts, inv, 1)
    return sums / counts[:, None]


def poisson_disk(points, min_distance, rng=None):
    """Greedy Poisson-disk thinning: keep points at least min_distance apart.

    Shuffles the input, then walks through it once and accepts a point only if
    no previously accepted point lies within min_distance. A cKDTree over the
    accepted set speeds up neighbour queries.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        min_distance (float): Minimum allowed separation between kept points.
        rng (np.random.Generator | int | None): Random generator or seed.

    Returns:
        np.ndarray: Thinned point set of shape (M, D) with M <= N.
    """
    from scipy.spatial import cKDTree

    points = _as_float64(points)
    if min_distance <= 0:
        raise ValueError('min_distance must be positive')
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    order = rng.permutation(len(points))
    accepted = []
    tree = None
    for i in order:
        p = points[i]
        if tree is None or not tree.query_ball_point(p, r=min_distance):
            accepted.append(p)
            tree = cKDTree(np.asarray(accepted))
    return np.asarray(accepted, dtype=np.float64)


_DISPATCH = {
    'random': random_uniform,
    'random_uniform': random_uniform,
    'farthest': farthest_point,
    'farthest_point': farthest_point,
    'maxmin': farthest_point,
    'voxel': voxel_grid,
    'voxel_grid': voxel_grid,
    'poisson': poisson_disk,
    'poisson_disk': poisson_disk,
}


def subsample(points, method, **kwargs):
    """Dispatch to a named subsampling strategy.

    Known methods: 'random' / 'random_uniform', 'farthest' / 'farthest_point' /
    'maxmin', 'voxel' / 'voxel_grid', 'poisson' / 'poisson_disk'.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        method (str): Name of the subsampling strategy.
        **kwargs: Extra keyword arguments forwarded to the chosen strategy.

    Returns:
        np.ndarray: Subsampled points of shape (M, D).
    """
    try:
        fn = _DISPATCH[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown subsampling method {method!r}; "
            f"expected one of {sorted(set(_DISPATCH))}"
        ) from exc
    return fn(points, **kwargs)
