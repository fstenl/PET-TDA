"""
Point-cloud subsampling strategies.

Every function takes and returns a (N, D) float64 numpy array.  When the
requested subsample is at least as large as the input, the input is returned
unchanged.  The module-level ``subsample()`` dispatches by method name and is
the function the rest of the pipeline should call.
"""

from __future__ import annotations

import numpy as np

from gudhi.subsampling import choose_n_farthest_points


def _as_float64(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)


def random_uniform(
    points,
    n: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Pick ``n`` points uniformly at random without replacement."""
    points = _as_float64(points)
    if n >= len(points):
        return points
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


def farthest_point(
    points,
    n: int,
    *,
    seed_idx: int | None = None,
) -> np.ndarray:
    """Greedy farthest-point (maxmin) sampling via GUDHI."""
    points = _as_float64(points)
    if n >= len(points):
        return points
    kwargs = {"points": points, "nb_points": n}
    if seed_idx is not None:
        kwargs["starting_point"] = int(seed_idx)
    return np.asarray(choose_n_farthest_points(**kwargs), dtype=np.float64)


def voxel_grid(points, voxel_size: float) -> np.ndarray:
    """Snap points to a grid of cell size ``voxel_size`` and keep one per cell.

    The representative of each non-empty cell is the centroid of the points
    that fell into it, which is a touch more stable than picking the first.
    """
    points = _as_float64(points)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_cells = inv.max() + 1
    sums = np.zeros((n_cells, points.shape[1]), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    np.add.at(sums, inv, points)
    np.add.at(counts, inv, 1)
    return sums / counts[:, None]


def poisson_disk(
    points,
    min_distance: float,
    *,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Greedy Poisson-disk thinning: keep points at least ``min_distance`` apart.

    Shuffles the input, walks through it once, accepting a point whenever no
    previously accepted point lies within ``min_distance``.  Uses a cKDTree
    over the accepted set for the neighbour query.
    """
    from scipy.spatial import cKDTree

    points = _as_float64(points)
    if min_distance <= 0:
        raise ValueError("min_distance must be positive")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    order = rng.permutation(len(points))
    accepted: list[np.ndarray] = []
    tree: cKDTree | None = None
    for i in order:
        p = points[i]
        if tree is None or not tree.query_ball_point(p, r=min_distance):
            accepted.append(p)
            tree = cKDTree(np.asarray(accepted))
    return np.asarray(accepted, dtype=np.float64)


_DISPATCH = {
    "random": random_uniform,
    "random_uniform": random_uniform,
    "farthest": farthest_point,
    "farthest_point": farthest_point,
    "maxmin": farthest_point,  # backwards-compat alias
    "voxel": voxel_grid,
    "voxel_grid": voxel_grid,
    "poisson": poisson_disk,
    "poisson_disk": poisson_disk,
}


def subsample(points, method: str, **kwargs) -> np.ndarray:
    """Dispatch to a named subsampling strategy.

    Known methods: ``random`` / ``random_uniform``, ``farthest`` /
    ``farthest_point`` / ``maxmin``, ``voxel`` / ``voxel_grid``,
    ``poisson`` / ``poisson_disk``.
    """
    try:
        fn = _DISPATCH[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown subsampling method {method!r}; "
            f"expected one of {sorted(set(_DISPATCH))}"
        ) from exc
    return fn(points, **kwargs)
