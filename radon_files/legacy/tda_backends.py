"""
Persistent-homology backends for PET point clouds.

Every backend takes a (N, 3) float64 numpy point cloud (in mm) and returns a
ripser-compatible ``list[np.ndarray]`` of (n_k, 2) birth/death pairs, one entry
per homology dimension 0..max_dim.  This is the contract the masspcf
vectorisation layer in ``persistence.py`` relies on.
"""

from __future__ import annotations

import numpy as np
import gudhi

import subsampling


def _apply_min_persistence(
    diagrams: list[np.ndarray],
    min_persistence: float,
) -> list[np.ndarray]:
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


def _report(name: str, diagrams: list[np.ndarray]) -> None:
    for dim, dgm in enumerate(diagrams):
        n_inf = int(np.sum(~np.isfinite(dgm[:, 1]))) if len(dgm) else 0
        n_fin = len(dgm) - n_inf
        print(f"  [{name}] H{dim}: {len(dgm)} features "
              f"({n_fin} finite, {n_inf} infinite)")


def witness(
    points: np.ndarray,
    *,
    n_landmarks: int | None = None,
    landmark_ratio: float = 0.10,
    landmark_method: str = "farthest",
    landmark_kwargs: dict | None = None,
    max_alpha_square: float = float("inf"),
    max_dim: int = 1,
    min_persistence: float = 0.0,
) -> list[np.ndarray]:
    """GUDHI Euclidean strong witness complex persistence.

    The full point cloud acts as the witness set; ``n_landmarks`` (or
    ``landmark_ratio * N``) points selected via the subsampling module become
    the vertices of the simplicial complex.

    GUDHI filtration values are squared distances, so we return ``sqrt`` of them
    — diagrams are in the same mm units as the input coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    N = len(points)
    if n_landmarks is None:
        n_landmarks = max(int(N * landmark_ratio), max_dim + 2)

    landmarks = subsampling.subsample(
        points,
        method=landmark_method,
        n=n_landmarks,
        **(landmark_kwargs or {}),
    )

    print(f"[witness] {N} points → {len(landmarks)} landmarks "
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

    diagrams: list[np.ndarray] = []
    for dim in range(max_dim + 1):
        pairs = simplex_tree.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            diagrams.append(np.empty((0, 2), dtype=np.float64))
        else:
            pairs = np.asarray(pairs, dtype=np.float64)
            diagrams.append(np.sqrt(np.clip(pairs, 0, None)))

    _report("witness", diagrams)
    return diagrams


def masspcf(
    points: np.ndarray,
    *,
    max_dim: int = 1,
    min_persistence: float = 0.0,
) -> list[np.ndarray]:
    """masspcf batched VR persistence (batch of 1).

    Wraps ``masspcf.persistence.compute_persistent_homology`` so it fits the
    same single-cloud contract as ``witness`` / ``ripser``.  For many clouds at
    once, prefer the batched helpers in ``gating.py`` — they keep the masspcf
    batching advantage.
    """
    import masspcf as mpcf
    from masspcf import persistence as mpers

    points = np.asarray(points, dtype=np.float64)
    print(f"[masspcf] {len(points)} points, max_dim={max_dim}")

    pclouds = mpcf.zeros((1,), dtype=mpcf.pcloud64)
    pclouds[0] = points
    bcs = mpers.compute_persistent_homology(pclouds, max_dim=max_dim)

    diagrams: list[np.ndarray] = []
    for dim in range(max_dim + 1):
        arr = np.asarray(bcs[0, dim], dtype=np.float64)
        if arr.ndim == 1 and arr.size == 0:
            arr = np.empty((0, 2), dtype=np.float64)
        elif arr.ndim != 2 or arr.shape[-1] != 2:
            arr = arr.reshape(-1, 2)
        diagrams.append(arr)

    diagrams = _apply_min_persistence(diagrams, min_persistence)
    _report("masspcf", diagrams)
    return diagrams


def ripser(
    points: np.ndarray,
    *,
    max_dim: int = 1,
    thresh: float = float("inf"),
    min_persistence: float = 0.0,
) -> list[np.ndarray]:
    """Vietoris–Rips persistence via the ``ripser`` package.

    Ripser returns Euclidean-distance filtration values directly, so no
    rescaling is needed — diagrams are already in mm.
    """
    from ripser import ripser as _ripser

    points = np.asarray(points, dtype=np.float64)
    print(f"[ripser] {len(points)} points, max_dim={max_dim}, thresh={thresh}")

    result = _ripser(points, maxdim=max_dim, thresh=thresh)
    diagrams = [np.asarray(d, dtype=np.float64) for d in result["dgms"]]
    diagrams = _apply_min_persistence(diagrams, min_persistence)

    _report("ripser", diagrams)
    return diagrams
