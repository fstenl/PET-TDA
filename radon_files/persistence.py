"""
Witness-complex persistent homology for PET TOF point clouds.

Pipeline:
  1. Take the 3D event coordinates from get_event_coords() (shape [N, 3], in mm).
  2. Select a small set of landmarks using farthest-point sampling (maxmin)
     or k-means++ — these become the vertices of the simplicial complex.
  3. Build a Euclidean strong witness complex (GUDHI) where every remaining
     point acts as a witness certifying that nearby landmarks are connected.
  4. Extract persistence diagrams from the resulting filtered simplicial complex.
  5. (Optional) Convert diagrams to piecewise-constant Betti / Euler curves
     via masspcf for frame-to-frame comparison.

Requires: gudhi, numpy, torch, masspcf.
"""

import numpy as np
import torch

import gudhi
from gudhi.subsampling import pick_n_random_points
from gudhi.subsampling import choose_n_farthest_points
from masspcf.persistence import Barcode, barcode_to_betti_curve


# ---------------------------------------------------------------------------
# Landmark selection
# ---------------------------------------------------------------------------

def select_landmarks(
    points: np.ndarray,
    n_landmarks: int,
    method: str = "maxmin",
) -> np.ndarray:
    """Pick landmark points from a point cloud.

    Args:
        points: (N, D) array of coordinates (mm).
        n_landmarks: How many landmarks to keep.
        method: 'maxmin' (farthest-point / greedy) or 'random'.

    Returns:
        (n_landmarks, D) array of landmark coordinates.
    """
    if n_landmarks >= len(points):
        return points

    if method == "maxmin":
        landmarks = choose_n_farthest_points(
            points=points,
            nb_points=n_landmarks,
        )
    elif method == "random":
        landmarks = pick_n_random_points(
            points=points,
            nb_points=n_landmarks,
        )
    else:
        raise ValueError(f"Unknown landmark method: {method!r}")

    return np.asarray(landmarks)


# ---------------------------------------------------------------------------
# Witness-complex persistence
# ---------------------------------------------------------------------------

def witness_persistence(
    points: np.ndarray | torch.Tensor,
    n_landmarks: int | None = None,
    landmark_ratio: float = 0.10,
    landmark_method: str = "maxmin",
    max_alpha_square: float = float("inf"),
    max_dim: int = 1,
    min_persistence: float = 0.0,
) -> list[np.ndarray]:
    """Compute persistent homology of a point cloud via the witness complex.

    This is the main entry point.  It handles torch→numpy conversion,
    landmark selection, witness-complex construction, and persistence
    computation, returning diagrams in the same format as ripser
    (list of (n_i, 2) arrays, one per homology dimension).

    Args:
        points: (N, 3) point cloud — TOF midpoint coordinates in mm.
        n_landmarks: Absolute number of landmarks.  If None, derived from
            ``landmark_ratio * N``.
        landmark_ratio: Fraction of points to use as landmarks (default 10 %).
        landmark_method: 'maxmin' (farthest-point sampling — better spatial
            coverage, good default) or 'random'.
        max_alpha_square: Upper bound on the squared filtration parameter.
            Points further apart than sqrt(max_alpha_square) are never connected.
            Defaults to infinity (let the filtration run to completion).
        max_dim: Maximum homology dimension (0 → H₀ only, 1 → H₀+H₁, …).
        min_persistence: Discard features whose (death − birth) is below this
            threshold (in mm, since the filtration is in Euclidean distance).

    Returns:
        diagrams: list of length (max_dim + 1).  diagrams[k] is an (n_k, 2)
            numpy array of (birth, death) pairs for H_k.  Matches the format
            returned by ripser and used by persim / masspcf.
    """
    # --- numpy conversion -------------------------------------------------
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float64)

    N = len(points)
    if n_landmarks is None:
        n_landmarks = max(int(N * landmark_ratio), max_dim + 2)

    print(f"[witness_persistence] {N} points → {n_landmarks} landmarks "
          f"({landmark_method}), max_dim={max_dim}")

    # --- landmark selection -----------------------------------------------
    landmarks = select_landmarks(points, n_landmarks, method=landmark_method)

    # --- build witness complex --------------------------------------------
    # GUDHI expects plain Python lists of lists.
    witness_complex = gudhi.EuclideanStrongWitnessComplex(
        landmarks=landmarks.tolist(),
        witnesses=points.tolist(),
    )

    simplex_tree = witness_complex.create_simplex_tree(
        max_alpha_square=max_alpha_square,
        limit_dimension=max_dim + 1,  # need (d+1)-simplices for H_d
    )

    print(f"[witness_persistence] simplex tree: "
          f"{simplex_tree.num_simplices()} simplices, "
          f"dimension {simplex_tree.dimension()}")

    # --- persistence ------------------------------------------------------
    simplex_tree.persistence(
        homology_coeff_field=2,       # Z/2Z coefficients (standard choice)
        min_persistence=min_persistence,
    )

    # --- collect diagrams in ripser-compatible format ----------------------
    diagrams = []
    for dim in range(max_dim + 1):
        pairs = simplex_tree.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            pairs = np.empty((0, 2), dtype=np.float64)
        else:
            pairs = np.asarray(pairs, dtype=np.float64)
            # GUDHI filtration values are squared distances; take sqrt so
            # birth/death are in the same mm units as the input coordinates.
            pairs = np.sqrt(np.clip(pairs, 0, None))
        diagrams.append(pairs)

    for dim, dgm in enumerate(diagrams):
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm
        n_inf = np.sum(~np.isfinite(dgm[:, 1])) if len(dgm) > 0 else 0
        print(f"  H{dim}: {len(dgm)} features "
              f"({len(finite)} finite, {n_inf} infinite)")

    return diagrams


# ---------------------------------------------------------------------------
# Betti / Euler curves via masspcf
# ---------------------------------------------------------------------------

def diagrams_to_betti_curves(
    diagrams: list[np.ndarray],
    t_min: float = 0.0,
    t_max: float | None = None,
    n_steps: int = 200,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Convert persistence diagrams to sampled Betti-number curves via masspcf.

    Uses masspcf.persistence.barcode_to_betti_curve() to build a piecewise-constant
    function for each homology dimension, then samples it on a uniform grid.
    For each dimension k the Betti number β_k(t) counts how many features
    are alive at filtration value t  (birth ≤ t < death).

    Args:
        diagrams: Output of witness_persistence().
        t_min: Start of the filtration range.
        t_max: End of the filtration range (default: max finite death value).
        n_steps: Number of sample points along the filtration axis.

    Returns:
        Dictionary  {dim: (t_values, betti_values)}  where both arrays have
        shape (n_steps,).
    """
    if t_max is None:
        all_deaths = np.concatenate(
            [dgm[:, 1][np.isfinite(dgm[:, 1])] for dgm in diagrams if len(dgm)]
        )
        t_max = float(all_deaths.max()) if len(all_deaths) else 1.0

    t = np.linspace(t_min, t_max, n_steps)
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            curves[dim] = (t, np.zeros(n_steps, dtype=int))
            continue
        barcode = Barcode(dgm.astype(np.float64))
        pcf = barcode_to_betti_curve(barcode)
        curves[dim] = (t, pcf(t).astype(int))

    return curves


def euler_characteristic_curve(
    diagrams: list[np.ndarray],
    t_min: float = 0.0,
    t_max: float | None = None,
    n_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Euler characteristic curve χ(t) = Σ_k (-1)^k β_k(t) via masspcf.

    Builds a masspcf Pcf for each homology dimension using barcode_to_betti_curve(),
    then computes the alternating sum by sampling each Betti Pcf independently.

    Args:
        diagrams: Output of witness_persistence().
        t_min, t_max, n_steps: Filtration sampling parameters.

    Returns:
        (t_values, chi_values) each of shape (n_steps,).
    """
    if t_max is None:
        all_deaths = np.concatenate(
            [dgm[:, 1][np.isfinite(dgm[:, 1])] for dgm in diagrams if len(dgm)]
        )
        t_max = float(all_deaths.max()) if len(all_deaths) else 1.0

    t = np.linspace(t_min, t_max, n_steps)
    chi = np.zeros(n_steps, dtype=float)

    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        barcode = Barcode(dgm.astype(np.float64))
        pcf = barcode_to_betti_curve(barcode)
        chi += ((-1) ** dim) * pcf(t)

    return t, chi
