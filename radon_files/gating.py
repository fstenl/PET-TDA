"""
Data-driven respiratory phase assignment for PET list-mode data.

Pipeline:
  1. window_events          — split event stream into short temporal windows
  2. compute_window_pcfs    — masspcf ripser (VR) on full point cloud → Betti curve PCFs
  3. compute_distance_matrix — all-pairs L2 distance between Betti curve PCFs (masspcf.pdist)

Requires: masspcf, numpy, parallelproj.
"""

import numpy as np
import masspcf as mpcf
from masspcf import pdist
from masspcf import persistence as mpers
from radon_files.scanner import get_event_coords


# ---------------------------------------------------------------------------
# Temporal windowing
# ---------------------------------------------------------------------------

def window_events(
    all_events: np.ndarray,
    window_size: int,
) -> list[np.ndarray]:
    """Split a 1-D event index array into consecutive equal-size chunks.

    In real list-mode data events are ordered by acquisition time, so
    consecutive chunks correspond to short temporal windows.  In simulation
    (where events are drawn per respiratory frame and concatenated) each chunk
    corresponds to one simulated respiratory state.

    Args:
        all_events: 1-D array of list-mode event indices.
        window_size: Number of events per window.

    Returns:
        List of N_w arrays each of shape (window_size,).
        The trailing events are dropped if ``len(all_events) % window_size != 0``.
    """
    n_windows = len(all_events) // window_size
    return [all_events[i * window_size : (i + 1) * window_size] for i in range(n_windows)]


# ---------------------------------------------------------------------------
# Per-window topological feature extraction
# ---------------------------------------------------------------------------

def compute_window_pcfs(
    event_windows: list[np.ndarray],
    proj,
    max_dim: int = 1,
) -> list:
    """Compute Betti curve PCFs for each time window via masspcf VR persistence.

    For each window: ``get_event_coords`` → full (N, 3) point cloud in mm.

    Then in a single batched call:
      ``mpers.compute_persistent_homology`` → Vietoris-Rips (ripser) on all windows
      ``mpers.barcode_to_betti_curve``       → PcfTensor(N_w, max_dim+1)

    Args:
        event_windows: Output of ``window_events()``.
        proj:          parallelproj projector / scanner object.
        max_dim:       Maximum homology dimension (1 → H₀ + H₁).

    Returns:
        List of length (max_dim + 1).  Element ``d`` is a masspcf PcfTensor of
        shape (N_w,) containing the H_d Betti curve for every window.
    """
    n_windows = len(event_windows)

    pclouds = mpcf.zeros((n_windows,), dtype=mpcf.pcloud64)
    for w_idx, window in enumerate(event_windows):
        print(f"[gating] preparing window {w_idx + 1}/{n_windows}")
        coords = get_event_coords(window, proj)
        pclouds[w_idx] = coords.detach().cpu().numpy().astype(np.float64)

    print("[gating] computing persistent homology (all windows) ...")
    bcs = mpers.compute_persistent_homology(pclouds, max_dim=max_dim)
    betti = mpers.barcode_to_betti_curve(bcs)

    return [betti[:, d] for d in range(max_dim + 1)]


# inter-window distance matrix

def compute_distance_matrix(pcf_tensors: list) -> np.ndarray:
    """Compute an all-pairs L2 distance matrix between windows.

    Uses ``masspcf.pdist`` (exact L2 distance between piecewise-constant
    functions, computed in C++/CUDA) for each homology dimension, then sums
    the results so that both H₀ (connectivity) and H₁ (loops) contribute.

    All N_w*(N_w-1)/2 pairs are computed in a single
    vectorised call per dimension.

    Args:
        pcf_tensors: Output of ``compute_window_pcfs()`` — list of PcfTensor(N_w,).

    Returns:
        Symmetric (N_w × N_w) float64 distance matrix.
    """
    D = None
    for pcf_tensor in pcf_tensors:
        d = pdist(pcf_tensor, p=2).to_dense()  # (N_w, N_w) numpy array
        D = d if D is None else D + d
    return D


