"""
TDA pipeline and data-driven respiratory phase assignment for PET data.

This module unifies the extraction of persistent homology from PET TOF 
point clouds and the subsequent respiratory phase assignment (gating).

Three orthogonal layers:
  1. Subsampling & Persistence   — reduce the cloud and compute diagrams.
  2. Vectorisation               — turn diagrams into Betti / Euler curves via masspcf.
  3. Gating                      — compute distance matrices between event frames.

Requires: masspcf, numpy, parallelproj, gudhi, ripser, scipy.
"""

from __future__ import annotations

import numpy as np
import masspcf as mpcf
from masspcf import pdist
from masspcf import persistence as mpers
from masspcf.persistence import Barcode, barcode_to_betti_curve

from scanner import get_event_coords
import subsampling
import tda_backends

# ---------------------------------------------------------------------------
# Core Utilities & TDA Pipeline
# ---------------------------------------------------------------------------

_BACKENDS = {
    "witness": tda_backends.witness,
    "ripser": tda_backends.ripser,
    "masspcf": tda_backends.masspcf,
}

def _to_numpy(arr) -> np.ndarray:
    """Coerce a torch / cupy / numpy array to a float64 numpy array."""
    if hasattr(arr, "detach"):           # torch.Tensor
        arr = arr.detach().cpu().numpy()
    elif hasattr(arr, "get"):            # cupy.ndarray
        arr = arr.get()
    return np.asarray(arr, dtype=np.float64)


def compute_persistence(
    points,
    *,
    subsample: str | None = None,
    subsample_kwargs: dict | None = None,
    method: str = "witness",
    method_kwargs: dict | None = None,
) -> list[np.ndarray]:
    """Run the point-cloud → persistence-diagram pipeline.

    Args:
        points: (N, 3) point cloud — numpy / torch / cupy.
        subsample: Optional subsampling method name (see
            ``radon_files.subsampling.subsample``).  ``None`` skips subsampling.
        subsample_kwargs: Extra kwargs for the subsampler (e.g. ``{"n": 5000}``).
        method: TDA backend name; one of ``"witness"``, ``"ripser"``, ``"masspcf"``.
        method_kwargs: Extra kwargs for the backend.

    Returns:
        ripser-format ``list[np.ndarray]`` of (n_k, 2) birth/death pairs,
        one entry per homology dimension 0..max_dim.
    """
    points = _to_numpy(points)

    if subsample is not None:
        points = subsampling.subsample(
            points,
            method=subsample,
            **(subsample_kwargs or {}),
        )

    try:
        backend = _BACKENDS[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown TDA method {method!r}; expected one of {sorted(_BACKENDS)}"
        ) from exc

    return backend(points, **(method_kwargs or {}))

# ---------------------------------------------------------------------------
# Vectorisation via masspcf
# ---------------------------------------------------------------------------

def _t_max_from_diagrams(diagrams: list[np.ndarray]) -> float:
    all_deaths = np.concatenate(
        [dgm[:, 1][np.isfinite(dgm[:, 1])] for dgm in diagrams if len(dgm)]
    ) if any(len(d) for d in diagrams) else np.empty(0)
    return float(all_deaths.max()) if len(all_deaths) else 1.0


def diagrams_to_betti_curves(
    diagrams: list[np.ndarray],
    t_min: float = 0.0,
    t_max: float | None = None,
    n_steps: int = 200,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Convert persistence diagrams to sampled Betti-number curves via masspcf."""
    if t_max is None:
        t_max = _t_max_from_diagrams(diagrams)

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
    """Euler characteristic curve χ(t) = Σ_k (-1)^k β_k(t) via masspcf."""
    if t_max is None:
        t_max = _t_max_from_diagrams(diagrams)

    t = np.linspace(t_min, t_max, n_steps)
    chi = np.zeros(n_steps, dtype=float)

    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        barcode = Barcode(dgm.astype(np.float64))
        pcf = barcode_to_betti_curve(barcode)
        chi += ((-1) ** dim) * pcf(t)

    return t, chi

# ---------------------------------------------------------------------------
# Phase Assignment / Gating
# ---------------------------------------------------------------------------

def _frames_to_pclouds(event_frames, proj) -> list[np.ndarray]:
    """Resolve each event-index frame to a (N, 3) numpy point cloud."""
    clouds: list[np.ndarray] = []
    for f_idx, frame in enumerate(event_frames):
        print(f"[gating] preparing frame {f_idx + 1}/{len(event_frames)}")
        coords = get_event_coords(frame, proj)
        clouds.append(_to_numpy(coords))
    return clouds


def _masspcf_betti_curves(clouds: list[np.ndarray], max_dim: int) -> list:
    """Fast path: one batched call for all frames (preserves masspcf batching)."""
    n_frames = len(clouds)
    pclouds = mpcf.zeros((n_frames,), dtype=mpcf.pcloud64)
    for f_idx, cloud in enumerate(clouds):
        pclouds[f_idx] = cloud

    print("[gating/masspcf] computing persistent homology (all frames) ...")
    bcs = mpers.compute_persistent_homology(pclouds, max_dim=max_dim)
    betti = mpers.barcode_to_betti_curve(bcs)
    return [betti[:, d] for d in range(max_dim + 1)]


def _per_frame_betti_curves(
    clouds: list[np.ndarray],
    method: str,
    max_dim: int,
    method_kwargs: dict,
) -> list:
    """Generic path: compute_persistence per frame, then build PcfTensors."""
    n_frames = len(clouds)
    method_kwargs = dict(method_kwargs or {})
    method_kwargs.setdefault("max_dim", max_dim)

    # Per-dim PcfTensors to match the batched return shape.
    pcfs_per_dim = [mpcf.zeros((n_frames,)) for _ in range(max_dim + 1)]

    for f_idx, cloud in enumerate(clouds):
        print(f"[gating/{method}] frame {f_idx + 1}/{n_frames}")
        dgms = compute_persistence(cloud, method=method, method_kwargs=method_kwargs)
        for d, dgm in enumerate(dgms):
            if len(dgm) == 0:
                continue
            bc = Barcode(dgm.astype(np.float64))
            pcfs_per_dim[d][f_idx] = barcode_to_betti_curve(bc)

    return pcfs_per_dim


def compute_frame_pcfs(
    event_frames: list[np.ndarray],
    proj,
    *,
    method: str = "masspcf",
    max_dim: int = 1,
    method_kwargs: dict | None = None,
    subsample: str | None = None,         # <-- ADDED
    subsample_kwargs: dict | None = None, # <-- ADDED
) -> list:
    """Compute Betti curve PCFs for each event frame."""
    clouds = _frames_to_pclouds(event_frames, proj)

    # Apply spatial subsampling universally before hitting the TDA backends
    if subsample is not None:
        print(f"[gating] Subsampling clouds using method: {subsample}")
        clouds = [
            subsampling.subsample(cloud, method=subsample, **(subsample_kwargs or {}))
            for cloud in clouds
        ]

    if method == "masspcf":
        return _masspcf_betti_curves(clouds, max_dim=max_dim)
    
    # We remove 'subsample' from method_kwargs if present so compute_persistence 
    # doesn't try to double-subsample it.
    return _per_frame_betti_curves(
        clouds, 
        method=method, 
        max_dim=max_dim, 
        method_kwargs=method_kwargs or {}
    )

def compute_distance_matrix(pcf_tensors: list) -> np.ndarray:
    """All-pairs L2 distance matrix between frames, summed over dimensions.

    Uses ``masspcf.pdist`` (exact L2 distance between piecewise-constant
    functions) for each homology dimension, then sums so that both H₀
    (connectivity) and H₁ (loops) contribute.
    """
    D = None
    for pcf_tensor in pcf_tensors:
        d = pdist(pcf_tensor, p=2).to_dense()
        D = d if D is None else D + d
    return D