import numpy as np
import torch
import gudhi
from ripser import ripser

from src.tda import backends


_BACKENDS = {
    'witness': backends.witness,
    'ripser': backends.ripser_vr,
    'masspcf': backends.masspcf_vr,
}


def compute_persistence_pointcloud(points, max_dim=1):
    """Compute persistence diagrams from a point cloud using Vietoris-Rips.

    Args:
        points (torch.Tensor or np.ndarray): Point coordinates of shape (N, D).
        max_dim (int): Maximum homology dimension to compute.

    Returns:
        list[np.ndarray]: Persistence diagrams, one array per dimension.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    result = ripser(points, maxdim=max_dim)
    return result['dgms']


def compute_persistence_volume(volume, max_dim=2):
    """Compute persistence diagrams from an image volume using cubical homology.

    Sublevel-set filtration is applied, meaning topological features are tracked
    as the intensity threshold increases from min to max voxel value. Infinite
    death times are removed before returning.

    Args:
        volume (torch.Tensor or np.ndarray): Image volume of shape (N0, N1, ...).
        max_dim (int): Maximum homology dimension to compute.

    Returns:
        list[np.ndarray]: Persistence diagrams, one array per dimension.
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    cubical = gudhi.CubicalComplex(top_dimensional_cells=volume)
    cubical.compute_persistence()

    diagrams = []
    for dim in range(max_dim + 1):
        pairs = cubical.persistence_intervals_in_dimension(dim)
        if len(pairs) > 0:
            pairs = np.array(pairs)
            pairs = pairs[np.isfinite(pairs).all(axis=1)]
            diagrams.append(pairs)
        else:
            diagrams.append(np.empty((0, 2)))

    return diagrams


def compute_persistence_plucker(p1, p2, alpha=1.0, beta=1.0, max_dim=1, n_samples=None):
    """Compute persistence diagrams from LOR endpoints via Plücker distance matrix.

    Converts endpoints to canonical Plücker coordinates, builds the hybrid
    angular+geometric distance matrix, then runs Vietoris-Rips persistence via
    ripser using that precomputed matrix.

    Args:
        p1 (torch.Tensor): Start detector coordinates of shape (N, 3).
        p2 (torch.Tensor): End detector coordinates of shape (N, 3).
        alpha (float): Angular distance weight.
        beta (float): Geometric distance weight (use 1/scanner_radius).
        max_dim (int): Maximum homology dimension to compute.
        n_samples (int | None): If given, randomly subsample to this many LORs
            before building the distance matrix. Required when N is large
            (N=25 000 → ~5 GB matrix; n_samples=2000 → ~32 MB).

    Returns:
        list[np.ndarray]: Persistence diagrams, one (n_k, 2) array per dimension.
    """
    from src.representation.plucker import plucker_distance_matrix

    if n_samples is not None and p1.shape[0] > n_samples:
        idx = torch.randperm(p1.shape[0], device=p1.device)[:n_samples]
        p1, p2 = p1[idx], p2[idx]

    dist = plucker_distance_matrix(p1, p2, alpha=alpha, beta=beta)
    dist_np = dist.detach().cpu().numpy().astype(np.float64)
    result = ripser(dist_np, distance_matrix=True, maxdim=max_dim)
    return result['dgms']


def compute_frame_diagrams_plucker(
    event_frames, proj, alpha=1.0, beta=1.0, max_dim=1, n_samples=None, n_workers=None
):
    """Compute Plücker persistence diagrams for every event frame.

    Vectorises in two ways:
    1. All frames' Plücker distance matrices are built in a single batched
       torch.bmm call (one GPU kernel per operation instead of F serial calls).
    2. The per-frame ripser calls are dispatched to a ThreadPoolExecutor; ripser
       releases the GIL so threads run in true parallel on CPU.

    Args:
        event_frames (list[torch.Tensor]): Per-frame flat LOR indices.
        proj (RegularPolygonPETProjector): Configured parallelproj projector.
        alpha (float): Angular distance weight.
        beta (float): Geometric distance weight (use 1/scanner_radius).
        max_dim (int): Maximum homology dimension to compute.
        n_samples (int | None): LORs to keep per frame before TDA. Strongly
            recommended (e.g. 2000) — N=25 000 produces a ~5 GB distance matrix.
        n_workers (int | None): Thread-pool size for parallel ripser. Defaults
            to number of frames (one thread per frame).

    Returns:
        list[list[np.ndarray]]: One diagram list per frame.
    """
    import os
    import concurrent.futures
    from src.simulation.listmode import get_lor_endpoints
    from src.representation.plucker import (
        to_canonical_plucker_batched,
        compute_hybrid_weighted_distance_batched,
    )

    n_frames = len(event_frames)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, n_frames)

    # --- Step 1: collect & optionally subsample endpoints ---
    print("[plucker] collecting LOR endpoints ...")
    p1_list, p2_list = [], []
    for f_idx, frame in enumerate(event_frames):
        p1, p2, _ = get_lor_endpoints(frame, proj)
        if n_samples is not None and p1.shape[0] > n_samples:
            idx = torch.randperm(p1.shape[0], device=p1.device)[:n_samples]
            p1, p2 = p1[idx], p2[idx]
        p1_list.append(p1)
        p2_list.append(p2)

    # --- Step 2: batched Plücker distance matrices (single bmm pass) ---
    print("[plucker] computing batched distance matrices ...")
    p1_batch = torch.stack(p1_list)   # (F, N, 3)
    p2_batch = torch.stack(p2_list)   # (F, N, 3)
    coords_batch = to_canonical_plucker_batched(p1_batch, p2_batch)
    dist_batch = compute_hybrid_weighted_distance_batched(coords_batch, alpha=alpha, beta=beta)
    dist_np_list = [
        dist_batch[f].detach().cpu().numpy().astype(np.float64) for f in range(n_frames)
    ]
    del dist_batch, coords_batch, p1_batch, p2_batch  # free memory before ripser

    # --- Step 3: parallel ripser across frames ---
    print(f"[plucker] running ripser on {n_frames} frames ({n_workers} workers) ...")

    def _ripser(args):
        f_idx, dist_np = args
        result = ripser(dist_np, distance_matrix=True, maxdim=max_dim)
        print(f"[plucker] frame {f_idx + 1}/{n_frames} done")
        return f_idx, result['dgms']

    all_diagrams = [None] * n_frames
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_ripser, (f, d)): f for f, d in enumerate(dist_np_list)
        }
        for future in concurrent.futures.as_completed(futures):
            f_idx, dgms = future.result()
            all_diagrams[f_idx] = dgms

    return all_diagrams


def compute_persistence(points, subsample=None, subsample_kwargs=None,
                        method='witness', method_kwargs=None):
    """Run the point-cloud persistence pipeline: subsample, then backend.

    Args:
        points (torch.Tensor or np.ndarray): Point coordinates of shape (N, D).
        subsample (str | None): Subsampling strategy name (see
            src.representation.subsampling.subsample). None skips subsampling.
        subsample_kwargs (dict | None): Extra kwargs for the subsampler.
        method (str): TDA backend name; one of 'witness', 'ripser', 'masspcf'.
        method_kwargs (dict | None): Extra kwargs for the backend.

    Returns:
        list[np.ndarray]: Persistence diagrams, one (n_k, 2) array per dimension.
    """
    from src.representation import subsampling as subsampling_module

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float64)

    if subsample is not None:
        points = subsampling_module.subsample(
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


def _frames_to_pclouds(event_frames, proj):
    """Resolve each event-index frame to a (N, 3) numpy point cloud.

    Uses the existing LOR-endpoint and TOF-localisation helpers, so the
    signed-TOF-bin convention is preserved throughout.

    Args:
        event_frames (list[torch.Tensor]): Per-frame flat LOR indices.
        proj (RegularPolygonPETProjector): Configured parallelproj projector.

    Returns:
        list[np.ndarray]: One (N_f, 3) float64 cloud per frame.
    """
    from src.simulation.listmode import get_lor_endpoints
    from src.representation.pointcloud import localize_events

    clouds = []
    for f_idx, frame in enumerate(event_frames):
        print(f"[gating] preparing frame {f_idx + 1}/{len(event_frames)}")
        p1, p2, tof_bins = get_lor_endpoints(frame, proj)
        if tof_bins is None:
            coords = 0.5 * (p1 + p2)
        else:
            coords = localize_events(p1, p2, tof_bins, proj)
        clouds.append(coords.detach().cpu().numpy().astype(np.float64))
    return clouds


def _masspcf_betti_curves(clouds, max_dim):
    """Fast path: one batched masspcf call for all frames at once."""
    import masspcf as mpcf
    from masspcf import persistence as mpers

    n_frames = len(clouds)
    pclouds = mpcf.zeros((n_frames,), dtype=mpcf.pcloud64)
    for f_idx, cloud in enumerate(clouds):
        pclouds[f_idx] = cloud

    print('[gating/masspcf] computing persistent homology (all frames) ...')
    bcs = mpers.compute_persistent_homology(pclouds, max_dim=max_dim)
    betti = mpers.barcode_to_betti_curve(bcs)
    return [betti[:, d] for d in range(max_dim + 1)]


def _per_frame_betti_curves(clouds, method, max_dim, method_kwargs):
    """Generic path: run compute_persistence per frame and assemble PcfTensors."""
    import masspcf as mpcf
    from masspcf.persistence import Barcode, barcode_to_betti_curve

    n_frames = len(clouds)
    method_kwargs = dict(method_kwargs or {})
    method_kwargs.setdefault('max_dim', max_dim)

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


def compute_frame_pcfs(event_frames, proj, method='masspcf', max_dim=1,
                       method_kwargs=None, subsample=None, subsample_kwargs=None):
    """Compute Betti-curve PCFs (one per homology dimension) across event frames.

    Args:
        event_frames (list[torch.Tensor]): Per-frame flat LOR indices.
        proj (RegularPolygonPETProjector): Configured parallelproj projector.
        method (str): TDA backend name; one of 'masspcf', 'witness', 'ripser'.
            'masspcf' runs a single batched persistence call over all frames.
        max_dim (int): Maximum homology dimension to compute.
        method_kwargs (dict | None): Extra kwargs for the backend.
        subsample (str | None): Optional spatial subsampling strategy applied
            per frame before hitting the backend.
        subsample_kwargs (dict | None): Extra kwargs for the subsampler.

    Returns:
        list: One masspcf PcfTensor per homology dimension, each indexed by
            frame. Pass directly to compute_pcf_distance_matrix.
    """
    from src.representation import subsampling as subsampling_module

    clouds = _frames_to_pclouds(event_frames, proj)

    if subsample is not None:
        print(f"[gating] subsampling clouds via {subsample!r}")
        clouds = [
            subsampling_module.subsample(cloud, method=subsample, **(subsample_kwargs or {}))
            for cloud in clouds
        ]

    if method == 'masspcf':
        return _masspcf_betti_curves(clouds, max_dim=max_dim)

    return _per_frame_betti_curves(
        clouds,
        method=method,
        max_dim=max_dim,
        method_kwargs=method_kwargs or {},
    )
