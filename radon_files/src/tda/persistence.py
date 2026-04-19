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
