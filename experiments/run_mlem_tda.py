"""TDA on MLEM-reconstructed PET volumes.

Phantom -> forward-project -> sample events -> MLEM reconstruct ->
cubical persistence on the reconstructed volume -> inter-frame distance
matrix. Supports three summary modes:

  - 'wasserstein'  : Wasserstein H1 distance between raw diagrams (legacy).
  - 'betti'        : L2 distance between Betti-curve PCFs (masspcf).
  - 'stable_rank'  : L2 distance between stable-rank PCFs (masspcf).
"""

import os
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.utils.device import get_device
from src.phantom.generator import load_xcat
from src.simulation.scanner import get_mct_projector
from src.simulation.listmode import sample_events, indices_to_sinogram
from src.representation.mlem import reconstruct_mlem
from src.tda.persistence import (
    compute_persistence_volume,
    compute_frame_pcfs_volume,
)
from src.tda.distances import (
    compute_distance_matrix,
    compute_pcf_distance_matrix,
)
from src.utils.visualization import (
    plot_phantom_frame,
    plot_distance_matrix,
)


XCAT_PATH = '../data/respiratory_only.npy'


FIG_STYLE = {
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
}


def _save_or_show(fig, save_path=None):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _reconstruct_frame(phantom_frame, proj, num_events, num_iterations):
    indices = sample_events(phantom_frame, proj, num_events=num_events)
    sinogram = indices_to_sinogram(indices, proj)
    return reconstruct_mlem(
        sinogram, proj, num_iterations=num_iterations, verbose=False
    )


def _frame_distance_matrix(volumes, summary, max_dim, persistence_kwargs):
    """Build the inter-frame distance matrix for the chosen summary."""
    if summary == 'wasserstein':
        print("Computing per-frame persistence diagrams ...")
        diagrams = [
            compute_persistence_volume(
                vol, max_dim=max_dim, **(persistence_kwargs or {})
            )
            for vol in volumes
        ]
        print("Computing Wasserstein H1 distance matrix ...")
        return np.asarray(
            compute_distance_matrix(diagrams, method='wasserstein', hom_dim=1)
        )

    print(f"Computing per-frame {summary} PCFs ...")
    pcf_tensors = compute_frame_pcfs_volume(
        volumes,
        summary=summary,
        max_dim=max_dim,
        persistence_kwargs=persistence_kwargs,
    )
    print("Computing L2 distance matrix ...")
    return np.asarray(compute_pcf_distance_matrix(pcf_tensors))


def run_mlem_tda(
    num_frames: int = 20,
    num_events_per_frame: int = 35000,
    num_iterations: int = 4,
    summary: str = 'stable_rank',
    max_dim: int = 1,
    persistence_kwargs: dict | None = None,
    xcat_path: str = XCAT_PATH,
    save_dir: str | None = None,
) -> None:
    print("=" * 60)
    print(f"Running: MLEM TDA (summary={summary!r})")
    print("=" * 60)

    device = get_device()

    print(f"Loading XCAT phantom from {xcat_path}")
    full = load_xcat(xcat_path, device=device)
    phantom = full[:num_frames, 300:300 + 109, :, :]
    print(f"Phantom shape: {phantom.shape}")
    plot_phantom_frame(phantom, frame=0)

    proj = get_mct_projector(
        device=device, img_shape=tuple(phantom.shape[1:]), tof=True
    )

    reconstructions = []
    for frame_idx in range(num_frames):
        print(f"MLEM frame {frame_idx + 1}/{num_frames}")
        reconstructions.append(
            _reconstruct_frame(
                phantom[frame_idx], proj, num_events_per_frame, num_iterations
            )
        )

    dist_matrix = _frame_distance_matrix(
        reconstructions, summary, max_dim, persistence_kwargs
    )
    print(
        f"Distance matrix shape: {dist_matrix.shape}  "
        f"range: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]"
    )

    plot_distance_matrix(
        dist_matrix,
        title=f"Inter-frame distance matrix (MLEM, {summary})",
        labels=list(range(num_frames)),
        cbar_label="distance",
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"distance_matrix_{summary}.png"),
            dpi=300, bbox_inches="tight",
        )

    # Cyclic Spearman correlation against ground-truth phase distance.
    n = dist_matrix.shape[0]
    phase_dist = np.array([
        [min(abs(i - j), n - abs(i - j)) for j in range(n)]
        for i in range(n)
    ], dtype=float)
    upper = np.triu_indices(n, k=1)
    rho, pval = spearmanr(dist_matrix[upper], phase_dist[upper])
    print(f"Cyclic Spearman ρ = {rho:.4f}  (p = {pval:.2e})")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(phase_dist[upper], dist_matrix[upper], s=6, alpha=0.6)
    ax.set_xlabel("Ground-truth phase distance (frames)")
    ax.set_ylabel("Topological distance")
    ax.set_title(f"Cyclic correlation  ρ = {rho:.3f}  ({summary})")
    plt.tight_layout()
    save_path = (
        os.path.join(save_dir, f"cyclic_correlation_{summary}.png")
        if save_dir else None
    )
    _save_or_show(fig, save_path)

    print("Done!")


def intra_variability_mlem_tda(
    num_frames: int = 10,
    num_events_per_frame: int = 35000,
    num_iterations: int = 4,
    num_samples: int = 5,
    ref_frame: int = 0,
    summary: str = 'stable_rank',
    max_dim: int = 1,
    persistence_kwargs: dict | None = None,
    xcat_path: str = XCAT_PATH,
    save_dir: str | None = None,
) -> None:
    print("=" * 60)
    print(f"Running: Intra-Frame Variability MLEM TDA (summary={summary!r})")
    print("=" * 60)

    if summary == 'wasserstein':
        raise ValueError(
            "intra_variability_mlem_tda requires a PCF summary "
            "('betti' or 'stable_rank'); 'wasserstein' has no PCF distance."
        )

    device = get_device()

    print(f"Loading XCAT phantom from {xcat_path}")
    full = load_xcat(xcat_path, device=device)
    phantom = full[:num_frames, 300:300 + 109, :, :]
    print(f"Phantom shape: {phantom.shape}")

    proj = get_mct_projector(
        device=device, img_shape=tuple(phantom.shape[1:]), tof=True
    )

    total_jobs = num_frames * num_samples
    print(
        f"Reconstructing {total_jobs} volumes "
        f"({num_samples} samples × {num_frames} frames) ..."
    )

    # Flat layout: [f0s0, f0s1, ..., f(F-1)s(S-1)].
    all_volumes = []
    for f in range(num_frames):
        for s in range(num_samples):
            print(f"  frame {f + 1}/{num_frames}, sample {s + 1}/{num_samples}")
            all_volumes.append(
                _reconstruct_frame(
                    phantom[f], proj, num_events_per_frame, num_iterations
                )
            )

    print(f"Computing {total_jobs} {summary} PCFs ...")
    pcf_tensors = compute_frame_pcfs_volume(
        all_volumes,
        summary=summary,
        max_dim=max_dim,
        persistence_kwargs=persistence_kwargs,
    )

    print("Computing all-pairs L2 distance matrix ...")
    full_dist = np.asarray(compute_pcf_distance_matrix(pcf_tensors))
    print(f"Distance matrix shape: {full_dist.shape}")

    def flat_idx(f, s):
        return f * num_samples + s

    intra_results: dict[int, list[float]] = {f: [] for f in range(num_frames)}
    for f in range(num_frames):
        for s1 in range(num_samples):
            for s2 in range(s1 + 1, num_samples):
                intra_results[f].append(
                    float(full_dist[flat_idx(f, s1), flat_idx(f, s2)])
                )

    inter_results: dict[int, list[float]] = {f: [] for f in range(num_frames)}
    for f in range(num_frames):
        for s0 in range(num_samples):
            for sf in range(num_samples):
                if f == ref_frame and s0 == sf:
                    continue
                inter_results[f].append(
                    float(full_dist[flat_idx(ref_frame, s0), flat_idx(f, sf)])
                )

    intra_frame_means = np.array([
        np.mean(intra_results[f]) if intra_results[f] else 0.0
        for f in range(num_frames)
    ])
    inter_frame_means = np.array(
        [np.mean(inter_results[f]) for f in range(num_frames)]
    )

    print("Distances computed. Visualizing results...")
    frame_indices = np.arange(num_frames)

    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.errorbar(
            frame_indices, inter_frame_means,
            yerr=intra_frame_means,
            fmt='s', markersize=6, capsize=4, capthick=1.2,
            color='#1f77b4', ecolor='#d62728', elinewidth=1.2,
            label=r'Mean $\pm$ intra-frame variability',
        )

        ax.set_xlabel('Frame index')
        ax.set_ylabel(f'{summary} L2 distance')
        ax.set_title(
            f'{summary} L2 from frame {ref_frame} – MLEM TDA'
        )
        ax.set_xticks(frame_indices)
        ax.legend(frameon=True, edgecolor='black', fancybox=False, loc='upper left')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        fig.tight_layout()
        _save_or_show(
            fig,
            os.path.join(save_dir, f"intra_mlem_tda_{summary}.png")
            if save_dir else None,
        )

    print("Done!")


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_mlem_tda(
        num_frames=20,
        summary='stable_rank',
        persistence_kwargs={
            'filtration': 'superlevel',
            'smooth_sigma': 1.0,
            'normalize': True,
        },
        save_dir=f"figures/run_mlem_tda_{timestamp}",
    )
    # intra_variability_mlem_tda(
    #     num_frames=10,
    #     num_samples=5,
    #     ref_frame=0,
    #     summary='stable_rank',
    #     persistence_kwargs={
    #         'filtration': 'superlevel',
    #         'smooth_sigma': 1.0,
    #         'normalize': True,
    #     },
    #     save_dir=f"figures/intra_variability_mlem_tda_{timestamp}",
    # )
