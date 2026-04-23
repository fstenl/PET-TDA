import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src.utils.device import get_device
from src.phantom.generator import load_xcat
from src.simulation.scanner import get_mct_projector
from src.simulation.listmode import (
    sample_events_from_sinogram,
    get_lor_endpoints,
)
from src.representation.pointcloud import localize_events
from src.tda.persistence import compute_frame_pcfs
from src.tda.distances import compute_pcf_distance_matrix
from src.utils.visualization import (
    plot_phantom_frame,
    plot_distance_matrix,
)

FIG_STYLE = {
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
}

XCAT_PATH = '../../../data/respiratory_only.npy'


def _save_or_show(fig, save_path=None):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def run_tof_tda(
    num_frames: int = 20,
    num_events_per_frame: int = 35000,
    gating_method: str = 'masspcf',
    xcat_path: str = XCAT_PATH,
    save_dir: str | None = None,
) -> None:
    print("=" * 60)
    print("Running: TOF TDA (inter-frame Betti-curve distance matrix)")
    print("=" * 60)

    device = get_device()

    print(f"Loading XCAT phantom from {xcat_path}")
    full = load_xcat(xcat_path, device=device)
    phantom = full[:num_frames, 300:300 + 109, :, :]
    proj = get_mct_projector(device=device, img_shape=tuple(phantom.shape[1:]), tof=True)

    print(f"Phantom shape: {phantom.shape}")
    plot_phantom_frame(phantom, frame=0)

    print(f"Sampling {num_events_per_frame} TOF events per frame ({num_frames} frames) ...")
    event_frames = []
    point_clouds = []
    for f in range(num_frames):
        sino = proj(phantom[f])
        indices = sample_events_from_sinogram(sino, num_events_per_frame)
        p1, p2, tof_bins = get_lor_endpoints(indices, proj)
        points = localize_events(p1, p2, tof_bins, proj)
        event_frames.append(indices)
        point_clouds.append(points)
        print(f"  frame {f + 1}/{num_frames}: {points.shape[0]} events")

    print(f"Computing per-frame Betti curve PCFs (method={gating_method!r}) ...")
    pcf_tensors = compute_frame_pcfs(
        event_frames,
        proj,
        method=gating_method,
        max_dim=1,
        subsample='voxel',
        subsample_kwargs={'voxel_size': 10.0},
    )

    print("Computing inter-frame L2 distance matrix ...")
    dist_matrix = compute_pcf_distance_matrix(pcf_tensors)
    dist_matrix = np.asarray(dist_matrix)
    print(f"Distance matrix shape: {dist_matrix.shape}  range: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]")

    plot_distance_matrix(
        dist_matrix,
        title=f"Inter-frame Betti curve L2 distance ({gating_method})",
        labels=list(range(num_frames)),
        cbar_label="L2 distance",
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "distance_matrix.png"), dpi=300, bbox_inches="tight")
    else:
        plt.savefig("distance_matrix_10mm.png", dpi=300, bbox_inches="tight")

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
    ax.set_ylabel("Betti-curve L2 distance")
    ax.set_title(f"Cyclic correlation  ρ = {rho:.3f}")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "cyclic_correlation.png") if save_dir else "cyclic_correlation.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print("Done!")


def intra_variability_tof_tda(
    num_frames: int = 10,
    num_events_per_frame: int = 35000,
    num_samples: int = 5,
    ref_frame: int = 0,
    gating_method: str = 'masspcf',
    xcat_path: str = XCAT_PATH,
    save_dir: str | None = None,
) -> None:
    print("=" * 60)
    print("Running: Intra-Frame Variability TOF TDA")
    print("=" * 60)

    device = get_device()

    print(f"Loading XCAT phantom from {xcat_path}")
    full = load_xcat(xcat_path, device=device)
    phantom = full[:num_frames, 300:300 + 109, :, :]
    proj = get_mct_projector(device=device, img_shape=tuple(phantom.shape[1:]), tof=True)
    print(f"Phantom shape: {phantom.shape}")

    total_jobs = num_frames * num_samples
    print(f"Sampling {total_jobs} event sets ({num_samples} samples × {num_frames} frames) ...")
    # Flat layout: [f0s0, f0s1, ..., f(F-1)s(S-1)] — preserves (frame, sample) indexing via flat_idx
    all_event_frames = []
    for f in range(num_frames):
        sino = proj(phantom[f])
        for s in range(num_samples):
            indices = sample_events_from_sinogram(sino, num_events_per_frame)
            all_event_frames.append(indices)
            print(f"  frame {f + 1}/{num_frames}, sample {s + 1}/{num_samples}")

    print(f"Computing {total_jobs} Betti curve PCFs (method={gating_method!r}) ...")
    pcf_tensors = compute_frame_pcfs(
        all_event_frames,
        proj,
        method=gating_method,
        max_dim=1,
        subsample='voxel',
        subsample_kwargs={'voxel_size': 10.0},
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
                intra_results[f].append(float(full_dist[flat_idx(f, s1), flat_idx(f, s2)]))

    inter_results: dict[int, list[float]] = {f: [] for f in range(num_frames)}
    for f in range(num_frames):
        for s0 in range(num_samples):
            for sf in range(num_samples):
                if f == ref_frame and s0 == sf:
                    continue
                inter_results[f].append(float(full_dist[flat_idx(ref_frame, s0), flat_idx(f, sf)]))

    intra_frame_means = np.array([
        np.mean(intra_results[f]) if intra_results[f] else 0.0
        for f in range(num_frames)
    ])
    inter_frame_means = np.array([np.mean(inter_results[f]) for f in range(num_frames)])

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
        ax.set_ylabel('Betti-curve L2 distance')
        ax.set_title(f'Betti-curve L2 from frame {ref_frame} – TOF TDA ({gating_method})')
        ax.set_xticks(frame_indices)
        ax.legend(frameon=True, edgecolor='black', fancybox=False, loc='upper left')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        fig.tight_layout()
        _save_or_show(
            fig,
            os.path.join(save_dir, "intra_tof_tda_errorbar.png") if save_dir else None,
        )

    print("Done!")


if __name__ == "__main__":
    intra_variability_tof_tda(
        num_frames=20,
        num_events_per_frame=35000,
        num_samples=5,
        ref_frame=0,
        gating_method='masspcf',
        save_dir='figures/intra_variability_tof_tda',
    )
