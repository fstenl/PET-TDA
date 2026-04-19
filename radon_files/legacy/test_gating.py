import matplotlib
# Use the 'Agg' backend to generate plots without needing a UI/display
matplotlib.use('Agg') 

import numpy as np
import torch
import matplotlib.pyplot as plt
import array_api_compat.torch as xp

from scanner import get_mCT_scanner, sample_events_from_sinogram
from persistence import compute_frame_pcfs, compute_distance_matrix
from visualization import plot_distance_matrix

def main():
    dev = "cpu"
    print(f"Using device: {dev}")

    proj = get_mCT_scanner(xp, dev)

    n_resp_frames = 20      # 2 respiratory cycles of 10 frames each
    n_events_per_frame = 35_000

    # TDA backend: "masspcf" (batched, fast), "witness" (GUDHI), or "ripser"
    method = "masspcf"
    PHANTOM_PATH = "../../../data/respiratory_only.npy"

    # Forward-project each respiratory frame
    print(f"Forward-projecting {n_resp_frames} frames from {PHANTOM_PATH} ...")
    xcat = np.load(PHANTOM_PATH)
    sinograms = []
    for f in range(n_resp_frames):
        image = xp.asarray(xcat[f, 300:300 + 109, :, :])
        sino = proj(image)
        sinograms.append(sino)
        print(f"  frame {f + 1}/{n_resp_frames} done, shape {tuple(sino.shape)}")
        del sino
        del image
        
    # One frame per respiratory frame
    print(f"Sampling {n_events_per_frame} events from each of {n_resp_frames} frames ...")
    frames = [
        sample_events_from_sinogram(sinograms[f], n_events_per_frame)
        for f in range(n_resp_frames)
    ]
    gt_frame = list(range(n_resp_frames))
    print(f"{len(frames)} frames of {n_events_per_frame} events")

    # Per-frame Betti curve PCFs
    print(f"Computing per-frame Betti curve PCFs (method={method!r}) ...")
    pcf_tensors = compute_frame_pcfs(
        frames,
        proj,
        method=method,
        max_dim=1,
        subsample="voxel",
        subsample_kwargs={"voxel_size": 5.0}, 
    )
    print(f"{len(pcf_tensors)} PcfTensor(s), each shape {pcf_tensors[0].shape}")

    print("Computing inter-frame distance matrix ...")
    D = compute_distance_matrix(pcf_tensors)
    print(f"Shape: {D.shape}  range: [{D.min():.4f}, {D.max():.4f}]")

    # Plotting
    print("Generating plot...")
    plot_distance_matrix(
        D,
        labels=gt_frame,
        title=f"Inter-frame Betti curve L2 distance matrix — {method}",
        cbar_label="L2 distance",
    )

    # Save the plot instead of showing it
    print("Saving plot image to distance_matrix_5mm.png...")
    plt.savefig("distance_matrix_5mm.png", dpi=300, bbox_inches="tight")
    
    print("Script completed successfully!")

if __name__ == "__main__":
    main()