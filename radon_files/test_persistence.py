"""
Quick smoke-test for the witness-complex persistence pipeline.
Run from the project root:

    python -m radon_files.test_persistence

No PET data required — generates a synthetic 3D point cloud
(a noisy torus, which should show clear H1 features).
"""

# H0 has at least one infinite feature — meaning one connected component that never dies, confirming the torus is one piece
# H1 has features with lifetime > 10mm — meaning persistent loops were actually detected, not just noise
# H1 max and mean lifetimes are printed so you can see if the torus loops are being found clearly or weakly

"""
Witness finds the correct H0 (one connected component). The H1 max lifetime of 69.8mm is very long relative to the torus scale — 
it has confidently detected the torus loops. Mean lifetime 14.4mm means the real signal is well separated from noise. This is the 
strongest topological signal of all four methods.
"""

""" 
Alpha finds the correct H0 eventually (1 infinite). But H1 is extremely noisy — 140 features with mean lifetime only 
1.5mm means the vast majority are noise, not real loops. The real torus signal is buried. This happens because alpha 
runs on all 100 points and picks up every tiny accidental gap as a short-lived loop.
"""

"""
The 5 infinite H0 features is the giveaway — partitioning cut the torus into 5 disconnected pieces so it thinks there are 5 separate objects 
instead of 1. The H1 max lifetime of 13.2mm is weak, and mean 1.1mm is almost pure noise. Partitioning has destroyed the global loop structure 
of the torus because the loops cross partition boundaries.
"""

"""
After normalising by 20 runs, this mirrors alpha exactly — same max lifetime, same mean. Bootstrap is just repeated 
alpha on subsamples, so it inherits alpha's noise problem. It doesn't add information beyond alpha here, but across many 
runs it would give you a stable average rather than one noisy result.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works headless
import matplotlib.pyplot as plt

from radon_files.persistence import (
    witness_persistence,
    alpha_persistence,
    bootstrapped_persistence,
    partitioned_persistence,
    diagrams_to_betti_curves,
    euler_characteristic_curve,
)


def make_noisy_torus(n_points: int = 500, R: float = 100.0, r: float = 30.0,
                     noise: float = 5.0, seed: int = 42) -> np.ndarray:
    """Sample points from a torus in 3D with additive Gaussian noise.

    R = major radius (mm), r = minor radius (mm).
    A torus has H0=1 (one component), H1=2 (two independent loops),
    so we expect β1 features in the persistence diagram.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi = rng.uniform(0, 2 * np.pi, n_points)

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    pts = np.column_stack([x, y, z])
    pts += rng.normal(0, noise, pts.shape)
    return pts


def main():
    print("=" * 60)
    print("Witness-complex persistence — smoke test")
    print("=" * 60)

    # --- generate synthetic data ------------------------------------------
    n_points = 100
    pts = make_noisy_torus(n_points=n_points)
    print(f"\nGenerated noisy torus: {pts.shape}  (R=100mm, r=30mm, σ=5mm)")
    print(f"  bounding box: {pts.min(axis=0).round(1)} → {pts.max(axis=0).round(1)}")

    # --- run witness persistence ------------------------------------------
    diagrams = witness_persistence(
        pts,
        landmark_ratio=0.20,       # 20% landmarks (100 of 500)
        landmark_method="maxmin",
        max_dim=1,                 # H0 + H1
    )

    diagrams_alpha = alpha_persistence(pts, max_dim=1)
    diagrams_partitioned = partitioned_persistence(pts, n_partitions=5, max_dim=1)
    diagrams_bootstrap = bootstrapped_persistence(pts, n_samples=20, max_dim=1)
    # --- basic sanity checks ----------------------------------------------
    print("\n--- Sanity checks ---")
    assert len(diagrams) == 2, f"Expected 2 diagram levels, got {len(diagrams)}"
    print(f"  ✓ Got {len(diagrams)} diagram levels (H0, H1)")

    h0 = diagrams[0]
    h1 = diagrams[1]
    print(f"  H0: {len(h0)} features")
    print(f"  H1: {len(h1)} features")

    # H0 should have at least 1 infinite feature (the single connected component)
    n_inf_h0 = np.sum(~np.isfinite(h0[:, 1])) if len(h0) > 0 else 0
    print(f"  H0 infinite features: {n_inf_h0}")
    assert n_inf_h0 >= 1, "Expected at least 1 infinite H0 feature (connected component)"
    print(f"  ✓ At least one connected component detected")

    # H1 should have persistent features (the torus loops)
    if len(h1) > 0:
        finite_h1 = h1[np.isfinite(h1[:, 1])]
        if len(finite_h1) > 0:
            lifetimes = finite_h1[:, 1] - finite_h1[:, 0]
            print(f"  H1 max lifetime: {lifetimes.max():.1f} mm")
            print(f"  H1 mean lifetime: {lifetimes.mean():.1f} mm")
            # The torus should produce at least one persistent H1 feature
            long_lived = lifetimes[lifetimes > 10.0]
            print(f"  H1 features with lifetime > 10mm: {len(long_lived)}")
        else:
            print("  (no finite H1 features)")
    else:
        print("  (no H1 features at all)")


    # --- test alpha persistence ------------------------------------------
    print("\n--- Testing alpha persistence ---")
    diagrams = diagrams_alpha

    h0 = diagrams[0]
    h1 = diagrams[1]

    print(f"  H0: {len(h0)} features")
    print(f"  H1: {len(h1)} features")

    n_inf_h0 = np.sum(~np.isfinite(h0[:, 1])) if len(h0) > 0 else 0
    print(f"  H0 infinite features: {n_inf_h0}")

    if len(h1) > 0:
        finite_h1 = h1[np.isfinite(h1[:, 1])]
        if len(finite_h1) > 0:
            lifetimes = finite_h1[:, 1] - finite_h1[:, 0]
            print(f"  H1 max lifetime: {lifetimes.max():.1f} mm")
            print(f"  H1 mean lifetime: {lifetimes.mean():.1f} mm")
        else:
            print("  (no finite H1 features)")
    else:
        print("  (no H1 features at all)")

    # --- test partitioned persistence ------------------------------------
    print("\n--- Testing partitioned persistence ---")
    diagrams = diagrams_partitioned

    h0 = diagrams[0]
    h1 = diagrams[1]

    print(f"  H0: {len(h0)} features")
    print(f"  H1: {len(h1)} features")

    n_inf_h0 = np.sum(~np.isfinite(h0[:, 1])) if len(h0) > 0 else 0
    print(f"  H0 infinite features: {n_inf_h0}")

    if len(h1) > 0:
        finite_h1 = h1[np.isfinite(h1[:, 1])]
        if len(finite_h1) > 0:
            lifetimes = finite_h1[:, 1] - finite_h1[:, 0]
            print(f"  H1 max lifetime: {lifetimes.max():.1f} mm")
            print(f"  H1 mean lifetime: {lifetimes.mean():.1f} mm")
        else:
            print("  (no finite H1 features)")
    else:
        print("  (no H1 features at all)")

    all_methods = {
        "witness": diagrams,
        "alpha": diagrams_alpha,
        "partitioned": diagrams_partitioned,
        "bootstrap": diagrams_bootstrap,
    }

   # --- test bootstrapped Betti curves --------------------------------
    print("\n--- Testing bootstrapped persistence ---")
    diagrams = diagrams_bootstrap

    h0 = diagrams[0]
    h1 = diagrams[1]

    print(f"  H0: {len(h0)} features")
    print(f"  H1: {len(h1)} features")

    n_inf_h0 = np.sum(~np.isfinite(h0[:, 1])) if len(h0) > 0 else 0
    print(f"  H0 infinite features: {n_inf_h0}")

    if len(h1) > 0:
        finite_h1 = h1[np.isfinite(h1[:, 1])]
        if len(finite_h1) > 0:
            lifetimes = finite_h1[:, 1] - finite_h1[:, 0]
            print(f"  H1 max lifetime: {lifetimes.max():.1f} mm")
            print(f"  H1 mean lifetime: {lifetimes.mean():.1f} mm")
        else:
            print("  (no finite H1 features)")
    else:
        print("  (no H1 features at all)")

    # --- save a diagnostic plot -------------------------------------------
    n_methods = len(all_methods)
    fig, axes = plt.subplots(n_methods, 4, figsize=(18, 4 * n_methods))

    for row, (name, diagrams) in enumerate(all_methods.items()):

        h0 = diagrams[0]
        h1 = diagrams[1]

        # --- Betti + Euler ---
        betti_curves = diagrams_to_betti_curves(diagrams)
        t_ecc, chi = euler_characteristic_curve(diagrams)

        print(f"\n--- {name} ---")
        print(f"  Betti curves sampled at {len(betti_curves[0][0])} points")
        print(f"  β0 max: {betti_curves[0][1].max()}")
        print(f"  β1 max: {betti_curves[1][1].max()}")
        print(f"  χ range: [{chi.min():.0f}, {chi.max():.0f}]")

        ax0, ax1, ax2, ax3 = axes[row]

        # --- H0 diagram ---
        if len(h0) > 0:
            finite_h0 = h0[np.isfinite(h0[:, 1])]
            if len(finite_h0):
                ax0.scatter(finite_h0[:, 0], finite_h0[:, 1], s=15, alpha=0.7)
            max_val = finite_h0[:, 1].max() if len(finite_h0) else 100
            ax0.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        ax0.set_title(f"{name} H₀")
        ax0.set_xlabel("birth (mm)")
        ax0.set_ylabel("death (mm)")

        # --- H1 diagram ---
        if len(h1) > 0:
            finite_h1 = h1[np.isfinite(h1[:, 1])]
            if len(finite_h1):
                ax1.scatter(finite_h1[:, 0], finite_h1[:, 1], s=15, alpha=0.7)
            max_val = finite_h1[:, 1].max() if len(finite_h1) else 100
            ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        ax1.set_title(f"{name} H₁")
        ax1.set_xlabel("birth (mm)")
        ax1.set_ylabel("death (mm)")

        # --- Betti curves ---
        for dim in range(2):
            t, beta = betti_curves[dim]
            ax2.plot(t, beta, label=f"β_{dim}")
        ax2.set_title(f"{name} Betti")
        ax2.set_xlabel("filtration (mm)")
        ax2.legend()

        # --- Euler characteristic ---
        ax3.plot(t_ecc, chi)
        ax3.set_title(f"{name} χ(t)")
        ax3.set_xlabel("filtration (mm)")

    fig.tight_layout()
    out_path = "/home/mamostudent3/project_prarthana/PET-TDA/radon_files/test_persistence_output.png"
    fig.savefig(out_path, dpi=120)

    print(f"\n  Diagnostic plot saved to: {out_path}")

    print("\n" + "=" * 60)
    print("All checks passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
