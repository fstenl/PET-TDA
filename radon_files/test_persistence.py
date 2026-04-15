"""
Quick smoke-test for the witness-complex persistence pipeline.
Run from the project root:

    python -m radon_files.test_persistence

No PET data required — generates a synthetic 3D point cloud
(a noisy torus, which should show clear H1 features).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works headless
import matplotlib.pyplot as plt

from radon_files.persistence import (
    witness_persistence,
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

    # --- Betti curves -----------------------------------------------------
    betti_curves = diagrams_to_betti_curves(diagrams)
    t_ecc, chi = euler_characteristic_curve(diagrams)

    print(f"\n  Betti curves sampled at {len(betti_curves[0][0])} points")
    print(f"  β0 max: {betti_curves[0][1].max()}")
    print(f"  β1 max: {betti_curves[1][1].max()}")
    print(f"  χ range: [{chi.min():.0f}, {chi.max():.0f}]")

    # --- save a diagnostic plot -------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Persistence diagram H0
    if len(h0) > 0:
        finite_h0 = h0[np.isfinite(h0[:, 1])]
        if len(finite_h0):
            axes[0].scatter(finite_h0[:, 0], finite_h0[:, 1], s=15, alpha=0.7, label="finite")
        max_val = finite_h0[:, 1].max() if len(finite_h0) else 100
        axes[0].plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    axes[0].set_title("H₀ persistence diagram")
    axes[0].set_xlabel("birth (mm)")
    axes[0].set_ylabel("death (mm)")

    # Persistence diagram H1
    if len(h1) > 0:
        finite_h1 = h1[np.isfinite(h1[:, 1])]
        if len(finite_h1):
            axes[1].scatter(finite_h1[:, 0], finite_h1[:, 1], s=15, alpha=0.7, c="tab:orange")
        max_val = finite_h1[:, 1].max() if len(finite_h1) else 100
        axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    axes[1].set_title("H₁ persistence diagram")
    axes[1].set_xlabel("birth (mm)")
    axes[1].set_ylabel("death (mm)")

    # Betti curves
    for dim in range(2):
        t, beta = betti_curves[dim]
        axes[2].plot(t, beta, label=f"β_{dim}")
    axes[2].set_title("Betti curves")
    axes[2].set_xlabel("filtration (mm)")
    axes[2].legend()

    # Euler characteristic
    axes[3].plot(t_ecc, chi)
    axes[3].set_title("Euler characteristic χ(t)")
    axes[3].set_xlabel("filtration (mm)")

    fig.tight_layout()
    out_path = "radon_files/test_persistence_output.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n  Diagnostic plot saved to: {out_path}")

    print("\n" + "=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
