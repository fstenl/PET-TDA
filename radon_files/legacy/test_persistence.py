"""
Smoke test for the TDA pipeline (subsample → backend → vectorise).

    python test_persistence.py

No PET data required — generates a synthetic noisy torus, which has known
topology (H0 = 1, H1 = 2), and runs both TDA backends against it.  Also
exercises each subsampling strategy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from persistence import (
    compute_persistence,
    diagrams_to_betti_curves,
    euler_characteristic_curve,
)
import subsampling


def make_noisy_torus(n_points: int = 500, R: float = 100.0, r: float = 30.0,
                     noise: float = 5.0, seed: int = 42) -> np.ndarray:
    """Sample points from a torus in 3D with additive Gaussian noise.

    A torus has H0=1 (one component), H1=2 (two independent loops).
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


def _assert_torus_like(diagrams, label):
    """Basic topological sanity: ≥1 infinite H0 feature, ≥1 long-lived H1."""
    h0, h1 = diagrams[0], diagrams[1]
    n_inf_h0 = int(np.sum(~np.isfinite(h0[:, 1]))) if len(h0) else 0
    assert n_inf_h0 >= 1, f"[{label}] expected ≥1 infinite H0 feature, got {n_inf_h0}"
    finite_h1 = h1[np.isfinite(h1[:, 1])] if len(h1) else h1
    lifetimes = finite_h1[:, 1] - finite_h1[:, 0] if len(finite_h1) else np.empty(0)
    long_lived = int(np.sum(lifetimes > 10.0))
    assert long_lived >= 1, f"[{label}] expected ≥1 long-lived H1 feature, got {long_lived}"
    print(f"  ✓ [{label}] H0 infinite: {n_inf_h0}, H1 long-lived: {long_lived}")


def test_subsamplers(pts):
    """Each subsampler should reduce the cloud and return a (n, 3) array."""
    print("\n--- Subsampling strategies ---")
    n_target = 200

    out = subsampling.random_uniform(pts, n_target, rng=0)
    assert out.shape == (n_target, 3), out.shape
    print(f"  ✓ random_uniform: {pts.shape} → {out.shape}")

    out = subsampling.farthest_point(pts, n_target)
    assert out.shape[1] == 3 and out.shape[0] <= n_target + 1
    print(f"  ✓ farthest_point: {pts.shape} → {out.shape}")

    out = subsampling.voxel_grid(pts, voxel_size=10.0)
    assert out.shape[1] == 3 and out.shape[0] < len(pts)
    print(f"  ✓ voxel_grid (10mm): {pts.shape} → {out.shape}")

    out = subsampling.poisson_disk(pts, min_distance=10.0, rng=0)
    assert out.shape[1] == 3 and out.shape[0] < len(pts)
    print(f"  ✓ poisson_disk (10mm): {pts.shape} → {out.shape}")


def main():
    print("=" * 60)
    print("TDA pipeline smoke test")
    print("=" * 60)

    pts = make_noisy_torus(n_points=500)
    print(f"\nGenerated noisy torus: {pts.shape}")

    test_subsamplers(pts)

    # --- Witness backend -----------------------------------------------------
    print("\n--- Witness backend ---")
    witness_dgms = compute_persistence(
        pts,
        method="witness",
        method_kwargs={
            "landmark_ratio": 0.20,
            "landmark_method": "farthest",
            "max_dim": 1,
        },
    )
    _assert_torus_like(witness_dgms, "witness")

    # --- Ripser backend ------------------------------------------------------
    print("\n--- Ripser backend ---")
    ripser_dgms = compute_persistence(
        pts,
        subsample="farthest",
        subsample_kwargs={"n": 200},   # keep Ripser fast on the smoke test
        method="ripser",
        method_kwargs={"max_dim": 1},
    )
    _assert_torus_like(ripser_dgms, "ripser")

    # --- Vectorisation contract ----------------------------------------------
    for label, dgms in [("witness", witness_dgms), ("ripser", ripser_dgms)]:
        curves = diagrams_to_betti_curves(dgms)
        t_ecc, chi = euler_characteristic_curve(dgms)
        print(f"  [{label}] β0 max: {curves[0][1].max()}, "
              f"β1 max: {curves[1][1].max()}, "
              f"χ range: [{chi.min():.0f}, {chi.max():.0f}]")

    # --- Diagnostic plot -----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (label, dgms) in enumerate([("witness", witness_dgms),
                                         ("ripser", ripser_dgms)]):
        h0, h1 = dgms[0], dgms[1]
        curves = diagrams_to_betti_curves(dgms)

        for col, (dgm, title) in enumerate([(h0, f"{label} H₀"),
                                            (h1, f"{label} H₁")]):
            finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) else dgm
            if len(finite):
                axes[row, col].scatter(finite[:, 0], finite[:, 1], s=15, alpha=0.7)
                mv = finite[:, 1].max()
                axes[row, col].plot([0, mv], [0, mv], 'k--', alpha=0.3)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("birth (mm)")
            axes[row, col].set_ylabel("death (mm)")

        for dim in range(2):
            t, beta = curves[dim]
            axes[row, 2].plot(t, beta, label=f"β_{dim}")
        axes[row, 2].set_title(f"{label} Betti curves")
        axes[row, 2].set_xlabel("filtration (mm)")
        axes[row, 2].legend()

    fig.tight_layout()
    out_path = "test_persistence_output.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n  Diagnostic plot saved to: {out_path}")

    print("\n" + "=" * 60)
    print("All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
