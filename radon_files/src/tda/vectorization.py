import numpy as np


def _t_max_from_diagrams(diagrams):
    """Pick the largest finite death time across all homology dimensions."""
    finite_deaths = [
        dgm[:, 1][np.isfinite(dgm[:, 1])]
        for dgm in diagrams if len(dgm)
    ]
    if not finite_deaths:
        return 1.0
    all_deaths = np.concatenate(finite_deaths)
    return float(all_deaths.max()) if len(all_deaths) else 1.0


def diagrams_to_betti_curves(diagrams, t_min=0.0, t_max=None, n_steps=200):
    """Convert persistence diagrams to sampled Betti-number curves via masspcf.

    Args:
        diagrams (list[np.ndarray]): Persistence diagrams, one per dimension.
        t_min (float): Lower filtration bound for sampling.
        t_max (float | None): Upper bound. If None, uses the largest finite
            death time across all dimensions.
        n_steps (int): Number of sample points along the filtration axis.

    Returns:
        dict[int, tuple[np.ndarray, np.ndarray]]: Map from homology dimension
            to (t, beta) where t has shape (n_steps,) and beta has shape
            (n_steps,) with integer Betti numbers.
    """
    from masspcf.persistence import Barcode, barcode_to_betti_curve

    if t_max is None:
        t_max = _t_max_from_diagrams(diagrams)

    t = np.linspace(t_min, t_max, n_steps)
    curves = {}

    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            curves[dim] = (t, np.zeros(n_steps, dtype=int))
            continue
        barcode = Barcode(dgm.astype(np.float64))
        pcf = barcode_to_betti_curve(barcode)
        curves[dim] = (t, pcf(t).astype(int))

    return curves


def euler_characteristic_curve(diagrams, t_min=0.0, t_max=None, n_steps=200):
    """Compute the Euler characteristic curve via masspcf.

    chi(t) = sum_k (-1)^k * beta_k(t)

    Args:
        diagrams (list[np.ndarray]): Persistence diagrams, one per dimension.
        t_min (float): Lower filtration bound for sampling.
        t_max (float | None): Upper bound. If None, uses the largest finite
            death time across all dimensions.
        n_steps (int): Number of sample points along the filtration axis.

    Returns:
        tuple[np.ndarray, np.ndarray]: (t, chi) where t has shape (n_steps,)
            and chi has shape (n_steps,).
    """
    from masspcf.persistence import Barcode, barcode_to_betti_curve

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
