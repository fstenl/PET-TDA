import torch
import numpy as np
import parallelproj
from typing import Optional


def histogram_events(event_indices: torch.Tensor, proj: parallelproj.RegularPolygonPETProjector) -> torch.Tensor:
    """Convert list-mode indices to a TOF sinogram histogram."""
    # Ensure event_indices is a 1D tensor of integers to prevent silent bincount failures
    if event_indices.ndim != 1 or event_indices.is_floating_point():
         raise ValueError("event_indices must be a 1D tensor of integers (e.g., torch.int64).")

    num_bins = int(np.prod(proj.out_shape))
    counts = torch.bincount(event_indices, minlength=num_bins).to(dtype=torch.float32)
    return counts.reshape(proj.out_shape)


def run_mlem(
    op: parallelproj.LinearOperator,
    measured_sino: torch.Tensor,
    num_iter: int = 3,
    contamination: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a few iterations of basic TOF-MLEM reconstruction."""
    device = measured_sino.device
    if contamination is None:
        contamination = torch.zeros_like(measured_sino, dtype=torch.float32, device=device)

    x = torch.ones(op.in_shape, dtype=torch.float32, device=device)
    # Adjoint is now calculated dynamically from the operator (which can include attenuation/resolution)
    adjoint_ones = op.adjoint(torch.ones(op.out_shape, dtype=torch.float32, device=device))
    eps = torch.finfo(torch.float32).eps

    for it in range(num_iter):
        ybar = op(x) + contamination
        ratio = measured_sino / torch.clamp(ybar, min=eps)
        correction = op.adjoint(ratio)
        x = x * correction / torch.clamp(adjoint_ones, min=eps)
        print(f"MLEM iteration {it + 1}/{num_iter}")

    return x
