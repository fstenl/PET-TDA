import torch


def reconstruct_mlem(sinogram, op, num_iterations=50, contamination=None, verbose=True):
    """Reconstruct a 3D image from a sinogram using MLEM.

    Args:
        sinogram (torch.Tensor): Measured sinogram, shape matching op.out_shape.
        op (LinearOperator): Forward operator, e.g. a plain
            RegularPolygonPETProjector or a CompositeLinearOperator.
        num_iterations (int): Number of MLEM iterations.
        contamination (torch.Tensor | None): Scatter and randoms background,
            same shape as sinogram. Defaults to 1e-6 if None.
        verbose (bool): Print iteration progress if True.

    Returns:
        torch.Tensor: Reconstructed image of shape op.in_shape.
    """
    device = sinogram.device
    dtype = torch.float32

    sinogram = sinogram.to(dtype=dtype)

    if contamination is None:
        contamination = torch.full(sinogram.shape, 1e-6, dtype=dtype, device=device)
    else:
        contamination = contamination.to(dtype=dtype, device=device)

    sensitivity = op.adjoint(torch.ones(sinogram.shape, dtype=dtype, device=device))
    sensitivity = torch.clamp(sensitivity, min=1e-9)

    x = torch.ones(op.in_shape, dtype=dtype, device=device)

    for i in range(num_iterations):
        if verbose:
            print(f"MLEM iteration {i + 1:03d} / {num_iterations:03d}", end="\r")
        expected = op(x) + contamination
        x = x * op.adjoint(sinogram / expected) / sensitivity

    if verbose:
        print()

    return x