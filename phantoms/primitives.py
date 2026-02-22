import torch

def create_sphere_phantom(shape, radius, center=None, intensity=1.0):
    """Creates an N-dimensional spherical phantom (sphere in 3D, disk in 2D).

    Args:
        shape (tuple): Grid size, e.g., (X, Y, Z) or (X, Y).
        radius (float): Radius of the shape in voxels/pixels.
        center (tuple): Center position matching the length of shape.
            Relative to the grid middle. Defaults to (0, ..., 0).
        intensity (float): Value inside the shape.

    Returns:
        torch.Tensor: Tensor with the generated sphere/disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure center matches shape dimensions
    if center is None:
        center = tuple(0.0 for _ in shape)

    # Generate coordinate grids
    axes = [torch.linspace(-s // 2, s // 2, s, device=device) for s in shape]
    grid = torch.meshgrid(*axes, indexing='ij')

    # Calculate squared distance
    dist_sq = sum((grid[i] - center[i])**2 for i in range(len(shape)))

    return (dist_sq <= radius**2).float() * intensity

def create_ellipsoid_phantom(shape, radii, center=None, intensity=1.0):
    """Creates an N-dimensional ellipsoid phantom.

    Args:
        shape (tuple): Grid size, e.g., (X, Y, Z) or (X, Y).
        radii (tuple): Radii for each axis, must match length of shape.
        center (tuple): Center position matching the length of shape.
            Relative to the grid middle. Defaults to (0, ..., 0).
        intensity (float): Value inside the shape.

    Returns:
        torch.Tensor: Tensor with the generated ellipsoid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if center is None:
        center = tuple(0.0 for _ in shape)

    axes = [torch.linspace(-s // 2, s // 2, s, device=device) for s in shape]
    grid = torch.meshgrid(*axes, indexing='ij')

    # Standard ellipsoid equation
    dist_sum = sum(((grid[i] - center[i]) / radii[i]) ** 2 for i in range(len(shape)))

    return (dist_sum <= 1.0).float() * intensity

def create_box_phantom(shape, side_lengths, center=None, intensity=1.0):
    """Creates an N-dimensional box phantom (cube in 3D, rectangle in 2D).

    Args:
        shape (tuple): Grid size, e.g., (X, Y, Z) or (X, Y).
        side_lengths (tuple): Full length of each side.
        center (tuple): Center position relative to grid middle.
            Defaults to (0, ..., 0).
        intensity (float): Value inside the shape.

    Returns:
        torch.Tensor: Tensor with the generated box.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if center is None:
        center = tuple(0.0 for _ in shape)

    axes = [torch.linspace(-s // 2, s // 2, s, device=device) for s in shape]
    grid = torch.meshgrid(*axes, indexing='ij')

    masks = [torch.abs(grid[i] - center[i]) <= (side_lengths[i] / 2.0) for i in range(len(shape))]

    # Logical AND across all dimensions
    box_mask = masks[0]
    for m in masks[1:]:
        box_mask = box_mask & m

    return box_mask.float() * intensity


def create_morphed_phantom(shape, radius, center=None, morph_factor=0.0, intensity=1.0):
    """Creates an N-dimensional phantom that morphs from a sphere to a cube.

    Args:
        shape (tuple): Grid size, e.g., (X, Y, Z) or (X, Y).
        radius (float): Size of the phantom (radius for sphere, half-side for cube).
        center (tuple): Center position relative to grid middle.
            Defaults to (0, ..., 0).
        morph_factor (float): 0.0 for a sphere, 1.0 for an approximate cube.
        intensity (float): Value inside the shape.

    Returns:
        torch.Tensor: Tensor with the generated morphed shape.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if center is None:
        center = tuple(0.0 for _ in shape)

    # Generate coordinate grids
    axes = [torch.linspace(-s // 2, s // 2, s, device=device) for s in shape]
    grid = torch.meshgrid(*axes, indexing='ij')

    # Map morph_factor (0 to 1) to p-norm exponent (2 to 10)
    # p=2 is a Euclidean sphere; p=10 is a visually sharp cube
    p = 2.0 + (morph_factor * 12.0)

    # Calculate p-norm distance: sum(|x_i - c_i|^p)^(1/p)
    # We normalize by radius to keep the threshold at 1.0
    dist_p_sum = sum((torch.abs(grid[i] - center[i]) / radius) ** p for i in range(len(shape)))

    # The shape is defined where the p-norm distance <= 1.0
    return (dist_p_sum <= 1.0).float() * intensity