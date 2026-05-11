import math
import torch

def create_sphere_phantom(shape: tuple, radius: float, center: tuple | None = None, intensity: float = 1.0, device: str = 'cpu') -> torch.Tensor:
    """Creates an N-dimensional spherical phantom (sphere in 3D, disk in 2D).

    Args:
        shape (tuple): Grid size, e.g., (X, Y, Z) or (X, Y).
        radius (float): Radius of the shape in voxels/pixels.
        center (tuple): Center position matching the length of shape.
            Relative to the grid middle. Defaults to (0, ..., 0).
        intensity (float): Value inside the shape.
        device (str): Compute device. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Tensor with the generated sphere/disk.
    """

    # Ensure center matches shape dimensions
    if center is None:
        center = tuple(0.0 for _ in shape)

    # Generate coordinate grids
    axes = [torch.linspace(-s // 2, s // 2, s, device=device) for s in shape]
    grid = torch.meshgrid(*axes, indexing='ij')

    # Calculate squared distance
    dist_sq = sum((grid[i] - center[i])**2 for i in range(len(shape)))

    return (dist_sq <= radius**2).float() * intensity

def create_ellipsoid_phantom(shape: tuple, radii: tuple, center: tuple | None = None, intensity: float = 1.0, device: str = 'cpu') -> torch.Tensor:
    """Creates an N-dimensional ellipsoid phantom.

    Args:
        shape (tuple): Grid size, e.g., (X, Y, Z) or (X, Y).
        radii (tuple): Radii for each axis, must match length of shape.
        center (tuple): Center position matching the length of shape.
            Relative to the grid middle. Defaults to (0, ..., 0).
        intensity (float): Value inside the shape.
        device (str): Compute device. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Tensor with the generated ellipsoid.
    """

    if center is None:
        center = tuple(0.0 for _ in shape)

    axes = [torch.linspace(-s // 2, s // 2, s, device=device) for s in shape]
    grid = torch.meshgrid(*axes, indexing='ij')

    # Standard ellipsoid equation
    dist_sum = sum(((grid[i] - center[i]) / radii[i]) ** 2 for i in range(len(shape)))

    return (dist_sum <= 1.0).float() * intensity

def create_box_phantom(shape: tuple, side_lengths: tuple, center: tuple | None = None, intensity: float = 1.0, device: str = 'cpu') -> torch.Tensor:
    """Creates an N-dimensional box phantom (cube in 3D, rectangle in 2D).

    Args:
        shape (tuple): Grid size, e.g., (X, Y, Z) or (X, Y).
        side_lengths (tuple): Full length of each side.
        center (tuple): Center position relative to grid middle.
            Defaults to (0, ..., 0).
        intensity (float): Value inside the shape.
        device (str): Compute device. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Tensor with the generated box.
    """

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


def create_morphed_phantom(shape: tuple, radius: float, center: tuple | None = None, morph_factor: float = 0.0, intensity: float = 1.0, device: str = 'cpu') -> torch.Tensor:
    """Creates an N-dimensional phantom that morphs from a sphere to a cube.

    Args:
        shape (tuple): Grid size, e.g., (X, Y, Z) or (X, Y).
        radius (float): Size of the phantom (radius for sphere, half-side for cube).
        center (tuple): Center position relative to grid middle.
            Defaults to (0, ..., 0).
        morph_factor (float): 0.0 for a sphere, 1.0 for an approximate cube.
        intensity (float): Value inside the shape.
        device (str): Compute device. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Tensor with the generated morphed shape.
    """

    if center is None:
        center = tuple(0.0 for _ in shape)

    # Generate coordinate grids
    axes = [torch.linspace(-s // 2, s // 2, s, device=device) for s in shape]
    grid = torch.meshgrid(*axes, indexing='ij')

    # Map morph_factor (0 to 1) to p-norm exponent (2 to 10)
    # p=2 is a Euclidean sphere; p=10 is a visually sharp cube
    p = 2.0 + (morph_factor * 10.0)

    # Calculate p-norm distance: sum of (|x_i - center_i| / radius)^p across all dimensions
    # We normalize by radius to keep the threshold at 1.0
    dist_p_sum = sum((torch.abs(grid[i] - center[i]) / radius) ** p for i in range(len(shape)))

    # The shape is defined where the p-norm distance <= 1.0
    return (dist_p_sum <= 1.0).float() * intensity


def _regular_simplex_vertices(ndim: int, radius: float) -> list[list[float]]:
    """Compute N+1 vertices of a regular N-simplex centered at origin with given circumradius.

    Uses the standard simplex in R^{N+1} projected down to R^N via Gram-Schmidt.

    Args:
        ndim (int): Number of spatial dimensions.
        radius (float): Circumradius of the simplex.

    Returns:
        list[list[float]]: List of N+1 vertices, each a list of N coordinates.
    """
    n = ndim

    # Centered standard simplex: w_i = e_i - centroid in R^{n+1}
    centroid_val = 1.0 / (n + 1)
    w = []
    for i in range(n + 1):
        wi = [-centroid_val] * (n + 1)
        wi[i] += 1.0
        w.append(wi)

    # Gram-Schmidt to obtain n orthonormal basis vectors for the subspace
    basis = []
    for i in range(n):
        v = w[i][:]
        for b in basis:
            dot = sum(v[k] * b[k] for k in range(n + 1))
            v = [v[k] - dot * b[k] for k in range(n + 1)]
        norm = math.sqrt(sum(x ** 2 for x in v))
        v = [x / norm for x in v]
        basis.append(v)

    # Project each w_i onto the basis and scale to desired circumradius.
    # The circumradius of the centered standard simplex is sqrt(n / (n + 1)).
    scale = radius / math.sqrt(n / (n + 1))
    vertices = []
    for i in range(n + 1):
        coords = [
            sum(w[i][k] * basis[d][k] for k in range(n + 1)) * scale
            for d in range(n)
        ]
        vertices.append(coords)

    return vertices


def create_simplex_phantom(shape: tuple, radius: float, center: tuple | None = None, intensity: float = 1.0, device: str = 'cpu') -> torch.Tensor:
    """Creates an N-dimensional simplex phantom (triangle in 2D, tetrahedron in 3D).

    The simplex is regular (equilateral) and inscribed in a sphere of the given
    circumradius.

    Args:
        shape (tuple): Grid size, e.g., (X, Y, Z) or (X, Y).
        radius (float): Circumradius of the simplex in voxels/pixels.
        center (tuple): Center position matching the length of shape.
            Relative to the grid middle. Defaults to (0, ..., 0).
        intensity (float): Value inside the shape.
        device (str): Compute device. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Tensor with the generated simplex.
    """
    ndim = len(shape)

    if center is None:
        center = tuple(0.0 for _ in shape)

    # Generate coordinate grids
    axes = [torch.linspace(-s // 2, s // 2, s, device=device) for s in shape]
    grid = torch.meshgrid(*axes, indexing='ij')

    # Compute vertices of a regular simplex
    vertices = _regular_simplex_vertices(ndim, radius)

    # A point is inside the simplex iff it is on the inward side of all N+1 faces.
    # For a regular simplex centred at the origin, the inward normal of face i
    # (opposite vertex i) is in the direction of v_i.
    # Condition per face: v_i · (x - v_j) >= 0, with j != i.
    mask = torch.ones(shape, dtype=torch.bool, device=device)
    for i in range(ndim + 1):
        j = (i + 1) % (ndim + 1)
        dot = sum(
            vertices[i][d] * (grid[d] - center[d] - vertices[j][d])
            for d in range(ndim)
        )
        mask = mask & (dot >= 0)

    return mask.float() * intensity
