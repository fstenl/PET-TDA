import torch

def static_trajectory(center: tuple, steps: int) -> list[tuple]:
    """Generates a static trajectory at a fixed position

    Args:
        center (tuple): The fixed center position (X, Y, Z) or (X, Y).
        steps (int): Number of time steps.

    Returns:
        list: List of identical center tuples.
    """
    return [center for _ in range(steps)]

def linear_trajectory(start: tuple, end: tuple, steps: int) -> list[tuple]:
    """Generates a linear path between two points in N dimensions.

    Args:
        start (tuple): Starting center position.
        end (tuple): Ending center position.
        steps (int): Number of points along the path.

    Returns:
        list: List of interpolated center tuples.
    """
    # Create interpolated coordinates for each dimension present in start/end
    lin_coords = [torch.linspace(s, e, steps).tolist() for s, e in zip(start, end)]

    # Zip them back into tuples of original dimension
    return list(zip(*lin_coords))

def circular_trajectory(radius: float, center: tuple, steps: int, plane: str = 'xy') -> list[tuple]:
    """Generates a circular path, compatible with both 2D and 3D.

    Args:
        radius (float): Radius of the circular path.
        center (tuple): Origin of the circle (X, Y, Z) or (X, Y).
        steps (int): Number of points along the circle.
        plane (str): The plane to circulate in ('xy', 'xz', 'yz'). Only used in 3D.

    Returns:
        list: List of center tuples.
    """
    angles = torch.linspace(0, 2 * torch.pi, steps)
    cos_vals = (radius * torch.cos(angles))
    sin_vals = (radius * torch.sin(angles))

    if len(center) == 2:
        # 2D case (X, Y)
        x = (center[0] + cos_vals).tolist()
        y = (center[1] + sin_vals).tolist()
        return list(zip(x, y))

    elif len(center) == 3:
        # 3D case (X, Y, Z)
        x_base, y_base, z_base = center
        if plane.lower() == 'xy':
            return [(x_base + cos_vals[i].item(), y_base + sin_vals[i].item(), z_base) for i in range(steps)]
        elif plane.lower() == 'xz':
            return [(x_base + cos_vals[i].item(), y_base, z_base + sin_vals[i].item()) for i in range(steps)]
        elif plane.lower() == 'yz':
            return [(x_base, y_base + cos_vals[i].item(), z_base + sin_vals[i].item()) for i in range(steps)]

    raise ValueError("Center must be a tuple of length 2 or 3.")

def periodic_trajectory(center: tuple, amplitude: tuple, steps: int, frequency: float = 1.0) -> list[tuple]:
    """Generates a sinusoidal periodic path to simulate breathing or cyclic motion.

    Args:
        center (tuple): The baseline origin (X, Y, Z) or (X, Y).
        amplitude (tuple): Max displacement for each axis, matching center length.
        steps (int): Number of time steps.
        frequency (float): Number of full cycles over the sequence duration.

    Returns:
        list: List of center tuples following a sine wave.
    """
    # Create time steps spanning one full cycle 
    t = torch.linspace(0, 2 * torch.pi * frequency, steps)
    sin_t = torch.sin(t)

    # Calculate the displacement for each dimension
    # New position = center + amplitude * sin(t)
    coords = []
    for i in range(len(center)):
        dim_coords = (center[i] + amplitude[i] * sin_t).tolist()
        coords.append(dim_coords)

    # Zip back into tuples of (X, Y, Z) or (X, Y)
    return list(zip(*coords))