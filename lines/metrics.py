import torch

def compute_euclidean_distance(coords: torch.Tensor) -> torch.Tensor:
    """Computes the pairwise Euclidean distance in Plücker space.

    Args:
        coords (torch.Tensor): Plücker coordinates of shape (N, 6).

    Returns:
        torch.Tensor: Symmetric distance matrix of shape (N, N).
    """
    return torch.cdist(coords, coords, p=2)

def compute_hybrid_weighted_distance(coords: torch.Tensor, alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
    """Computes distance using optimized angular and geometric components.

    Args:
        coords (torch.Tensor): Plücker coordinates of shape (N, 6).
        alpha (float): Weight for the angular distance component.
        beta (float): Weight for the shortest geometric distance component.

    Returns:
        torch.Tensor: Symmetric distance matrix of shape (N, N).
    """
    d = coords[:, :3]
    m = coords[:, 3:]
    eps = 1e-10

    # Calculate angular distance between direction vectors
    dot_prod = torch.mm(d, d.t())
    dot_prod = torch.clamp(dot_prod, -1.0, 1.0)
    angle_dist = torch.acos(torch.abs(dot_prod))

    # Reciprocal product for skew lines logic
    recip_prod = torch.abs(torch.mm(d, m.t()) + torch.mm(m, d.t()))

    # Optimized norm of cross product for distance formula denominator
    cross_norm = torch.sqrt(torch.clamp(1.0 - dot_prod ** 2, min=eps))

    # Handle parallel case where cross_norm approaches zero
    s = torch.sign(dot_prod).clamp(min=-1.0, max=1.0)
    m1 = m.unsqueeze(1)
    m2 = m.unsqueeze(0)
    d1 = d.unsqueeze(1)
    parallel_dist = torch.linalg.norm(torch.cross(d1, m1 - m2 / s.unsqueeze(-1), dim=2), dim=2)

    # Select geometric distance based on line relationship
    geo_dist = torch.where(cross_norm > 1e-5, recip_prod / cross_norm, parallel_dist)

    # Combine components
    combined_dist_sq = (alpha * angle_dist) ** 2 + (beta * geo_dist) ** 2

    return torch.sqrt(torch.clamp(combined_dist_sq, min=eps))
