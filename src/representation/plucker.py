"""Plücker coordinate representation and hybrid distance metric for LORs.

Ported from lines/representations.py and lines/metrics.py.
"""

import torch


def to_canonical_plucker(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Convert LOR endpoints to canonical Plücker coordinates.

    Args:
        p1: Start points of shape (N, 3).
        p2: End points of shape (N, 3).

    Returns:
        Canonical Plücker coordinates of shape (N, 6) as [d, m].
    """
    d = p2 - p1
    eps = 1e-10

    is_nonzero_x = torch.abs(d[:, 0]) > eps
    is_nonzero_y = torch.abs(d[:, 1]) > eps
    first_comp = torch.where(
        is_nonzero_x, d[:, 0], torch.where(is_nonzero_y, d[:, 1], d[:, 2])
    )

    signs = torch.sign(first_comp).view(-1, 1)
    signs[signs == 0] = 1

    d = (d * signs) / torch.linalg.norm(d, dim=1, keepdim=True)
    m = torch.linalg.cross(p1 * signs, d, dim=1)

    return torch.cat([d, m], dim=1)


def compute_hybrid_weighted_distance(
    coords: torch.Tensor, alpha: float = 1.0, beta: float = 1.0
) -> torch.Tensor:
    """Pairwise hybrid angular+geometric distance in Plücker space.

    Args:
        coords: Plücker coordinates of shape (N, 6).
        alpha: Weight for the angular (direction) component.
        beta: Weight for the shortest geometric distance component.
            Set to 1/scanner_radius to match scale of the angular term.

    Returns:
        Symmetric distance matrix of shape (N, N).
    """
    d = coords[:, :3]
    m = coords[:, 3:]
    eps = 1e-10

    dot_prod = torch.mm(d, d.t()).clamp(-1.0, 1.0)
    angle_dist = torch.acos(torch.abs(dot_prod))

    recip_prod = torch.abs(torch.mm(d, m.t()) + torch.mm(m, d.t()))
    cross_norm = torch.sqrt(torch.clamp(1.0 - dot_prod ** 2, min=eps))

    s = torch.sign(dot_prod).clamp(-1.0, 1.0)
    m1 = m.unsqueeze(1)
    m2 = m.unsqueeze(0)
    d1 = d.unsqueeze(1)
    parallel_dist = torch.linalg.norm(
        torch.cross(d1, m1 - m2 / s.unsqueeze(-1), dim=2), dim=2
    )

    geo_dist = torch.where(cross_norm > 1e-5, recip_prod / cross_norm, parallel_dist)

    combined = (alpha * angle_dist) ** 2 + (beta * geo_dist) ** 2
    return torch.sqrt(torch.clamp(combined, min=eps))


def to_canonical_plucker_batched(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Batched version of to_canonical_plucker for F frames simultaneously.

    Args:
        p1: Start points of shape (F, N, 3).
        p2: End points of shape (F, N, 3).

    Returns:
        Canonical Plücker coordinates of shape (F, N, 6) as [d, m].
    """
    d = p2 - p1
    eps = 1e-10

    is_nonzero_x = torch.abs(d[..., 0]) > eps
    is_nonzero_y = torch.abs(d[..., 1]) > eps
    first_comp = torch.where(
        is_nonzero_x, d[..., 0], torch.where(is_nonzero_y, d[..., 1], d[..., 2])
    )

    signs = torch.sign(first_comp).unsqueeze(-1)  # (F, N, 1)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)

    d = (d * signs) / torch.linalg.norm(d, dim=-1, keepdim=True)
    m = torch.linalg.cross(p1 * signs, d, dim=-1)

    return torch.cat([d, m], dim=-1)


def compute_hybrid_weighted_distance_batched(
    coords: torch.Tensor, alpha: float = 1.0, beta: float = 1.0
) -> torch.Tensor:
    """Batched pairwise hybrid distance for F frames in a single pass.

    Uses torch.bmm so all frames share one GPU kernel launch.

    Args:
        coords: Plücker coordinates of shape (F, N, 6).
        alpha: Angular distance weight.
        beta: Geometric distance weight (use 1/scanner_radius).

    Returns:
        Symmetric distance matrices of shape (F, N, N).
    """
    d = coords[..., :3]   # (F, N, 3)
    m = coords[..., 3:]   # (F, N, 3)
    eps = 1e-10
    N = d.shape[1]

    dot_prod = torch.bmm(d, d.transpose(1, 2)).clamp(-1.0, 1.0)   # (F, N, N)
    angle_dist = torch.acos(torch.abs(dot_prod))

    recip_prod = torch.abs(
        torch.bmm(d, m.transpose(1, 2)) + torch.bmm(m, d.transpose(1, 2))
    )
    cross_norm = torch.sqrt(torch.clamp(1.0 - dot_prod ** 2, min=eps))

    s = torch.sign(dot_prod).clamp(-1.0, 1.0)                      # (F, N, N)
    d1 = d.unsqueeze(2).expand(-1, -1, N, -1)                      # (F, N, N, 3)
    m1 = m.unsqueeze(2).expand(-1, -1, N, -1)                      # (F, N, N, 3)
    m2 = m.unsqueeze(1).expand(-1, N, -1, -1)                      # (F, N, N, 3)
    parallel_dist = torch.linalg.norm(
        torch.linalg.cross(d1, m1 - m2 / s.unsqueeze(-1), dim=-1), dim=-1
    )                                                                # (F, N, N)

    geo_dist = torch.where(cross_norm > 1e-5, recip_prod / cross_norm, parallel_dist)
    combined = (alpha * angle_dist) ** 2 + (beta * geo_dist) ** 2
    return torch.sqrt(torch.clamp(combined, min=eps))


def plucker_distance_matrix(
    p1: torch.Tensor, p2: torch.Tensor, alpha: float = 1.0, beta: float = 1.0
) -> torch.Tensor:
    """LOR endpoints → canonical Plücker → hybrid distance matrix.

    Args:
        p1: Start detector coordinates of shape (N, 3).
        p2: End detector coordinates of shape (N, 3).
        alpha: Angular distance weight.
        beta: Geometric distance weight (use 1/scanner_radius to normalise).

    Returns:
        Symmetric distance matrix of shape (N, N).
    """
    coords = to_canonical_plucker(p1, p2)
    return compute_hybrid_weighted_distance(coords, alpha=alpha, beta=beta)
