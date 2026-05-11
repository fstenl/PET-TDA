import torch

def to_canonical_plucker(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Converts LOR endpoints to standardized Plücker coordinates.

    Args:
        p1 (torch.Tensor): Start points of shape (N, 3).
        p2 (torch.Tensor): End points of shape (N, 3).

    Returns:
        torch.Tensor: Canonical Plücker coordinates of shape (N, 6) as [d, m].
    """
    # Calculate direction and determine unique orientation
    d = p2 - p1
    eps = 1e-10

    # Priority-based sign selection
    is_nonzero_x = torch.abs(d[:, 0]) > eps
    is_nonzero_y = torch.abs(d[:, 1]) > eps
    first_comp = torch.where(is_nonzero_x,d[:, 0],
                             torch.where(is_nonzero_y, d[:, 1], d[:, 2]))

    # Flip lines to ensure the first significant component is always positive
    signs = torch.sign(first_comp).view(-1, 1)
    signs[signs == 0] = 1

    # Normalize d to unit length and calculate moment vector m = p x d
    d = (d * signs) / torch.linalg.norm(d , dim=1, keepdim=True)
    m = torch.linalg.cross(p1 * signs, d, dim=1)

    return torch.cat([d, m], dim=1)