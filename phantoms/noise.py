import torch


def add_gaussian_noise(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Adds zero-mean Gaussian noise to an image.

    Args:
        image (torch.Tensor): Input phantom image (any shape).
        sigma (float): Standard deviation of the noise.

    Returns:
        torch.Tensor: Noisy image (clamped to >= 0).
    """
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + noise, min=0.0)


def add_poisson_noise(image: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Adds Poisson noise to an image.

    The image intensities are treated as rate parameters (scaled by ``scale``).
    Higher ``scale`` → more photon counts → lower relative noise.

    Args:
        image (torch.Tensor): Input phantom image (non-negative).
        scale (float): Scaling factor applied before Poisson sampling.
            Acts as a photon-count multiplier.

    Returns:
        torch.Tensor: Noisy image rescaled back to the original intensity range.
    """
    return torch.poisson(image * scale) / scale


def add_salt_and_pepper_noise(image: torch.Tensor, probability: float) -> torch.Tensor:
    """Adds salt-and-pepper (impulse) noise to an image.

    Args:
        image (torch.Tensor): Input phantom image.
        probability (float): Fraction of voxels affected (0–1).

    Returns:
        torch.Tensor: Image with random voxels set to 0 or max intensity.
    """
    noisy = image.clone()
    rand = torch.rand_like(image)
    max_val = image.max()

    # Salt
    noisy[rand < probability / 2] = max_val
    # Pepper 
    noisy[(rand >= probability / 2) & (rand < probability)] = 0.0

    return noisy


# Registry: noise_type -> (function, kwarg_name, default_level)
_NOISE_TYPES = {
    'gaussian':         (add_gaussian_noise,          'sigma',       0.1),
    'poisson':          (add_poisson_noise,            'scale',       10.0),
    'salt_and_pepper':  (add_salt_and_pepper_noise,    'probability', 0.05),
}


def apply_noise(image: torch.Tensor, noise_type: str, level: float | None = None) -> torch.Tensor:
    """Apply a single noise type at a given level to one image.

    Args:
        image (torch.Tensor): Input phantom image.
        noise_type (str): One of 'gaussian', 'poisson', 'salt_and_pepper'.
        level (float, optional): Noise intensity parameter. If None the
            built-in default for the chosen noise type is used:
            gaussian=0.1, poisson=10.0, salt_and_pepper=0.05.

    Returns:
        torch.Tensor: Noisy image.
    """
    if noise_type not in _NOISE_TYPES:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            f"Choose from {list(_NOISE_TYPES.keys())}."
        )
    func, kwarg, default = _NOISE_TYPES[noise_type]
    if level is None:
        level = default
    return func(image, **{kwarg: level})


def apply_increasing_noise(frames: list[torch.Tensor], noise_type: str, min_level: float | None = None, max_level: float | None = None) -> list[torch.Tensor]:
    """Apply noise with linearly increasing intensity across a frame sequence.

    The first frame receives ``min_level`` and the last receives ``max_level``.
    If there is only one frame it receives ``max_level``.
    When levels are omitted the defaults are 0 for ``min_level`` and the
    noise type's built-in default for ``max_level``.

    Args:
        frames (list[torch.Tensor]): Ordered list of phantom frames.
        noise_type (str): One of 'gaussian', 'poisson', 'salt_and_pepper'.
        min_level (float, optional): Noise level for the first frame.
        max_level (float, optional): Noise level for the last frame.

    Returns:
        list[torch.Tensor]: New list of frames with increasing noise applied.
    """
    if noise_type not in _NOISE_TYPES:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            f"Choose from {list(_NOISE_TYPES.keys())}."
        )
    _, _, default = _NOISE_TYPES[noise_type]
    if min_level is None:
        min_level = 0.0
    if max_level is None:
        max_level = default

    n = len(frames)
    noisy_frames = []

    for i, frame in enumerate(frames):
        t = i / max(n - 1, 1)          # 0 → 1
        level = min_level + t * (max_level - min_level)
        noisy_frames.append(apply_noise(frame, noise_type, level))

    return noisy_frames
