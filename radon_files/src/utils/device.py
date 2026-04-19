import torch


def get_device():
    """Detect the best available torch compute device.

    Returns:
        str: 'cuda' if a CUDA device is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
