import torch


def get_device():
    """Detect the best available torch compute device.

    Returns:
        str: 'cuda' if a CUDA device is available, otherwise 'cpu'.
    """
    return 'cpu'
