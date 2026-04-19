import array_api_compat.torch as xp
import torch


def get_device():
    """Detect the best available compute device based on the array backend:
      - numpy  → 'cpu'
      - cupy   → cupy cuda device
      - torch  → 'cuda' if available, else 'cpu'
    """
    if "numpy" in xp.__name__:
        return "cpu"
    elif "cupy" in xp.__name__:
        import cupy
        return cupy.cuda.Device(0)
    elif "torch" in xp.__name__:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cpu"
