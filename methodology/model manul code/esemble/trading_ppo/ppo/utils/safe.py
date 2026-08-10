import numpy as np
import torch

EPS = 1e-12
MAX_ABS = 1e6

def safe_clamp(x, low, high):
    """Clamp numpy arrays or torch tensors safely."""
    if isinstance(x, np.ndarray):
        return np.clip(x, low, high)
    try:
        if torch.is_tensor(x):
            return torch.clamp(x, low, high)
    except Exception:
        pass
    return x

def sanitize_np_array(a, nan_val=0.0, posinf=1e6, neginf=-1e6, max_abs=MAX_ABS):
    """Sanitize numpy array by replacing NaN/Inf values and clamping to reasonable range."""
    a = np.nan_to_num(a, nan=nan_val, posinf=posinf, neginf=neginf)
    # clamp to max_abs
    a = np.clip(a, -max_abs, max_abs)
    return a

def sanitize_tensor(t, nan_val=0.0, posinf=1e6, neginf=-1e6, max_abs=MAX_ABS):
    """Sanitize torch tensor by replacing NaN/Inf values and clamping to reasonable range."""
    # torch.nan_to_num available in recent torch
    t = torch.nan_to_num(t, nan=nan_val, posinf=posinf, neginf=neginf)
    return torch.clamp(t, -max_abs, max_abs)
