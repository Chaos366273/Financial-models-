"""
Memory optimizer compatibility layer.
Minimal wrapper to satisfy imports expecting `ppo.utils.memory_optimizer`.
"""

import gc
from typing import Optional

__all__ = [
    "MemoryOptimizerBackup",
]


class MemoryOptimizerBackup:
    """
    Provides minimal hooks for memory cleanup without impacting behavior.
    """

    def __init__(self, enable_cuda_cleanup: bool = True) -> None:
        self.enable_cuda_cleanup = enable_cuda_cleanup

    def cleanup(self) -> None:
        try:
            gc.collect()
            if self.enable_cuda_cleanup:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass


