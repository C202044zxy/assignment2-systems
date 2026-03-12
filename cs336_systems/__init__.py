import importlib.metadata
from .flash_attention import FlashAttention

__version__ = importlib.metadata.version("cs336-systems")

__all__ = [
    "FlashAttention",
]