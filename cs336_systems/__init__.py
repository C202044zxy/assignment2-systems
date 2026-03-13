import importlib.metadata
from .flash_attention import FlashAttention
from .triton_flash_attention import TritonFlashAttention

__version__ = importlib.metadata.version("cs336-systems")

__all__ = [
    "FlashAttention",
    "TritonFlashAttention",
]