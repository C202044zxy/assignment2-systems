import importlib.metadata
from .flash_attention import FlashAttention
from .triton_flash_attention import TritonFlashAttention
from .ddp import DDP
from .bucket_ddp import BucketDDP

__version__ = importlib.metadata.version("cs336-systems")

__all__ = [
    "FlashAttention",
    "TritonFlashAttention",
    "DDP",
    "BucketDDP",
]