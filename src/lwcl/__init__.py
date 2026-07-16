"""LWCL reconstructed research framework."""

from .config import load_config

__all__ = ["LWCLModel", "load_config"]


def __getattr__(name: str):
    if name == "LWCLModel":
        from .models.lwcl import LWCLModel

        return LWCLModel
    raise AttributeError(f"module 'lwcl' has no attribute {name!r}")
