"""
# imslice package
"""
from .version import version as __version__  # noqa: F401
from .version import version_tuple as __version_info__  # noqa: F401

from .camera import Camera
from .scene import Scene

__all__ = ["Camera", "Scene"]
