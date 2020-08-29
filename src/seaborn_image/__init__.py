try:
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

from ._context import *
from ._general import *
from ._filters import *
from ._grid import *
from .utils import *


set_context()
set_scalebar()
set_image()
set_save_context(dpi=300)

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
