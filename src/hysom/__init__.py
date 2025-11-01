from hysom.hysom import HSOM
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hysom")
except PackageNotFoundError:
    __version__ = "unknown"