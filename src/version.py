"""Version information for HEDit."""

__version__ = "0.6.7a1"
__version_info__ = (0, 6, 7, "alpha")


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> tuple:
    """Get the version info tuple (major, minor, patch, prerelease)."""
    return __version_info__
