"""Version information for HED-BOT."""

__version__ = "0.5.1-dev"
__version_info__ = (0, 5, 1, "dev")


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> tuple:
    """Get the version info tuple (major, minor, patch, prerelease)."""
    return __version_info__
