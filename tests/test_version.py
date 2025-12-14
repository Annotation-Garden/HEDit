"""Tests for version module."""

from src.version import __version__, __version_info__, get_version, get_version_info


def test_version_string():
    """Test version string format."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0
    # Should be a semver-like string
    assert "." in __version__


def test_version_info_tuple():
    """Test version info is a tuple."""
    assert isinstance(__version_info__, tuple)
    assert len(__version_info__) >= 3
    # First three should be integers
    assert isinstance(__version_info__[0], int)
    assert isinstance(__version_info__[1], int)
    assert isinstance(__version_info__[2], int)


def test_get_version():
    """Test get_version function."""
    version = get_version()
    assert version == __version__


def test_get_version_info():
    """Test get_version_info function."""
    info = get_version_info()
    assert info == __version_info__
