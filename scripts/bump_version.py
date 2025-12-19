#!/usr/bin/env python3
"""Version bumping script for HEDit.

This script helps manage semantic versioning with support for:
- major.minor.patch version bumping
- Pre-release labels (alpha, beta, rc, dev) in PEP 440 format
- Changing prerelease label without version bump
- Automatic git tagging and GitHub release creation

PEP 440 Version Format:
    - dev:   0.6.3.dev0  (development pre-release)
    - alpha: 0.6.3a0     (alpha pre-release)
    - beta:  0.6.3b0     (beta pre-release)
    - rc:    0.6.3rc0    (release candidate)
    - stable: 0.6.3      (final release)

Usage:
    python scripts/bump_version.py [major|minor|patch] [--prerelease alpha|beta|rc|dev|stable]
    python scripts/bump_version.py --prerelease alpha  # Change only prerelease label
    python scripts/bump_version.py --current  # Show current version

Examples:
    python scripts/bump_version.py patch                      # 0.3.0a0 -> 0.3.1a0
    python scripts/bump_version.py minor --prerelease beta    # 0.3.0a0 -> 0.4.0b0
    python scripts/bump_version.py major --prerelease stable  # 0.3.0a0 -> 1.0.0
    python scripts/bump_version.py --prerelease alpha         # 0.4.5.dev0 -> 0.4.5a0
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


class VersionBumper:
    """Handle version bumping and Git operations."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.version_file = project_root / "src" / "version.py"
        self.pyproject_file = project_root / "pyproject.toml"

    # PEP 440 prerelease suffix mapping
    PRERELEASE_SUFFIXES = {
        "dev": ".dev0",
        "alpha": "a0",
        "beta": "b0",
        "rc": "rc0",
        "stable": "",
    }

    # Reverse mapping for parsing
    PRERELEASE_PATTERNS = {
        r"\.dev\d*": "dev",
        r"a\d*": "alpha",
        r"b\d*": "beta",
        r"rc\d*": "rc",
    }

    def get_current_version(self) -> tuple[int, int, int, str]:
        """Read the current version from version.py."""
        content = self.version_file.read_text()

        # Extract version string
        version_match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
        if not version_match:
            raise ValueError("Could not find __version__ in version.py")

        version_str = version_match.group(1)

        # Parse PEP 440 version string
        # Handles: 0.6.3, 0.6.3a0, 0.6.3b0, 0.6.3rc0, 0.6.3.dev0
        # Also handles legacy format: 0.6.3-alpha, 0.6.3-dev
        match = re.match(r"(\d+)\.(\d+)\.(\d+)(?:[-.]?(.+))?", version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")

        major, minor, patch, suffix = match.groups()

        # Determine prerelease type from suffix
        prerelease = "stable"
        if suffix:
            # Check for PEP 440 suffixes
            for pattern, label in self.PRERELEASE_PATTERNS.items():
                if re.match(pattern, suffix):
                    prerelease = label
                    break
            else:
                # Legacy format (e.g., -alpha, -dev)
                suffix_lower = suffix.lower()
                if suffix_lower in self.PRERELEASE_SUFFIXES:
                    prerelease = suffix_lower

        return int(major), int(minor), int(patch), prerelease

    def format_version(self, major: int, minor: int, patch: int, prerelease: str) -> str:
        """Format version tuple into PEP 440 compliant version string."""
        version = f"{major}.{minor}.{patch}"
        suffix = self.PRERELEASE_SUFFIXES.get(prerelease, "")
        return version + suffix

    def bump_version(self, bump_type: str | None, new_prerelease: str = None) -> tuple[str, str]:
        """Bump version and return (old_version, new_version)."""
        major, minor, patch, prerelease = self.get_current_version()
        old_version = self.format_version(major, minor, patch, prerelease)

        # Apply bump type (if specified)
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        elif bump_type is not None:
            raise ValueError(f"Invalid bump type: {bump_type}")

        # Apply prerelease change if specified
        if new_prerelease is not None:
            prerelease = new_prerelease

        new_version = self.format_version(major, minor, patch, prerelease)

        # Write new version to file
        self._write_version_file(major, minor, patch, prerelease)

        return old_version, new_version

    def _write_version_file(self, major: int, minor: int, patch: int, prerelease: str):
        """Write new version to version.py."""
        version_str = self.format_version(major, minor, patch, prerelease)

        # Build version info tuple
        if prerelease and prerelease != "stable":
            version_info = f'({major}, {minor}, {patch}, "{prerelease}")'
        else:
            version_info = f"({major}, {minor}, {patch})"

        content = f'''"""Version information for HEDit."""

__version__ = "{version_str}"
__version_info__ = {version_info}


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> tuple:
    """Get the version info tuple (major, minor, patch, prerelease)."""
    return __version_info__
'''

        self.version_file.write_text(content)
        print(f"✓ Updated {self.version_file.relative_to(self.project_root)}")

        # Also update pyproject.toml
        self._update_pyproject_toml(version_str)

    def _update_pyproject_toml(self, version: str):
        """Update version in pyproject.toml."""
        import re

        content = self.pyproject_file.read_text()

        # Update version line in [project] section only
        # Use a more specific pattern that captures the [project] section context
        updated_content = re.sub(
            r'(\[project\][^\[]*?version\s*=\s*)"[^"]+"',
            f'\\1"{version}"',
            content,
            flags=re.DOTALL,
        )

        self.pyproject_file.write_text(updated_content)
        print(f"✓ Updated {self.pyproject_file.relative_to(self.project_root)}")

    def git_commit_and_tag(self, version: str):
        """Commit version bump and create Git tag."""
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"], cwd=self.project_root, capture_output=True, text=True
        )

        if result.returncode != 0:
            print("⚠ Not in a Git repository. Skipping Git operations.")
            return False

        # Check for uncommitted changes (excluding version files)
        result = subprocess.run(
            ["git", "diff", "--name-only"], cwd=self.project_root, capture_output=True, text=True
        )

        uncommitted_files = [
            f
            for f in result.stdout.strip().split("\n")
            if f and not f.startswith("src/version.py") and not f.startswith("pyproject.toml")
        ]

        if uncommitted_files:
            print(f"⚠ Warning: Uncommitted changes detected in: {', '.join(uncommitted_files)}")
            response = input("Continue with version bump? (y/N): ")
            if response.lower() != "y":
                print("Aborted.")
                return False

        # Stage version files
        subprocess.run(
            ["git", "add", "src/version.py", "pyproject.toml"], cwd=self.project_root, check=True
        )

        # Commit
        commit_message = f"Bump version to {version}"
        subprocess.run(["git", "commit", "-m", commit_message], cwd=self.project_root, check=True)
        print(f"✓ Committed version bump: {commit_message}")

        # Create tag
        tag_name = f"v{version}"
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Release {version}"],
            cwd=self.project_root,
            check=True,
        )
        print(f"✓ Created tag: {tag_name}")

        return True

    def create_github_release(self, version: str):
        """Create GitHub release using gh CLI."""
        # Check if gh CLI is available
        result = subprocess.run(["gh", "--version"], capture_output=True, text=True)

        if result.returncode != 0:
            print("⚠ GitHub CLI (gh) not found. Install it to create releases automatically.")
            print("  See: https://cli.github.com/")
            return False

        tag_name = f"v{version}"

        # Generate release notes
        result = subprocess.run(
            ["git", "log", "--oneline", "--no-decorate", "-10"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        recent_commits = result.stdout.strip()

        release_notes = f"""Release {version}

## Recent Changes
{recent_commits}

For full changelog, see commit history.
"""

        # Build release command
        release_cmd = [
            "gh",
            "release",
            "create",
            tag_name,
            "--title",
            f"Release {version}",
            "--notes",
            release_notes,
        ]

        # Add --prerelease flag for alpha/beta/rc/dev versions
        if any(label in version.lower() for label in ["alpha", "beta", "rc", "dev"]):
            release_cmd.append("--prerelease")
            print("  (Marking as pre-release since version contains alpha/beta/rc/dev)")

        # Create release
        print(f"\nCreating GitHub release for {tag_name}...")
        result = subprocess.run(release_cmd, cwd=self.project_root, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Created GitHub release: {tag_name}")
            return True
        else:
            print(f"⚠ Failed to create GitHub release: {result.stderr}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bump HEDit version and create Git tag/release",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "bump_type", nargs="?", choices=["major", "minor", "patch"], help="Type of version bump"
    )

    parser.add_argument(
        "--prerelease",
        choices=["alpha", "beta", "rc", "dev", "stable"],
        help="Set pre-release label (dev for develop branch, omit for stable release)",
    )

    parser.add_argument("--current", action="store_true", help="Show current version and exit")

    parser.add_argument(
        "--no-git", action="store_true", help="Skip Git operations (commit, tag, release)"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    bumper = VersionBumper(project_root)

    # Show current version
    if args.current:
        major, minor, patch, prerelease = bumper.get_current_version()
        version = bumper.format_version(major, minor, patch, prerelease)
        print(f"Current version: {version}")
        return 0

    # Validate arguments - need either bump_type or prerelease
    if not args.bump_type and not args.prerelease:
        parser.print_help()
        return 1

    # Perform version bump
    try:
        old_version, new_version = bumper.bump_version(args.bump_type, args.prerelease)
        print(f"\n✓ Version bumped: {old_version} -> {new_version}\n")

        if not args.no_git:
            # Git operations
            if bumper.git_commit_and_tag(new_version):
                print("\nNext steps:")
                print(f"  1. Review the changes: git show v{new_version}")
                print("  2. Push to remote: git push origin feature/your-branch")
                print(f"  3. Push tag: git push origin v{new_version}")
                print("  4. Create PR and merge to main")
                print("  5. After merge, the GitHub release will be created automatically")

                # Optionally create GitHub release
                response = input("\nCreate GitHub release now? (y/N): ")
                if response.lower() == "y":
                    bumper.create_github_release(new_version)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
