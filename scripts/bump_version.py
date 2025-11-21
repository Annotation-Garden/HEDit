#!/usr/bin/env python3
"""Version bumping script for HED-BOT.

This script helps manage semantic versioning with support for:
- major.minor.patch version bumping
- Pre-release labels (alpha, beta, rc)
- Automatic git tagging and GitHub release creation

Usage:
    python scripts/bump_version.py [major|minor|patch] [--prerelease alpha|beta|rc|stable]
    python scripts/bump_version.py --current  # Show current version

Examples:
    python scripts/bump_version.py patch              # 0.3.0-alpha -> 0.3.1-alpha
    python scripts/bump_version.py minor --prerelease beta  # 0.3.0-alpha -> 0.4.0-beta
    python scripts/bump_version.py major --prerelease stable  # 0.3.0-alpha -> 1.0.0
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

    def get_current_version(self) -> tuple[int, int, int, str]:
        """Read the current version from version.py."""
        content = self.version_file.read_text()

        # Extract version string
        version_match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
        if not version_match:
            raise ValueError("Could not find __version__ in version.py")

        version_str = version_match.group(1)

        # Parse version string (e.g., "0.3.0-alpha" or "1.0.0")
        match = re.match(r"(\d+)\.(\d+)\.(\d+)(?:-(\w+))?", version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")

        major, minor, patch, prerelease = match.groups()
        return int(major), int(minor), int(patch), prerelease or "stable"

    def format_version(self, major: int, minor: int, patch: int, prerelease: str) -> str:
        """Format version tuple into version string."""
        version = f"{major}.{minor}.{patch}"
        if prerelease and prerelease != "stable":
            version += f"-{prerelease}"
        return version

    def bump_version(self, bump_type: str, new_prerelease: str = None) -> tuple[str, str]:
        """Bump version and return (old_version, new_version)."""
        major, minor, patch, prerelease = self.get_current_version()
        old_version = self.format_version(major, minor, patch, prerelease)

        # Apply bump type
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
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

        content = f'''"""Version information for HED-BOT."""

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
            flags=re.DOTALL
        )

        self.pyproject_file.write_text(updated_content)
        print(f"✓ Updated {self.pyproject_file.relative_to(self.project_root)}")

    def git_commit_and_tag(self, version: str):
        """Commit version bump and create Git tag."""
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("⚠ Not in a Git repository. Skipping Git operations.")
            return False

        # Check for uncommitted changes (excluding version files)
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        uncommitted_files = [
            f for f in result.stdout.strip().split("\n")
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
            ["git", "add", "src/version.py", "pyproject.toml"],
            cwd=self.project_root,
            check=True
        )

        # Commit
        commit_message = f"Bump version to {version}"
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=self.project_root,
            check=True
        )
        print(f"✓ Committed version bump: {commit_message}")

        # Create tag
        tag_name = f"v{version}"
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Release {version}"],
            cwd=self.project_root,
            check=True
        )
        print(f"✓ Created tag: {tag_name}")

        return True

    def create_github_release(self, version: str):
        """Create GitHub release using gh CLI."""
        # Check if gh CLI is available
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True
        )

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
            text=True
        )
        recent_commits = result.stdout.strip()

        release_notes = f"""Release {version}

## Recent Changes
{recent_commits}

For full changelog, see commit history.
"""

        # Create release
        print(f"\nCreating GitHub release for {tag_name}...")
        result = subprocess.run(
            ["gh", "release", "create", tag_name, "--title", f"Release {version}", "--notes", release_notes],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✓ Created GitHub release: {tag_name}")
            return True
        else:
            print(f"⚠ Failed to create GitHub release: {result.stderr}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bump HED-BOT version and create Git tag/release",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "bump_type",
        nargs="?",
        choices=["major", "minor", "patch"],
        help="Type of version bump"
    )

    parser.add_argument(
        "--prerelease",
        choices=["alpha", "beta", "rc", "stable"],
        help="Set pre-release label (omit for stable release)"
    )

    parser.add_argument(
        "--current",
        action="store_true",
        help="Show current version and exit"
    )

    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Skip Git operations (commit, tag, release)"
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

    # Validate arguments
    if not args.bump_type:
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
                print(f"  2. Push to remote: git push origin feature/your-branch")
                print(f"  3. Push tag: git push origin v{new_version}")
                print(f"  4. Create PR and merge to main")
                print(f"  5. After merge, the GitHub release will be created automatically")

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
