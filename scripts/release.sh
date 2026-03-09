#!/usr/bin/env bash
# Usage: ./scripts/release.sh 0.2.0
#
# Bumps version in pyproject.toml and __init__.py, then creates a git tag.
# Push the tag to trigger automated PyPI publishing via GitHub Actions.

set -euo pipefail

VERSION="${1:?Usage: $0 <version> (e.g., 0.2.0)}"

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in semver format (e.g., 0.2.0)" >&2
    exit 1
fi

# Check for clean working tree
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: Working tree is not clean. Commit or stash changes first." >&2
    exit 1
fi

echo "Bumping version to $VERSION..."

# Update pyproject.toml
sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# Update __init__.py
sed -i '' "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" src/fangbot/__init__.py

# Verify tests pass
echo "Running tests..."
uv run python -m pytest -v --tb=short

echo "Committing version bump..."
git add pyproject.toml src/fangbot/__init__.py
git commit -m "Bump version to $VERSION"

echo "Creating tag v$VERSION..."
git tag "v$VERSION"

echo ""
echo "Done! To publish:"
echo "  git push origin main"
echo "  git push origin v$VERSION"
echo ""
echo "GitHub Actions will automatically publish to PyPI."
