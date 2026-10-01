#!/usr/bin/env bash
# Builds this branch and commits the npm-pack payload to the lean `streetcat-dist` branch.
# Usage: bash utils/streetcat-dist.sh [--push]. Prints the dist commit sha for streetcat's package.json.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"
[ -z "$(git status --porcelain -- src build package.json)" ] || { echo "commit src/build first" >&2; exit 1; }
src_sha="$(git rev-parse HEAD)"
src_branch="$(git rev-parse --abbrev-ref HEAD)"

npm run build >/dev/null
[ -z "$(git status --porcelain -- build)" ] || { echo "build/ is stale: rebuild and commit it first" >&2; exit 1; }

tmp="$(mktemp -d)"
npm pack --silent --pack-destination "$tmp" >/dev/null
tar -xzf "$tmp"/three-*.tgz -C "$tmp"

dist="$root/../three.js-dist"
if [ ! -d "$dist" ]; then
	if git show-ref --verify --quiet refs/heads/streetcat-dist || git fetch origin streetcat-dist:streetcat-dist 2>/dev/null; then
		git worktree add "$dist" streetcat-dist >/dev/null
	else
		git worktree add --detach "$dist" >/dev/null
		git -C "$dist" checkout --orphan streetcat-dist >/dev/null
	fi
fi

git -C "$dist" rm -rq --ignore-unmatch . >/dev/null 2>&1 || true
find "$dist" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$tmp/package/." "$dist/"
rm -rf "$tmp"

git -C "$dist" add -A
if git -C "$dist" diff --cached --quiet; then
	echo "dist unchanged" >&2
else
	git -C "$dist" commit -qm "dist: ${src_branch}@${src_sha}"
fi
[ "${1:-}" = "--push" ] && git -C "$dist" push -q -u origin streetcat-dist
git -C "$dist" rev-parse HEAD
