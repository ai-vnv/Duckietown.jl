#!/usr/bin/env bash
# One-shot media migration to ai-vnv/Duckietown-artifacts (registry review:
# 78 MB of media in a 99 MB registered tree). Copies every tracked media file
# to the companion repo PRESERVING PATHS, pushes it, then git-rm's them here.
# Reference updates in text files are a separate, reviewed step.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="$HOME/aivnv/Duckietown-artifacts"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/ai-vnv/Duckietown-artifacts "$DEST"
fi

mapfile -d '' FILES < <(git ls-files -z | grep -zEi '\.(gif|mp4|png|pdf|jpe?g)$')
echo "moving ${#FILES[@]} files"

for f in "${FILES[@]}"; do
  mkdir -p "$DEST/$(dirname "$f")"
  cp "$f" "$DEST/$f"
done

cat > "$DEST/README.md" << 'MDEOF'
# Duckietown-artifacts

Recorded experiment media for
[ai-vnv/Duckietown.jl](https://github.com/ai-vnv/Duckietown.jl): rendered
laps, animations, per-decision diagnostics, search-tree renders and
publication figures, at the same paths they occupied in the package
repository before its v0.1.0 media diet.

These files are OUTPUTS of recorded experiments. The evidence the package's
test suite actually reads (fixtures, decision logs, structured reports,
exported actor weights) stays in the package repository; this one exists so
that `Pkg.add` does not download 73 MB of media.

Rendered Duckietown environments courtesy of the
[Duckietown Project](https://www.duckietown.org).
MDEOF

cd "$DEST"
git add -A
git commit -m "Import the recorded media from ai-vnv/Duckietown.jl (pre-v0.1.0 tree, paths preserved)"
git push origin HEAD
cd - > /dev/null

git rm -q "${FILES[@]}"
echo "REMOVED_FROM_PACKAGE=${#FILES[@]}"
git status --short | head -5
