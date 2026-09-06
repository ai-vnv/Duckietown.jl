#!/usr/bin/env bash
# Print the zero-hit source lines from the newest .cov file of each given
# src file — precise targets for coverage tests.
set -u
cd "$(dirname "$0")/.."
for f in "$@"; do
  cov=$(ls -t "$f".*.cov 2>/dev/null | head -1)
  [ -z "$cov" ] && { echo "== $f: NO COV FILE"; continue; }
  echo "== $f"
  awk 'match($0, /^ *0 /) { line = substr($0, RSTART + RLENGTH); printf "%5d  %s\n", NR, line }' "$cov" | head -40
done
