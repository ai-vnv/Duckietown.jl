#!/usr/bin/env bash
# Tracked media files by size (largest first) plus the total — the input to
# the v0.1.0 media diet.
set -u
cd "$(dirname "$0")/.."
git ls-files -z | while IFS= read -r -d '' f; do
  case "${f,,}" in
    *.gif|*.mp4|*.png|*.pdf|*.jpg|*.jpeg)
      printf '%s %s\n' "$(stat -c%s "$f")" "$f" ;;
  esac
done | sort -rn | awk '{ t += $1; printf "%8.2f MB  %s\n", $1/1048576, $2 }
                       END { printf "TOTAL %.1f MB in %d files\n", t/1048576, NR }'
