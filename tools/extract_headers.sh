#!/bin/bash
# List every source file with the first line of its own header comment —
# input for docs/BUILDING.md, so file descriptions come from the files
# themselves rather than from anyone's memory.
cd "$(dirname "$0")/.."
for f in $(find src -name '*.jl' | sort); do
    h=$(head -20 "$f" | grep -m1 '^#' | sed 's/^#\s*//')
    if [ -z "$h" ]; then
        h=$(head -20 "$f" | grep -m1 -v '^"""' | grep -m1 '[A-Za-z]' | head -c 100)
    fi
    printf '%-42s :: %s\n' "$f" "$h"
done
