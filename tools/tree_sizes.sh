#!/bin/bash
# Committed-tree size per top-level directory (what a Pkg.add tarball ships).
cd "$(dirname "$0")/.."
total=0
for d in src ext test docs tools artifacts notebooks .github; do
    s=0
    while IFS= read -r f; do
        [ -f "$f" ] && s=$((s + $(stat -c%s "$f")))
    done < <(git ls-files "$d")
    printf '%-12s %8.2f MB\n' "$d" "$(echo "$s / 1048576" | bc -l)"
    total=$((total + s))
done
printf '%-12s %8.2f MB\n' "TOTAL" "$(echo "$total / 1048576" | bc -l)"
echo "--- 10 file ter-commit terbesar ---"
git ls-files | while IFS= read -r f; do
    [ -f "$f" ] && printf '%d\t%s\n' "$(stat -c%s "$f")" "$f"
done | sort -rn | head -10 | awk '{printf "%6.2f MB  %s\n", $1/1048576, $2}'
