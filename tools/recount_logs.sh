#!/bin/bash
# Re-summarise every captured suite log with the current parsers.
#
# Exists because a parser bug (a seconds-only elapsed-time pattern) silently
# dropped any test set that ran longer than a minute. Whenever the parsers
# change, previously reported totals must be recomputed rather than trusted.
S="$(dirname "$0")/suite_summary.sh"
P="$(dirname "$0")/suite_summary.py"
for L in "$@"; do
    [ -f "$L" ] || continue
    a=$(bash "$S" "$L" | grep '^assertions' | tr -dc '0-9')
    p=$(python3 "$P" "$L" | grep '^assertions' | tr -dc '0-9')
    t=$(grep -cE '^Test Summary:' "$L")
    m=$(grep -cE '\| +[0-9]+ +[0-9]+ +([0-9]+h)?[0-9]+m[0-9.]+s$' "$L")
    x=$(grep -oE 'PKGTEST_EXIT=[0-9]+' "$L" | tail -1)
    printf '%-18s testsets=%-5s awk=%-8s py=%-8s minute_rows=%-3s %s\n' \
        "$(basename "$L")" "$t" "$a" "$p" "$m" "$x"
done
