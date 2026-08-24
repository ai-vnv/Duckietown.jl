#!/usr/bin/env python3
"""Independent cross-check of tools/suite_summary.sh.

Parses a `Pkg.test` log and sums the Pass column of every top-level testset.

Since FJ9.9b the suite runs under one parent testset, so the log has a single
`Test Summary:` header followed by an unindented ROOT row and one indented row
per top-level testset. The root is the sum of its children and is skipped;
counting it would double every total.

This is now a SANITY CHECK. `artifacts/fj9/test_report.json`, written by
test/reporter.jl from the Test result tree, is authoritative. Two independent
parsers are kept because agreement between three implementations is evidence,
and a silent regression in the reporter would otherwise be invisible.
"""
import re
import sys

# The elapsed column may read "3.5s", "3m17.5s" or "1h2m3s" — and "-0.5s"
# when the system clock steps backwards mid-run. Accepting only positive
# seconds silently drops test sets from the total.
# A PASSING row is "| Pass Total Time"; a FAILING one is
# "| Pass Fail Total Time" (or with Error/Broken columns). Matching only
# the two-number form drops failing test sets from the count entirely,
# which is the worst moment to lose them.
ROW = re.compile(r"\|((?:\s+\d+){2,5})\s+-?(?:\d+h)?(?:\d+m)?[\d.]+s\s*$")

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fj_final.log"
with open(path, encoding="utf-8", errors="replace") as fh:
    lines = fh.read().splitlines()

rows, total, mismatched, headers = [], 0, [], 0
for i, line in enumerate(lines):
    if not line.startswith("Test Summary:"):
        continue
    headers += 1
    for j in range(i + 1, len(lines)):
        row = lines[j]
        m = ROW.search(row)
        if not m:
            break
        # the unindented row is the parent and is the sum of the rest
        if not row.startswith(" "):
            continue
        nums = [int(x) for x in m.group(1).split()]
        passed, count = nums[0], nums[-1]
        rows.append((row[: m.start()].strip(), passed, count))
        total += passed
        if passed != count:
            mismatched.append(row)

print(f"top-level testsets : {len(rows)}")
print(f"assertions passed  : {total}")
print(f"rows unparsed or pass != total : {len(mismatched)}")
for row in mismatched:
    print("   ", row)
print(f"summary headers    : {headers}")
