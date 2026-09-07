#!/usr/bin/env python3
"""Extract the measured ULP envelopes per assertion class from a CI
--log-failed dump (the fma-platform failures)."""
import re
import sys
from collections import defaultdict

log = open(sys.argv[1], encoding="utf-8", errors="replace").read().split("\n")
pairs = defaultdict(list)
expr = None
for line in log:
    m = re.search(r"Expression: (.+)$", line)
    if m:
        expr = m.group(1).strip()
        continue
    m = re.search(r"Evaluated: ([0-9]+) <= ([0-9]+)", line)
    if m and expr:
        pairs[expr].append(int(m.group(1)))
        expr = None

for e, vals in sorted(pairs.items(), key=lambda kv: -max(kv[1])):
    print(f"max {max(vals):>12}  n {len(vals):>5}  {e[:90]}")
