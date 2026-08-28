#!/usr/bin/env python3
"""Local dry-run of the VnV badge verification (same logic as vnv.yml).

Maps each spec requirement's metadata.testset_prefixes onto the committed
release-grade suite report and prints per-requirement verdicts.
"""
import json
import pathlib
import sys

import yaml

root = pathlib.Path(__file__).resolve().parent.parent
spec = yaml.safe_load((root / ".vnvspec/spec.yaml").read_text())
report = json.loads((root / "artifacts/fj9/test_report.json").read_text())
rows = report["testsets_detail"]

failed = 0
for r in spec["requirements"]:
    prefixes = r["metadata"]["testset_prefixes"]
    hits = [x for x in rows if any(x["name"].startswith(p) for p in prefixes)]
    ok = bool(hits) and all(x["fails"] == 0 and x["errors"] == 0 for x in hits)
    print(f"{r['id']}: {len(hits)} testsets, {'PASS' if ok else 'FAIL'}")
    failed += 0 if ok else 1

n = len(spec["requirements"])
print(f"badge message: {n - failed}/{n} pass")
sys.exit(1 if failed else 0)
