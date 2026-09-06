#!/usr/bin/env bash
# What license does the pinned gym-duckietown actually ship? Reads the
# installed package metadata in the ddm-ref conda env (read-only).
set -u
SP=$(ls -d "$HOME"/miniconda3/envs/ddm-ref/lib/python*/site-packages 2>/dev/null | head -1)
echo "site-packages: $SP"
echo "--- duckietown-ish dists:"
ls "$SP" | grep -i "duckietown\|gym_duckietown\|duckietown_world" | head -10
echo "--- license fields in metadata:"
for d in "$SP"/*duckietown*-info "$SP"/*gym*duckietown*-info; do
  [ -e "$d" ] || continue
  echo "== $d"
  grep -i -E "^(License|Classifier: License|Home-page|Name|Version)" "$d/METADATA" "$d/PKG-INFO" 2>/dev/null | head -8
  ls "$d" | grep -i -E "license" || true
done
echo "--- LICENSE files inside the packages:"
find "$SP" -maxdepth 2 -ipath "*duckietown*" -iname "LICENSE*" 2>/dev/null | head -5
