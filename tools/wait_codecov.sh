#!/usr/bin/env bash
# Wait for the workflows on a commit, then poll Codecov until that commit's
# totals are processed. Usage: wait_codecov.sh <commit-sha>
set -u
cd "$(dirname "$0")/.."
sha="${1:?usage: wait_codecov.sh <commit-sha>}"

bash tools/wait_ci.sh "$sha" || exit 1

for i in $(seq 1 20); do
  line=$(curl -s "https://api.codecov.io/api/v2/github/ai-vnv/repos/Duckietown.jl/commits?page_size=1" |
    python3 -c 'import json,sys
c = json.load(sys.stdin)["results"][0]
t = c["totals"]
print(c["commitid"][:7],
      t["coverage"] if t else "processing",
      t["lines"] if t else "-", t["hits"] if t else "-")')
  echo "$line"
  case "$line" in
    "$sha"*processing*) sleep 60 ;;
    "$sha"*) echo "CODECOV FINAL: $line"; exit 0 ;;
    *) sleep 60 ;;
  esac
done
echo "codecov: timed out"
exit 1
