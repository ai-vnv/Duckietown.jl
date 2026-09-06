#!/usr/bin/env bash
# Poll GitHub Actions until every workflow run for the given commit finishes,
# then print one status line. Usage: wait_ci.sh <commit-sha>
set -u
cd "$(dirname "$0")/.."
sha="${1:?usage: wait_ci.sh <commit-sha>}"
for i in $(seq 1 60); do
  line=$(gh run list --commit "$sha" --json name,status,conclusion \
    --jq '[.[] | .name + ":" + .status + ":" + (.conclusion // "?")] | join(" | ")')
  case "$line" in
    ""|*in_progress*|*queued*|*pending*) sleep 30 ;;
    *) echo "CI: $line"; exit 0 ;;
  esac
done
echo "CI: timed out waiting for $sha"
exit 1
