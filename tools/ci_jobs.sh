#!/usr/bin/env bash
# Per-job conclusions for every workflow run on a commit.
# Usage: ci_jobs.sh <sha>
set -u
cd "$(dirname "$0")/.."
sha="${1:?usage: ci_jobs.sh <sha>}"
for id in $(gh run list --commit "$sha" --json databaseId --jq '.[].databaseId'); do
  gh run view "$id" --json workflowName,jobs \
    --jq '.jobs[] | "\(.conclusion // "running")  \(.name)"'
done | sort
