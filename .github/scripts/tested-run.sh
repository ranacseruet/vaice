#!/usr/bin/env bash
# Usage: tested-run.sh OWNER/REPO TREE_SHA
# Prints the URL of a successful CI run, from a branch of this repository (never a fork), that tested the git tree
# TREE_SHA, and prints nothing when there is none. ci.yml's `check` job uploads an artifact named
# `tested-tree-<tree>` as its last step, so the artifact exists only when ruff and pytest passed on that tree. A push
# to main whose tree was already tested (by the PR's last run, on refs/pull/N/merge) uses this to skip the macOS
# job (#150). Exits non-zero when the GitHub API fails, and ci.yml then runs the full check.
set -euo pipefail

repo=$1
tree=$2
repo_id=$(gh api "repos/$repo" --jq .id)
runs=$(gh api "repos/$repo/actions/artifacts?name=tested-tree-$tree&per_page=100" --jq \
  ".artifacts[] | select(.expired == false and .workflow_run.repository_id == $repo_id
     and .workflow_run.head_repository_id == $repo_id) | .workflow_run.id")
for run in $runs; do
  read -r path conclusion url < <(gh api "repos/$repo/actions/runs/$run" --jq '"\(.path) \(.conclusion) \(.html_url)"')
  if [ "$path" = ".github/workflows/ci.yml" ] && [ "$conclusion" = success ]; then
    echo "$url"
    exit 0
  fi
done
