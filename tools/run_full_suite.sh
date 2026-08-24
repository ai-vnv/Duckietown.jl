#!/bin/bash
# Full acceptance run: every gate, complete log, real exit code.
#
# Written as a file rather than an inline heredoc on purpose — an inline
# heredoc gets expanded at write time when it crosses the Windows/WSL shell
# boundary, which silently replaced `$?` with a stale literal and emptied
# JULIA_PYTHONCALL_EXE in an earlier run.
set -uo pipefail

export PATH="$HOME/.juliaup/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ddm-ref

export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE="$(command -v python)"
export PYGLET_HEADLESS=1

LOG="${1:-/tmp/fj_full.log}"
cd "$(dirname "$0")/.."

{
  echo "=== $(date -Is) ==="
  echo "julia   : $(julia --version)"
  echo "python  : $JULIA_PYTHONCALL_EXE  ($(python --version 2>&1))"
  echo "project : $(pwd)"
} | tee "$LOG"

julia --project=. -e 'using Pkg; Pkg.test()' >> "$LOG" 2>&1
status=$?

echo "PKGTEST_EXIT=$status" | tee -a "$LOG"
exit $status
