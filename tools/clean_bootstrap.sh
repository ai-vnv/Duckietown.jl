#!/bin/bash
# FJ9.9a — clean-environment bootstrap.
#
# A fresh process is not enough: it still sees the user's depot, so a package
# that happens to be installed globally can satisfy a dependency the project
# never declares. This uses a THROWAWAY depot and no startup file, so anything
# the package silently relies on has to fail here.
#
#   tools/clean_bootstrap.sh [log] [--keep]
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${1:-/tmp/fj99_clean.log}"
DEPOT="$(mktemp -d /tmp/ddm_clean_depot.XXXXXX)"
export PATH="$HOME/.juliaup/bin:$PATH"
export JULIA_DEPOT_PATH="$DEPOT"
export JULIA_PKG_PRECOMPILE_AUTO=1
# Write the structured report somewhere of its own. A clean run is a DIFFERENT
# run, and overwriting the canonical report with it would silently replace the
# numbers the manifest cites.
export DDM_TEST_REPORT="${DDM_TEST_REPORT:-/tmp/fj99_clean_test_report.json}"
unset JULIA_LOAD_PATH JULIA_PROJECT JULIA_PYTHONCALL_EXE 2>/dev/null || true
cd "$REPO"

{
  echo "=== FJ9.9a clean bootstrap $(date -Iseconds) ==="
  echo "depot   : $DEPOT"
  echo "julia   : $(julia --version)"
  echo "--- instantiate ---"
  julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.status()'
  echo "INSTANTIATE_EXIT=$?"
  echo "--- test ---"
  julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
  echo "CLEAN_TEST_EXIT=$?"
  echo "--- core loads with no optional backend ---"
  julia --startup-file=no --project=. -e '
    using DuckietownDecisionModels
    loaded = [string(m.name) for m in keys(Base.loaded_modules)]
    for p in ("PythonCall","CondaPkg","PyCall","MCTS","Makie","CairoMakie")
        println("LOADED_", p, "=", any(==(p), loaded))
    end
    cfg = joinpath(pkgdir(DuckietownDecisionModels), "..", "duckduck",
        "policies", "q_learning", "training_config.yaml")
    if isfile(cfg)
        println("CLEAN_CORE_FINGERPRINT=",
            core_fingerprint(DuckietownMDP(cfg; action_space=:discrete)))
    else
        println("CLEAN_CORE_FINGERPRINT=config_absent")
    end
    println("CORE_ONLY_OK=true")'
  echo "CORE_ONLY_EXIT=$?"
} > "$LOG" 2>&1

status=$?
echo "depot size: $(du -sh "$DEPOT" 2>/dev/null | cut -f1)" >> "$LOG"
if [ "${2:-}" != "--keep" ]; then
  rm -rf "$DEPOT"
  echo "depot removed" >> "$LOG"
fi
echo "CLEAN_BOOTSTRAP_EXIT=$status" | tee -a "$LOG"
exit $status
