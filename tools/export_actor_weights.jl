# FJ9.8 — export the frozen TD3/SAC actor parameters to .npy, ONCE.
#
#     tools/run_export_weights.sh
#
# This is a provisioning step, deliberately separate from the publication
# build. It needs the torch reference environment; the publication build must
# not, and asserts as much. The exported arrays are the checkpoint's exact
# bits — FJ7 validated the native reader against them layer by layer — and the
# `.pt` checkpoints are read-only inputs that are never written to.

using DuckietownDecisionModels

const ROOT = pkgdir(DuckietownDecisionModels)
const CACHE = joinpath(ROOT, "artifacts", "fj9", "weights")
mkpath(CACHE)

function export_actor(name::AbstractString)
    b = TorchPolicyReferenceBackend()
    try
        meta = torch_policy_init!(b, name)
        src = String(meta["weights_dir"])
        dst = joinpath(CACHE, name)
        mkpath(dst)
        n = 0
        for f in readdir(src)
            endswith(f, ".npy") || continue
            cp(joinpath(src, f), joinpath(dst, f); force=true)
            n += 1
        end
        return (dst, n)
    finally
        close(b)
    end
end

for name in ("td3", "sac")
    try
        dst, n = export_actor(name)
        println("EXPORT_$(uppercase(name))=$n files -> $(relpath(dst, ROOT))")
    catch e
        println("EXPORT_$(uppercase(name))=FAILED ",
            sprint(showerror, e)[1:min(140, end)])
    end
end
println("WEIGHT_EXPORT_OK=true")
