# Validation for docs/src/building.md:
#   1. every repository path named in backticks exists;
#   2. every relative markdown link resolves;
#   3. the repository's own documentation audit accepts the file.

using DuckietownDecisionModels

root = normpath(joinpath(@__DIR__, ".."))
doc = read(joinpath(root, "docs", "src", "building.md"), String)

# 1. backticked paths that look like repo files
paths = unique(m.captures[1] for m in eachmatch(
    r"`((?:src|test|tools|docs|artifacts|notebooks|ext)/[A-Za-z0-9_./\-]+)`", doc))
missing_paths = [p for p in paths if !ispath(joinpath(root, p))]
println("backticked repo paths: ", length(paths), "   missing: ",
        length(missing_paths))
foreach(p -> println("  MISSING ", p), missing_paths)

# 2. relative markdown links (BUILDING.md lives in docs/)
links = unique(m.captures[1] for m in eachmatch(r"\]\(([^)#:]+)\)", doc))
rel = [l for l in links if !startswith(l, "http")]
missing_links = [l for l in rel if !ispath(normpath(joinpath(root, "docs", "src", l)))]
println("relative links: ", length(rel), "   dead: ", length(missing_links))
foreach(l -> println("  DEAD ", l), missing_links)

# 3. the package's own audit over the whole repo (stale claims + doc checks)
issues = documentation_audit(root)
mine = [i for i in issues if occursin("BUILDING", string(i))]
println("documentation_audit issues total: ", length(issues),
        "   touching BUILDING.md: ", length(mine))
foreach(i -> println("  ", i), mine)

ok = isempty(missing_paths) && isempty(missing_links) && isempty(mine)
println(ok ? "BUILDING.md: PASS" : "BUILDING.md: FAIL")
exit(ok ? 0 : 1)
