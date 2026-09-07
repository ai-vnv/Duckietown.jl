# Print the exported lowercase/underscore names (the function surface the
# registry review flagged), one line, for curation.
using Duckietown
ns = sort(string.(names(Duckietown)))
lower = [n for n in ns if islowercase(n[1]) || startswith(n, "_")]
println(length(lower))
println(join(lower, " "))
