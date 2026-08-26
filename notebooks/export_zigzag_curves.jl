# Export the Julia-side zigzag tile curves for a parity check against the
# reference simulator. `parse_map_tiles` was FJ3.1-validated on small_loop,
# which never uses `curve_right` — zigzag uses it six times, so the curve
# templates' rotation indexing is unvalidated surface here.

using DuckietownDecisionModels, JSON3

zigzag_tiles() = [
    "asphalt" "asphalt"      "asphalt"       "asphalt"       "asphalt"       "asphalt"    "asphalt"       "asphalt"      "asphalt"
    "asphalt" "curve_left/W" "curve_left/N"  "asphalt"       "curve_left/W"  "straight/W" "straight/W"    "curve_left/N" "asphalt"
    "asphalt" "straight/S"   "curve_right/W" "straight/W"    "curve_right/S" "asphalt"    "curve_right/N" "curve_left/E" "asphalt"
    "asphalt" "straight/S"   "asphalt"       "asphalt"       "asphalt"       "asphalt"    "straight/N"    "asphalt"      "asphalt"
    "asphalt" "straight/S"   "asphalt"       "asphalt"       "curve_right/N" "straight/E" "curve_left/E"  "asphalt"      "asphalt"
    "asphalt" "straight/S"   "asphalt"       "curve_right/N" "curve_left/E"  "asphalt"    "asphalt"       "asphalt"      "asphalt"
    "asphalt" "straight/S"   "asphalt"       "straight/N"    "asphalt"       "asphalt"    "asphalt"       "asphalt"      "asphalt"
    "asphalt" "curve_left/S" "straight/E"    "curve_left/E"  "asphalt"       "asphalt"    "asphalt"       "asphalt"      "asphalt"
    "asphalt" "asphalt"      "asphalt"       "asphalt"       "asphalt"       "asphalt"    "asphalt"       "asphalt"      "asphalt"
]

grid = parse_map_tiles(zigzag_tiles(), 0.585)
h, w = size(grid)

tiles = []
for j in 1:h, i in 1:w
    isassigned(grid, j, i) || continue
    t = grid[j, i]
    t.drivable || continue
    push!(tiles, Dict(
        "i" => i - 1, "j" => j - 1,
        "kind" => string(t.kind), "angle_deg" => t.angle_deg,
        # control points of every lane Bezier, world frame
        "curves" => [[collect(p) for p in c] for c in t.curves]))
end

out = joinpath(@__DIR__, "zigzag_curves_julia.json")
open(io -> JSON3.write(io, Dict("tile_size" => 0.585, "tiles" => tiles)), out, "w")
println("wrote ", basename(out), " with ", length(tiles), " drivable tiles")
