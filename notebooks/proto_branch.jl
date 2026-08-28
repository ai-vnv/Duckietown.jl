# Does the package admit a map with actual branching?
#
# small_loop is a pure ring: every drivable tile has exactly two drivable
# neighbours, so at route level there is no decision to make and a shortest
# path solver is degenerate. DORA belongs at route level, so this is the
# question that decides whether it has anything to solve here at all.

using DuckietownDecisionModels
using Printf

function neighbours(m)
    tiles = Set(drivable_tiles(m))
    out = Dict{NTuple{2,Int},Int}()
    for t in tiles
        out[t] = count(d -> (t[1] + d[1], t[2] + d[2]) in tiles,
                       ((1, 0), (-1, 0), (0, 1), (0, -1)))
    end
    return out
end

function report(name, tiles)
    m = try
        RoadMap(name, 0.585, parse_map_tiles(tiles, 0.585), StopSignState[])
    catch e
        @printf("%-22s FAILED: %s\n", name,
                first(sprint(showerror, e), 90))
        return
    end
    nb = neighbours(m)
    isempty(nb) && (@printf("%-22s no drivable tiles\n", name); return)
    hist = Dict(k => count(==(k), values(nb)) for k in 0:4)
    @printf("%-22s %2d drivable, neighbours: %s   branching tiles: %d\n",
            name, length(nb),
            join(["$k=>$(hist[k])" for k in 0:4 if hist[k] > 0], " "),
            count(>=(3), values(nb)))
end

# the shipped ring, for reference
report("small_loop", ["curve_left/W" "straight/W" "curve_left/N";
                      "straight/S"   "asphalt"      "straight/N";
                      "curve_left/S" "straight/E" "curve_left/E"])

# two loops sharing a middle column: the shared tiles should be T-junctions
report("double_loop (3way)",
    ["curve_left/W" "straight/W"  "3way_left/W" "straight/W"  "curve_left/N";
     "straight/S"   "asphalt"       "straight/N"  "asphalt"       "straight/N";
     "curve_left/S" "straight/E"  "3way_left/E" "straight/E"  "curve_left/E"])

# a plus shape around a 4way
report("plus (4way)",
    ["asphalt"        "curve_left/W" "straight/W"  "curve_left/N" "asphalt";
     "curve_left/W" "4way"         "asphalt"       "4way"         "curve_left/N";
     "straight/S"   "straight/N"   "asphalt"       "straight/N"   "straight/N";
     "curve_left/S" "4way"         "straight/E"  "4way"         "curve_left/E";
     "asphalt"        "curve_left/S" "straight/E"  "curve_left/E" "asphalt"])

# minimal: one 4way with four stubs
report("cross (4way + stubs)",
    ["asphalt"        "curve_left/W" "asphalt";
     "curve_left/W" "4way"         "curve_left/N";
     "asphalt"        "curve_left/S" "asphalt"])
