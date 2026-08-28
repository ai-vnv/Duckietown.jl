# FJ3.8: exact NumPy RNG stream reproduction (RNG-C compatibility layer).
#
# Two reference streams, ported bit-for-bit and pinned by
# test/fixtures/fj38_rng.json (numpy 1.20.0):
# - `NumpyMT19937`: `np.random.RandomState(seed)` — MT19937 with the legacy
#   `init_genrand` seeding and the 53-bit `random_sample` double. This is the
#   DuckController stream; it subtypes `Random.AbstractRNG` so it plugs
#   directly into `simulate_decision(m, s, a, rng)` — the canonical model is
#   unchanged, RNG-C is purely a choice of rng argument.
# - `NumpySeedSequence` + `NumpyPCG64`: `gym.utils.seeding.np_random(seed)` =
#   `Generator(PCG64(SeedSequence(seed)))` — the simulator reset/spawn stream,
#   with `np_random_double` / `np_uniform` / `np_integers` (Lemire bounded)
#   mirroring the `Generator` methods the reset path calls.

import Random

# ---------------------------------------------------------------------------
# MT19937 (np.random.RandomState legacy)
# ---------------------------------------------------------------------------

"""
    NumpyMT19937(seed) <: Random.AbstractRNG

`np.random.RandomState(seed)` for integer seeds in `[0, 2^32-1]` (the legacy
`init_genrand` path; larger seeds use `init_by_array`, which no reference
config needs). `rand(rng)` produces the same Float64 stream as
`RandomState.random_sample()`.
"""
mutable struct NumpyMT19937 <: Random.AbstractRNG
    mt::Vector{UInt32}
    mti::Int   # numpy `pos`: number of words already consumed since twist
end

function NumpyMT19937(seed::Integer)
    0 <= seed <= 0xffffffff ||
        throw(ArgumentError("NumpyMT19937 supports seeds in [0, 2^32-1]"))
    mt = Vector{UInt32}(undef, 624)
    mt[1] = UInt32(seed)
    for i in 1:623
        mt[i + 1] = 0x6c078965 * (mt[i] ⊻ (mt[i] >> 30)) + UInt32(i)
    end
    return NumpyMT19937(mt, 624)
end

function _mt_twist!(r::NumpyMT19937)
    mt = r.mt
    @inbounds for k in 1:624
        y = (mt[k] & 0x80000000) | (mt[k == 624 ? 1 : k + 1] & 0x7fffffff)
        src = k <= 227 ? k + 397 : k + 397 - 624
        mt[k] = mt[src] ⊻ (y >> 1) ⊻ ((y & 0x1) == 0x1 ? 0x9908b0df : 0x00000000)
    end
    r.mti = 0
    return nothing
end

"""
    mt_next_uint32!(r) -> UInt32

One tempered MT19937 output word (`mt19937_next`).
"""
function mt_next_uint32!(r::NumpyMT19937)
    r.mti >= 624 && _mt_twist!(r)
    r.mti += 1
    y = @inbounds r.mt[r.mti]
    y ⊻= y >> 11
    y ⊻= (y << 7) & 0x9d2c5680
    y ⊻= (y << 15) & 0xefc60000
    y ⊻= y >> 18
    return y
end

"""
    random_sample(r) -> Float64

`RandomState.random_sample()`: 53-bit double from two tempered words,
`((a >> 5) * 67108864 + (b >> 6)) / 9007199254740992`.
"""
function random_sample(r::NumpyMT19937)
    a = mt_next_uint32!(r) >> 5
    b = mt_next_uint32!(r) >> 6
    return (Float64(a) * 67108864.0 + Float64(b)) / 9007199254740992.0
end

Random.rand(r::NumpyMT19937,
    ::Random.SamplerTrivial{Random.CloseOpen01{Float64}}) = random_sample(r)

Base.copy(r::NumpyMT19937) = NumpyMT19937(copy(r.mt), r.mti)

"""
    mt_state(r) -> (Vector{UInt32}, Int)

The 624-word key and position, comparable with `RandomState.get_state()`.
"""
mt_state(r::NumpyMT19937) = (copy(r.mt), r.mti)

# ---------------------------------------------------------------------------
# SeedSequence (np.random.SeedSequence, pool_size = 4, empty spawn key)
# ---------------------------------------------------------------------------

const _SS_XSHIFT = UInt32(16)
const _SS_INIT_A = 0x43b0d7e5
const _SS_MULT_A = 0x931e8875
const _SS_INIT_B = 0x8b51f9dd
const _SS_MULT_B = 0x58f38ded
const _SS_MIX_L = 0xca01f9dd
const _SS_MIX_R = 0x4973f715

"""
    NumpySeedSequence(entropy)

`np.random.SeedSequence(entropy)` with the default pool size 4 and no spawn
key: the entropy integer is split into little-endian 32-bit words, hashed and
mixed into the pool; `seedseq_generate_state` reproduces `generate_state`.
"""
struct NumpySeedSequence
    pool::Vector{UInt32}
end

_int_to_uint32_words(v::Integer) = v == 0 ? UInt32[0x00000000] : begin
    words = UInt32[]
    x = BigInt(v)
    while x > 0
        push!(words, UInt32(x & 0xffffffff))
        x >>= 32
    end
    words
end

function NumpySeedSequence(entropy::Integer)
    entropy >= 0 || throw(ArgumentError("entropy must be non-negative"))
    words = _int_to_uint32_words(entropy)
    pool = zeros(UInt32, 4)
    hash_const = Ref(_SS_INIT_A)
    hash = value::UInt32 -> begin
        value ⊻= hash_const[]
        hash_const[] *= _SS_MULT_A
        value *= hash_const[]
        value ⊻= value >> _SS_XSHIFT
        value
    end
    # reference `mix`: MIX_MULT_L*x MINUS MIX_MULT_R*y (uint32 wrap), not xor
    mix = (x::UInt32, y::UInt32) -> begin
        result = _SS_MIX_L * x - _SS_MIX_R * y
        result ⊻= result >> _SS_XSHIFT
        result
    end
    for i in 1:4
        pool[i] = hash(i <= length(words) ? words[i] : 0x00000000)
    end
    for i_src in 1:4, i_dst in 1:4
        if i_src != i_dst
            pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))
        end
    end
    for i_src in 5:length(words), i_dst in 1:4
        pool[i_dst] = mix(pool[i_dst], hash(words[i_src]))
    end
    return NumpySeedSequence(pool)
end

"""
    seedseq_generate_state(ss, n) -> Vector{UInt64}

`SeedSequence.generate_state(n, np.uint64)`: 2n hashed uint32 words from the
cycled pool, paired little-endian (low word first).
"""
function seedseq_generate_state(ss::NumpySeedSequence, n::Integer)
    words = Vector{UInt32}(undef, 2n)
    hash_const = _SS_INIT_B
    for i in 1:2n
        data = ss.pool[mod1(i, 4)]
        data ⊻= hash_const
        hash_const *= _SS_MULT_B
        data *= hash_const
        data ⊻= data >> _SS_XSHIFT
        words[i] = data
    end
    return [UInt64(words[2i - 1]) | (UInt64(words[2i]) << 32) for i in 1:n]
end

# ---------------------------------------------------------------------------
# PCG64 (np.random.PCG64, XSL-RR 128/64) + the Generator methods gym uses
# ---------------------------------------------------------------------------

const _PCG_MULT = (UInt128(0x2360ed051fc65da4) << 64) | UInt128(0x4385df649fccf645)

"""
    NumpyPCG64(seed) / NumpyPCG64(ss::NumpySeedSequence)

`np.random.PCG64(SeedSequence(seed))`: seeded from
`generate_state(4, uint64)` as `(hi, lo)` pairs for initstate/initseq
(`pcg64_srandom_r`). `pcg64_next_uint64` / `np_random_double` /
`np_uniform` / `np_integers` mirror the bit generator and the `Generator`
methods called by the simulator reset path.
"""
mutable struct NumpyPCG64 <: Random.AbstractRNG
    state::UInt128
    inc::UInt128
    # pcg64_next32 half-word buffer: next32 returns the LOW half of a fresh
    # uint64 and keeps the HIGH half here, ACROSS calls (it lives in the
    # bit-generator state); the double/uint64 paths bypass it untouched
    has_uint32::Bool
    uinteger::UInt32
end

function NumpyPCG64(ss::NumpySeedSequence)
    v = seedseq_generate_state(ss, 4)
    initstate = (UInt128(v[1]) << 64) | UInt128(v[2])
    initseq = (UInt128(v[3]) << 64) | UInt128(v[4])
    g = NumpyPCG64(UInt128(0), (initseq << 1) | UInt128(1), false, 0x00000000)
    _pcg_step!(g)
    g.state += initstate
    _pcg_step!(g)
    return g
end

NumpyPCG64(seed::Integer) = NumpyPCG64(NumpySeedSequence(seed))

_pcg_step!(g::NumpyPCG64) = (g.state = g.state * _PCG_MULT + g.inc; nothing)

"""
    pcg64_next_uint64(g) -> UInt64

`pcg_setseq_128_xsl_rr_64_random_r`: step, then XSL-RR output
(`rotr64(hi ⊻ lo, state >> 122)`).
"""
function pcg64_next_uint64(g::NumpyPCG64)
    _pcg_step!(g)
    rot = Int(g.state >> 122)
    xored = UInt64(g.state >> 64) ⊻ UInt64(g.state & typemax(UInt64))
    return bitrotate(xored, -rot)
end

"""
    np_random_double(g) -> Float64

`Generator.random()`: `(next_uint64 >> 11) * 2^-53`.
"""
np_random_double(g::NumpyPCG64) =
    Float64(pcg64_next_uint64(g) >> 11) * (1.0 / 9007199254740992.0)

# `rand(g)` IS `Generator.random()`, so any code written against the standard
# `AbstractRNG` interface (e.g. `sample_spawn_pose`) reproduces the reference
# `np_random` draw stream when handed a NumpyPCG64.
Random.rand(g::NumpyPCG64,
    ::Random.SamplerTrivial{Random.CloseOpen01{Float64}}) = np_random_double(g)

Base.copy(g::NumpyPCG64) =
    NumpyPCG64(g.state, g.inc, g.has_uint32, g.uinteger)

"""
    np_uniform(g, low, high) -> Float64

`Generator.uniform(low, high)`: `low + (high - low) * random()`.
"""
np_uniform(g::NumpyPCG64, low::Real, high::Real) =
    Float64(low) + (Float64(high) - Float64(low)) * np_random_double(g)

"""
    np_standard_normal(g) -> Float64

`random_standard_normal` (numpy ziggurat, verbatim incl. the GH-13361
`log(1 - u)` tail form). The fast path (~99.3%) is pure integer × table and
bit-exact; the wedge/tail paths go through libm `log`/`exp`, subject to the
documented ≤1-ULP cross-libm caveat at astronomically rare rejection
boundaries.
"""
function np_standard_normal(g::NumpyPCG64)
    while true
        r = pcg64_next_uint64(g)
        idx = Int(r & 0xff) + 1   # tables are 1-based in Julia
        r >>= 8
        sign = r & 0x1
        rabs = (r >> 1) & 0x000fffffffffffff
        x = Float64(rabs) * WI_DOUBLE[idx]
        if sign == 0x1
            x = -x
        end
        rabs < KI_DOUBLE[idx] && return x
        if idx == 1
            while true
                xx = -ZIGGURAT_NOR_INV_R * log(1.0 - np_random_double(g))
                yy = -log(1.0 - np_random_double(g))
                if yy + yy > xx * xx
                    return ((rabs >> 8) & 0x1) == 0x1 ?
                        -(ZIGGURAT_NOR_R + xx) : ZIGGURAT_NOR_R + xx
                end
            end
        else
            if (FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * np_random_double(g) +
                FI_DOUBLE[idx] < exp(-0.5 * x * x)
                return x
            end
        end
    end
end

"""
    np_normal(g, loc, scale) -> Float64

`Generator.normal(loc, scale)`: `loc + scale * standard_normal()`.
"""
np_normal(g::NumpyPCG64, loc::Real, scale::Real) =
    Float64(loc) + Float64(scale) * np_standard_normal(g)

"""
    pcg64_next_uint32(g) -> UInt32

`pcg64_next32`: LOW half of a fresh uint64 first; the HIGH half is buffered
in the generator state and returned by the next call.
"""
function pcg64_next_uint32(g::NumpyPCG64)
    if g.has_uint32
        g.has_uint32 = false
        return g.uinteger
    end
    next = pcg64_next_uint64(g)
    g.has_uint32 = true
    g.uinteger = UInt32(next >> 32)
    return UInt32(next & 0xffffffff)
end

"""
    np_integers(g, low, high) -> Int

`Generator.integers(low, high)` (default `endpoint=false`, int64 dtype):
`random_bounded_uint64_fill` with `use_masked=false` — for ranges that fit in
32 bits (every reference call site) this is Lemire's rejection on the
BUFFERED 32-bit source (`bounded_lemire_uint32` over `pcg64_next32`); larger
ranges use the 64-bit Lemire.
"""
function np_integers(g::NumpyPCG64, low::Integer, high::Integer)
    high > low || throw(ArgumentError("high must be > low"))
    rng64 = UInt64(high - low - 1)   # inclusive range
    rng64 == 0 && return Int(low)
    if rng64 <= UInt64(typemax(UInt32))
        rng = UInt32(rng64)
        rng == typemax(UInt32) && return Int(low + pcg64_next_uint32(g))
        rng_excl = rng + UInt32(1)
        m = UInt64(pcg64_next_uint32(g)) * UInt64(rng_excl)
        leftover = UInt32(m & 0xffffffff)
        if leftover < rng_excl
            threshold = (typemax(UInt32) - rng) % rng_excl
            while leftover < threshold
                m = UInt64(pcg64_next_uint32(g)) * UInt64(rng_excl)
                leftover = UInt32(m & 0xffffffff)
            end
        end
        return Int(low + Int(m >> 32))
    end
    if rng64 == typemax(UInt64)
        return Int(low + Int128(pcg64_next_uint64(g)))
    end
    rng_excl64 = rng64 + 1
    m = UInt128(pcg64_next_uint64(g)) * UInt128(rng_excl64)
    leftover = UInt64(m & typemax(UInt64))
    if leftover < rng_excl64
        threshold = (typemax(UInt64) - rng64) % rng_excl64
        while leftover < threshold
            m = UInt128(pcg64_next_uint64(g)) * UInt128(rng_excl64)
            leftover = UInt64(m & typemax(UInt64))
        end
    end
    return Int(low + Int(m >> 64))
end
