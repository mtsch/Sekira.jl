module Sekira

using Reexport
@reexport using DataFrames
@reexport using Rimu
@reexport using Rimu.StatsTools
@reexport using Rimu.RMPI

@reexport using Statistics, LinearAlgebra

export reference

using KrylovKit
using SHA
using Serialization

cache(args...) = joinpath(@__DIR__, "..", "cache", args...)

include("plateau.jl")

"""
    get_id(x)

Get unique id of arbitrary struct `x`. Works by serializing the struct and computing the
SHA256 checksum of the serialization.
"""
function get_id(x)
    buff = IOBuffer()
    serialize(buff, x)
    id = bytes2hex(sha256(buff.data))
end

"""
    reference(H; force_recompute=false) -> (E0, reference_vector)

Get the exact ground-state energy a wave function of Hamiltonian `H`. Values are cached to
`Sekira/cache`.
"""
function reference(H; force_recompute=false)
    !isdir(cache()) && mkpath(cache())

    Hs = Rimu.Hamiltonians
    id = get_id(H)
    if !force_recompute && isfile(cache("$id.bson"))
        @mpi_root @info "Reference found."
        ref = RimuIO.load_dvec(cache("$id.bson"))
        E0 = read(cache("$id.energy"), Float64)
    else
        @mpu_root @info "Computing reference."
        dv = DVec(starting_address(H) => 1.0)
        ishermitian = Hs.LOStructure(H) == Hs.Hermitian()
        issymmetric = ishermitian && eltype(H) <: Real
        vals, vecs, info = eigsolve(H, dv, 1, :SR; issymmetric, ishermitian)
        ref = vecs[1]
        E0 = vals[1]
        @assert info.converged ≥ 1

        RimuIO.save_dvec(cache("$id.bson"), ref)
        write(cache("$id.energy"), E0)
    end
    return E0, ref
end

end # module
