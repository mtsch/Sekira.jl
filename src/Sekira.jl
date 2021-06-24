module Sekira

using Reexport
@reexport using DataFrames
@reexport using Rimu
@reexport using Rimu.StatsTools

@reexport using Statistics, LinearAlgebra

export reference

using KrylovKit
using SHA
using Serialization

cache(args...) = joinpath(@__DIR__, "..", "cache", args...)

function get_id(x)
    buff = IOBuffer()
    serialize(buff, x)
    id = bytes2hex(sha256(buff.data))
end

function reference(H)
    Hs = Rimu.Hamiltonians
    id = get_id(H)
    if isfile(cache("$id.bson"))
        @info "Reference found."
        ref = RimuIO.load_dvec(cache("$id.bson"))
        E0 = read(cache("$id.energy"), Float64)
    else
        @info "Computing reference."
        dv = DVec(starting_address(H) => 1.0)
        issymmetric = Hs.LOStructure(H) == Hs.Hermitian() && eltype(H) <: Real
        all_results = eigsolve(H, dv, 1, :SR; issymmetric)
        ref = all_results[2][1]
        E0 = all_results[1][1]

        RimuIO.save_dvec(cache("$id.bson"), ref)
        write(cache("$id.energy"), E0)
    end
    return E0, ref
end

end # module
