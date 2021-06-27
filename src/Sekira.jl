module Sekira

using Reexport
# Reexport stuff to create a nice environment to work in.
@reexport using DataFrames
@reexport using Rimu
@reexport using Rimu.StatsTools
@reexport using Rimu.RMPI
@reexport using Rimu: RimuIO

@reexport using Statistics, LinearAlgebra

export reference
export summarize, @summary

using ArgParse
using FileTrees
using JLSO
using KrylovKit
using MacroTools
using NamedTupleTools
using Pipe
using Serialization
using SHA

include("references.jl")
include("files.jl")
include("plateau.jl")
include("analysis.jl")

end
