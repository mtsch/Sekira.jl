using ArgParse
using Rimu.RMPI

function parse_n_walkers(n)
    nums = map(x -> parse(Int, x), split(n, ':'))
    if length(nums) == 1
        return nums[1]:nums[1]
    else
        return Colon()(nums...)
    end
end

function parse_style(s)
    if startswith(s, "int")
        return IsStochasticInteger()
    elseif startswith(s, "semi")
        return IsDynamicSemistochastic()
    elseif startswith(s, "early")
        return IsDynamicSemistochastic(late_projection=false)
    else
        error("Unknown style '", s, "'")
    end
end

function unpack_commandline_args(args)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--num_walkers"
        help="Number of walkers to use."
        required=true

        "--style"
        help="Stochastic style to use."
        required=true

        "--id"
        help="The id of the computation."
        required=true

        "--dir"
        help="Output directory."
        default="."

        "--initiator"
        help="Use initiator?"
        arg_type=Bool
        default=false

        "--steps"
        help="Record this many steps."
        arg_type=Int
        default=60_000

        "--warmup"
        help="Do this many steps on the first walker number before starting measuring."
        arg_type=Int
        default=10_000

        "--dt"
        help="Timestep size."
        arg_type=Float64
        default=1e-3

        "--mpi"
        help="Use mpi?"
        arg_type=Bool
        default=true
    end
    res = Dict{Symbol,Any}()
    for (k, v) in parse_args(args, s)
        res[Symbol(k)] = v
    end
    return res
end

"""
    plateau(ham; kwargs...)
    plateau(ham; ARGS)

Perform FCIQMC for various target walkers and save all results to a directory of files.
Uses continuations to reduce equilibration times.

# Arguments:

* `num_walkers` (required): range of walkers to run the computation with.
* `style` (required): the stochastic style ("int", "semi", or "early").
* `id` (required): the id. A directory with that name will be created.
* `dir="."`: Output directory.
* `initiator=false`: Use initiators?
* `steps=60_000`: Record this many steps.
* `warmup=10_000`: Do this many steps on the first walker number before measuring.
* `dt=1e-3`: Timestep size.
* `mpi=true`: Use MPI?

Arguments can also be passed in an array of strings. Example usage:

```julia
# script.jl
using Sekira
H = HubbardMom1D(BoseFS((0,0,0,7,0,0,0)); u=5)
Sekira.plateau(H, ARGS)
```

```
\$ mpirun julia script.jl --num_walkers=100:100:500 --style="int" --id="test"
```

This will create a directory `test` where output files are saved.
"""
plateau(ham, args) = plateau(ham; unpack_commandline_args(args)...)
function plateau(
    ham; num_walkers, style, id,
    dir=".",
    initiator=false,
    steps=60_000,
    warmup=10_000,
    dt=1e-3,
    mpi=true,
)
    dvec_type = initiator ? InitiatorDVec : DVec
    style = parse_style(style)
    num_walkers = parse_n_walkers(num_walkers)
    return plateau(ham, num_walkers, style, id, dir, dvec_type, steps, warmup, dt, mpi)
end

function plateau(ham, num_walkers, style, id, dir, dvec_type, steps, warmup, dτ, mpi)
    @mpi_root begin
        dir = joinpath(dir, id)
        if isdir(dir)
            @warn "Directory `$dir` exists!"
        end
        mkpath(dir)
        metafile = joinpath(dir, "metadata.csv")
        write(metafile, "filename,hamiltonian,steps,targetwalkers,dt,dvec_type,style,time\n")
    end

    if mpi
        dv = MPIData(dvec_type(starting_address(ham) => 1; style))
    else
        dv = dvec_type(starting_address(ham) => 1; style)
    end

    _, ref = reference(ham)

    s_strat = DoubleLogUpdate(targetwalkers=num_walkers[1])
    post_step = (
        ProjectedEnergy(ham, dv),
        WalkerLoneliness(),
        SignCoherence(ref),
        SignCoherence(copy(localpart(dv)); name=:single_coherence),
    )

    @mpi_root @info "Warming up."
    params = RunTillLastStep(; laststep=warmup, dτ)
    lomc!(ham, dv; s_strat, post_step, params)

    for t in num_walkers
        @mpi_root @info "Computing targetwalkers=$t"
        s_strat = DoubleLogUpdate(targetwalkers=t)
        params.step = 0
        params.laststep = steps
        time = @elapsed df = lomc!(ham, dv; s_strat, post_step, params).df

        @mpi_root begin
            @info "Done in $time seconds."
            filename = joinpath(dir, string(lpad(t, 10, '0'), ".arrow"))
            open(metafile, "a") do f
                write(
                    f,
                    "\"$(basename(filename))\",\"$(ham)\""*
                    ",$(steps),$(t),$(dτ),\"$(dvec_type)\",\"$(style)\",$(time)\n"
                )
            end
            RimuIO.save_df(filename, df)
        end
    end
end
