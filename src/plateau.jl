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

        "--dt"
        help="Timestep size."
        arg_type=Float64
        default=1e-3

        "--continuation"
        help="Do continuation runs between walker numbers?"
        arg_type=Bool
        default=true

        "--warmup"
        help="Warm up on the first walker number for this many steps. Only applicable if \"--continuation=true\""
        arg_type=Int
        default=10_000
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
* `dt=1e-3`: Timestep size.
* `continuation`: Do continuation runs between walker numbers?
* `warmup=10_000`: Do this many steps on the first walker number before measuring.

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
    dt=1e-3,
    continuation=true,
    warmup=10_000,
)
    dvec_type = initiator ? InitiatorDVec : DVec
    style = parse_style(style)
    num_walkers = parse_n_walkers(num_walkers)
    return plateau(
        ham, num_walkers, style, id, dir, dvec_type, steps, dt, continuation, warmup
    )
end

function plateau(
    ham, num_walkers, style, id, dir, dvec_type, steps, dτ, continuation, warmup
)
    @mpi_root begin
        root = initialize_run(dir, id)
    end

    dv = MPIData(dvec_type(starting_address(ham) => 10; style))

    _, ref = reference(ham)

    s_strat = DoubleLogUpdate(targetwalkers=num_walkers[1])
    post_step = (
        ProjectedEnergy(ham, dv),
        WalkerLoneliness(),
        SignCoherence(ref),
        SignCoherence(copy(localpart(dv)); name=:single_coherence),
    )

    if continuation
        @mpi_root @info "Warming up."
        params = RunTillLastStep(; laststep=warmup, dτ)
        # Set high `maxlength` here as low walker numbers are more likely to overflow. We
        # know we have enough memory allocated for 2 * last(num_walkers)
        lomc!(ham, dv; s_strat, post_step, params, maxlength=2 * last(num_walkers))
    end
    prev_file = "__warmup__"

    for n_target in num_walkers
        @mpi_root @info "Computing targetwalkers=$n_target"
        s_strat = DoubleLogUpdate(targetwalkers=n_target)
        if continuation
            params.step = 0
            params.laststep = steps
        else
            params = RunTillLastStep(; laststep=steps, dτ)
        end
        time = @elapsed df = lomc!(ham, dv; s_strat, post_step, params).df

        @mpi_root begin
            schema = Tables.schema(df)
            style = StochasticStyle(localpart(dv))
            dvec_type = typeof(localpart(dv))
            continued = continuation ? prev_file : nothing
            metadata = (
                ; ham, n_target, steps, time, schema, style, dvec_type, params, continued,
            )
            @info "Done in $time seconds."
            show_metadata(stdout, metadata)
            save_run(root, "$n_target", df, metadata)
            prev_file = "$n_target"
        end
    end
end

function show_metadata(io::IO, metadata)
    for (k, v) in pairs(metadata)
        println(io, rpad(k, 10), " : ", v)
    end
end
