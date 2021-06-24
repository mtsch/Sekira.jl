using ArgParse

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

function get_args(args)
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
        help="Steps to equilibriate."
        arg_type=Int
        default=60_000

        "--warmup"
        help="Do this many steps before starting measuring."
        arg_type=Int
        default=10_000

        "--dt"
        help="Timestep size."
        arg_type=Float64
        default=1e-3

        "--mpi"
        arg_type=Bool
        default=true
    end
    values = parse_args(args, s)

    return (
        targets=parse_n_walkers(values["num_walkers"]),
        style=parse_style(values["style"]),
        dvec_type=values["initiator"] ? InitiatorDVec : DVec,
        dτ=values["dt"],
        dir=joinpath(values["dir"], values["id"]),
        mpi=values["mpi"],
        warmup=values["warmup"],
        steps=values["steps"],
    )
end

function plateau(ham, args)
    targets, style, dvec_type, dτ, dir, mpi, warmup, steps = get_args(args)
    if isdir(dir)
        @warn "Directory `$dir` exists!"
    end
    mkpath(dir)
    metafile = joinpath(dir, "metadata.csv")
    write(metafile, "filename,steps,targetwalkers,dt,time\n")

    if mpi
        dv = MPIData(dvec_type(starting_address(ham) => 1; style))
    else
        dv = dvec_type(starting_address(ham) => 1; style)
    end

    _, ref = reference(ham)

    s_strat = DoubleLogUpdate(targetwalkers=targets[1])
    post_step = (
        ProjectedEnergy(ham, dv),
        WalkerLoneliness(),
        SignCoherence(ref),
        SignCoherence(copy(dv); name=:single_coherence)
    )

    @info "Warming up."
    params = RunTillLastStep(; laststep=warmup, dτ)
    lomc!(ham, dv; s_strat, post_step, params)

    for t in targets
        @info "Computing targetwalkers=$t"
        s_strat = DoubleLogUpdate(targetwalkers=t)
        params.step = 0
        params.laststep = steps
        time = @elapsed df = lomc!(ham, dv; s_strat, post_step, params).df

        @info "Done in $time seconds."
        filename = joinpath(dir, string(lpad(t, 10, '0'), ".arrow"))
        open(metafile, "a") do f
            write(f, "\"$(basename(filename))\",$steps,$t,$dτ,$time\n")
        end
        RimuIO.save_df(filename, df)
    end
end
