"""
    initialize_run(dir, id)

Create directory structure for a run. The directory created will live in `dir/id`.
"""
function initialize_run(dir, id)
    root = joinpath(dir, id)
    if isdir(root)
        @warn "'$root' already exists. Returning existing structure."
    else
        mkpath(root)
        mkdir(joinpath(root, "raw"))     # Raw output DataFrames
        mkdir(joinpath(root, "meta"))    # Metadata
        mkdir(joinpath(root, "results")) # Processed data
    end
    return FileTree(root)
end

save_run(root::AbstractString, args...) = save_run(FileTree(root), args...)

function save_run(root, id, df, meta)
    meta_path = joinpath(path(root["meta"]), "$id.bson")
    save_meta(meta_path, meta)

    raw_path = joinpath(path(root["raw"]), "$id.arrow")
    RimuIO.save_df(raw_path, df)
    return root
end

function save_meta(path, meta)
    d = Dict{Symbol,Any}()
    for (k, v) in pairs(meta)
        d[Symbol(k)] = v
    end
    JLSO.save(path, d)
end

StatsTools.mean_and_se(path::AbstractString, args...) = mean_and_se(FileTree(path), args...)
function StatsTools.mean_and_se(root, cols, skip)
    @pipe FileTrees.load(root; lazy=true) do file
        RimuIO.load_df(path(file))
    end |>
    mapvalues(_) do df
        mapreduce(vcat, cols) do col
            μ, σ = mean_and_se(df[skip:end, col])
            [Symbol("μ_", col) => μ, Symbol("σ_", col) => σ]
        end |> namedtuple
    end |>
    exec |> reducevalues(vcat, _) |> DataFrame
end

extract_results(path::AbstractString, args...) = extract_results(FileTree(path), args...)
function extract_results(root, skip; mean=(), max=(), ratio=())
    means = mean_and_se(root, mean, skip)
end
