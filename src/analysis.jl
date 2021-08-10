using MacroTools: postwalk

"""
    @summary function ... end

Create a function to be passed to [`summarize`](@ref).

# Example

```julia
julia> @summary function energies(meta, data)
    id = "test"
    Ntw = meta.n_target
    len = nrow(data)
    μ_shift, σ_shift = mean_and_se(data.shift)
    μ_proj, _, _, σ_proj_l, σ_proj_u = med_and_errs(ratio_of_means(data.hproj, data.vproj))
end;

julia> summarize(energies, dir) # dir can be a path or a `FileTree`
10×8 DataFrame
 Row │ id      Ntw    len    μ_shift    σ_shift    μ_proj    σ_proj_l    σ_proj_u
     │ String  Int64  Int64  Float64    Float64    Float64   Float64     Float64
─────┼──────────────────────────────────────────────────────────────────────────────
   1 │ test      100  60000  -24.6576   0.145638   -2.23911  140.029     139.326
   2 │ test     1000  60000   -4.59347  0.0395217  -4.5541     0.110722    0.105786
   3 │ test      200  60000   -8.77444  0.100985   -1.08539    9.95662    23.2936
   4 │ test      300  60000   -4.4406   0.102841   -4.64745    0.325664    0.299909
   5 │ test      400  60000   -4.41512  0.0753507  -4.45474    0.239063    0.229831
   6 │ test      500  60000   -4.51584  0.0588216  -4.63888    0.190949    0.181465
   7 │ test      600  60000   -4.44399  0.0519303  -4.68629    0.147418    0.136815
   8 │ test      700  60000   -4.49641  0.0513159  -4.56       0.156817    0.146055
   9 │ test      800  60000   -4.58289  0.0455694  -4.56392    0.133013    0.128233
  10 │ test      900  60000   -4.49216  0.0429634  -4.74169    0.125266    0.119338

julia> @summary(function(meta, data)
    id = "test"
    Ntw = meta.n_target
    len = nrow(data)
    μ_shift, σ_shift = mean_and_se(data.shift)
    μ_proj, _, _, σ_proj_l, σ_proj_u = med_and_errs(ratio_of_means(data.hproj, data.vproj))
end) |> summarize(dir; skip=10_000)
10×8 DataFrame
 Row │ id      Ntw    len    μ_shift    σ_shift    μ_proj    σ_proj_l    σ_proj_u
     │ String  Int64  Int64  Float64    Float64    Float64   Float64     Float64
─────┼──────────────────────────────────────────────────────────────────────────────
   1 │ test      100  50000  -24.6273   0.154834   -4.05181  124.113     115.389
   2 │ test     1000  50000   -4.58082  0.0429057  -4.52871    0.120729    0.11253
   3 │ test      200  50000   -8.83234  0.111493   -1.77266   12.5945     22.3408
   4 │ test      300  50000   -4.39811  0.124721   -4.72357    0.351143    0.326857
   5 │ test      400  50000   -4.41533  0.0802611  -4.46048    0.269546    0.250092
   6 │ test      500  50000   -4.54364  0.0652312  -4.63162    0.19611     0.190288
   7 │ test      600  50000   -4.47056  0.0578356  -4.68352    0.165806    0.15526
   8 │ test      700  50000   -4.49484  0.0543399  -4.61254    0.152938    0.151639
   9 │ test      800  50000   -4.59333  0.0472654  -4.55456    0.148458    0.142926
  10 │ test      900  50000   -4.52037  0.047915   -4.70404    0.131165    0.125946

```
"""
macro summary(expr)
    if @capture(expr, (function fname_(args__) body_ end))
        header = Expr(:call, fname, args...)
    elseif @capture(expr, (function(args__) body_ end))
        header = Expr(:tuple, args...)
    else
        @show expr
        error("`expr` must be a `function` expression")
    end
    # Validate body
    for ex in body.args
        if ex isa LineNumberNode
            continue
        elseif ex isa Expr && ex.head == :(=)
            ex.args[2] = esc(ex.args[2])
        else
            error("only assignment expressions allowed in function body")
        end
    end
    header.args .= esc.(header.args)

    # Collect all names that appear on the left.
    left = Symbol[]
    postwalk(body) do ex
        if @capture(ex, ((names__,) = value_) | (names__ = value_))
            append!(left, names)
        else
            ex
        end
    end
    names = filter(!=(:_), (left...,))
    values = Expr(:tuple, names...)
    new_body = quote
        $body
        namedtuple($names, $values)
    end
    return Expr(:function, header, new_body)
end

"""
    summarize(f, root; skip=0)

Summarize a run by applying the function `f` to every pair of files in the `meta` and `raw`.
`f` should take a `NamedTuple` as the fist argument and `DataFrame` as the second. The first
`skip` rows of the `DataFrames` are skipped.

See also [`@summary`](@ref), a convenience macro for creating `f`.
"""
function summarize(f, root::FileTree; skip=0)
    meta = FileTrees.load(root["meta"]; lazy=true) do f
        namedtuple(JLSO.load(path(f)))
    end
    raw = FileTrees.load(root["raw"]; lazy=true) do f
        RimuIO.load_df(path(f))[skip + 1:end,:]
    end
    map(values(meta), values(raw)) do m, r
        f(exec(m), exec(r))
    end |> DataFrame
end
summarize(f, root::String; kwargs...) = summarize(f, FileTree(root); kwargs...)
summarize(root; kwargs...) = f -> summarize(f, FileTree(root); kwargs...)

export memory_use
function memory_use(df)
    if hasproperty(df, :len_before)
        return maximum(max.(df.len, df.len_before))
    else
        return maximum(df.len)
    end
end

export safe_proj_e
function safe_proj_e(df)
    try
        return med_and_errs(ratio_of_means(df.hproj, df.vproj))
    catch
        return (NaN, NaN, NaN, NaN, NaN)
    end
end
