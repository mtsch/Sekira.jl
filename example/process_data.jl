using Sekira

t = @summary function t(meta, data)
    nt = meta.n_target
    μ_shift, σ_shift = mean_and_se(data.shift)
    μ_norm, σ_norm = mean_and_se(data.norm)
    μ_proj, _, _, σ_proj_l, σ_proj_u = med_and_errs(ratio_of_means(data.hproj, data.vproj))
    μ_coherence, σ_coherence = mean_and_se(data.coherence)
    μ_ref_coherence, σ_ref_coherence = mean_and_se(data.single_coherence)
    μ_loneliness, σ_loneliness = mean_and_se(data.loneliness)
end

for dir in ARGS
    @info "Processing `$dir`"
    RimuIO.save_df(joinpath(dir, "results", "summary.arrow"), summarize(t, dir; skip=50_000))
end
