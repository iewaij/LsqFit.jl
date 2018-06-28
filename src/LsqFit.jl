module LsqFit

    export mse,
           sse,
           r2,
           adjr2,
           curve_fit,
           standard_error,
           margin_error,
           confidence_interval,
           estimate_covar

    using Calculus
    using Distributions
    using Compat
    using OptimBase

    import Base.summary

    include("levenberg_marquardt.jl")
    include("curve_fit.jl")
    include("curve_fit_tools.jl")
end
