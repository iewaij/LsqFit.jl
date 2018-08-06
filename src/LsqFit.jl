module LsqFit

    export mse,
           sse,
           sst,
           r2,
           adjr2,
           curve_fit,
           standard_error,
           margin_error,
           confidence_interval,
           estimate_covar

    using Calculus
    using Distributions
    using OptimBase
    using LinearAlgebra

    import Base.summary

    include("levenberg_marquardt.jl")
    include("curve_fit.jl")
    include("curve_fit_tools.jl")
end
