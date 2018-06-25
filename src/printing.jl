function Base.show(io::IO, fit::LsqFitResult)
    print(io, """Results of Least Squares Algorithm:
    * Convergence: $(fit.converged)
    * Estimated Parameters: $(fit.param)
    * Degrees of Freedom: $(fit.dof)
    * Weights: $(fit.wt)
    * Residual Sum of Squars: $(sum(fit.resid))
    """)
    return
end
