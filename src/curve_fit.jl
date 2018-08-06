struct LsqFitResult{T,N}
    dof::Int
    param::Vector{T}
    ydata::Vector{T}
    resid::Vector{T}
    jacobian::Matrix{T}
    wt::Array{T,N}
    algorithm::String
    iterations::Int
    converged::Bool
end

# provide a method for those who have their own Jacobian function
function lmfit(ydata, f::Function, g::Function, p0, wt; kwargs...)
    results = levenberg_marquardt(f, g, p0; kwargs...)
    p = minimizer(results)
    resid = f(p)
    n = length(resid)
    dof = n - length(p)
    return LsqFitResult(n, dof, p, ydata, f(p), g(p), wt, summary(results), iterations(results), converged(results))
end

function lmfit(ydata, f::Function, p0, wt; kwargs...)
    # this is a convenience function for the curve_fit() methods
    # which assume f(p) is the cost functionj i.e. the residual of a
    # model where
    #   model(xpts, params...) = ydata + error (noise)

    # this minimizes f(p) using a least squares sum of squared error:
    #   sse = sum(f(p)^2)
    # This is currently embedded in Optim.levelberg_marquardt()
    # which calls sum(abs2)
    #
    # returns p, f(p), g(p) where
    #   p    : best fit parameters
    #   f(p) : function evaluated at best fit p, (weighted) residuals
    #   g(p) : estimated Jacobian at p (Jacobian with respect to p)

    # construct Jacobian function, which uses finite difference method
    g = Calculus.jacobian(f)
    lmfit(ydata, f, g, p0, wt; kwargs...)
end


"""
    curve_fit(model, xdata, ydata, p0) -> fit
Fit data to a non-linear `model`. `p0` is an initial model parameter guess (see Example).
The return object is a composite type (`LsqFitResult`), with some interesting values:

* `fit.dof` : degrees of freedom
* `fit.param` : best fit parameters
* `fit.resid` : residuals = vector of residuals
* `fit.jacobian` : estimated Jacobian at solution

## Example
```julia
# a two-parameter exponential model
# x: array of independent variables
# p: array of model parameters
model(x, p) = p[1]*exp.(-x.*p[2])

# some example data
# xdata: independent variables
# ydata: dependent variable
xdata = range(0, stop=10, length=20)
ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5]

fit = curve_fit(model, xdata, ydata, p0)
```
"""
function curve_fit end

function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, p0; kwargs...)
    # construct the cost function
    f(p) = model(xpts, p) - ydata
    T = eltype(ydata)
    lmfit(ydata, f,p0,T[]; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
            xpts::AbstractArray, ydata::AbstractArray, p0; kwargs...)
    f(p) = model(xpts, p) - ydata
    g(p) = jacobian_model(xpts, p)
    T = eltype(ydata)
    lmfit(ydata, f, g, p0, T[]; kwargs...)
end

function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, wt::Vector, p0; kwargs...)
    # construct a weighted cost function, with a vector weight for each ydata
    # for example, this might be wt = 1/sigma where sigma is some error term
    f(p) = wt .* ( model(xpts, p) - ydata )
    lmfit(ydata, f,p0,wt; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
            xpts::AbstractArray, ydata::AbstractArray, wt::Vector, p0; kwargs...)
    f(p) = wt .* ( model(xpts, p) - ydata )
    g(p) = wt .* ( jacobian_model(xpts, p) )
    lmfit(ydata, f, g, p0, wt; kwargs...)
end

function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, wt::Matrix, p0; kwargs...)
    # as before, construct a weighted cost function with where this
    # method uses a matrix weight.
    # for example: an inverse_covariance matrix

    # Cholesky is effectively a sqrt of a matrix, which is what we want
    # to minimize in the least-squares of levenberg_marquardt()
    # This requires the matrix to be positive definite
    u = chol(wt)

    f(p) = u * ( model(xpts, p) - ydata )
    lmfit(ydata, f,p0,wt; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
            xpts::AbstractArray, ydata::AbstractArray, wt::Matrix, p0; kwargs...)
    u = chol(wt)

    f(p) = u * ( model(xpts, p) - ydata )
    g(p) = u * ( jacobian_model(xpts, p) )
    lmfit(ydata, f, g, p0, wt; kwargs...)
end
