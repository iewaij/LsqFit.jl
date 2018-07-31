"""
    curve_fit(model, [jacobian], x, y, [sigma,] p0; kwargs...)

Fit data to a non-linear `model` by minimizing the (transformed) residual between the model and the dependent variable data (`y`). `p0` is an initial model parameter guess.

The return object is a composite type (`LsqFitResult`), with some interesting values:

* `fit.dof` : degrees of freedom
* `fit.param` : best fit parameters
* `fit.resid` : vector of residuals
* `fit.jacobian` : estimated Jacobian at solution

# Arguments
* `model`: function that takes two arguments (x, params)
* `jacobian`: (optional) function that returns the Jacobian matrix of `model`
* `x`: the independent variable
* `y`: the dependent variable that constrains `model`
* `sigma::Vector`: (optional) the standard deviations of errors to perform Weighted Least Squares.
* `sigma::Matrix`: (optional) the covariance matrix of errors to perform General Least Squares.
* `p0`: initial guess of the model parameters
* `kwargs`: tuning parameters for fitting, passed to `levenberg_marquardt`, such as `maxIter` or `show_trace`

# Example
```julia
# a two-parameter exponential model
# t: array of independent variables
# p: array of model parameters
julia> model(x, p) = p[1] * exp.(-p[2] * t)

# some example data
# tdata: independent variables
# ydata: dependent variable
julia> tdata = linspace(0,10,20)
julia> ydata = model(tdata, [1.0 2.0]) + 0.01*randn(length(tdata))
julia> p0 = [0.5, 0.5]

julia> fit = curve_fit(model, tdata, ydata, p0)
```
"""
function curve_fit end

function curve_fit(m::Function, xpts::AbstractArray, ydata::AbstractArray, initial_p::AbstractArray, method::M = LevenbergMarquardt(), options::Options = Options(;default_options(method)...); inplace = true, autodiff = :finite, kwargs...)
    # construct the residual function
    r(p) = m(xpts, p) - ydata
    d = OnceDifferentiable(r, initial_p, inplace = inplace, autodiff = autodiff, kwargs...)
    # get least squares result
    result = least_squares(d, initial_p, method, options)
    p = minimizer(results)
    f, j = value_jacobian!!(d, p)
    n = length(resid)
    dof = n - length(p)
    return LsqFitResult(n, dof, p, ydata, f, j, wt, summary(results), iterations(results), converged(results))
end

function curve_fit(m::Function, j::Function, xpts::AbstractArray, ydata::AbstractArray, initial_p::AbstractArray, method::M = LevenbergMarquardt(), options::Options = Options(;default_options(method)...); inplace = true, autodiff = :finite, kwargs...)
    # construct the residual function
    r(p) = m(xpts, p) - ydata
    d = OnceDifferentiable(r, j, initial_p, inplace = inplace, kwargs...)
    # get least squares result
    result = least_squares(d, initial_p, method, options)
    p = minimizer(results)
    f, j = value_jacobian!!(d, p)
    n = length(resid)
    dof = n - length(p)
    return LsqFitResult(n, dof, p, ydata, f, j, wt, summary(results), iterations(results), converged(results))
end

"""
    curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, sigma::Vector, p0; kwargs...)

use `sigma` to construct a weighted cost function to perform Weighted Least Squares, where `sigma` is a vector of the standard deviations of errors, i.e. ϵ_i ~ N(0, σ_i^2), which could be estimated as `abs(fit.resid)`.
"""
function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, sigma::Vector, p0; kwargs...)
    warn("The `weight` argument has been deprecated. Please make sure that you're passing the vector of the standard deviations of errors, which could be estimated by `abs(fit.resid)`.")
    sqrt_wt = 1 ./ sigma
    wt = sqrt_wt.^2
    f(p) = sqrt_wt .* ( model(xpts, p) - ydata )
    lmfit(f,p0,wt; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
                   xpts::AbstractArray, ydata::AbstractArray, sigma::Vector, p0; kwargs...)
    warn("The `weight` argument has been deprecated. Please make sure that you're passing the vector of the standard deviations of errors, which could be estimated by `abs(fit.resid)`.")
    sqrt_wt = 1 ./ sigma
    wt = sqrt_wt.^2
    f(p) = sqrt_wt .* ( model(xpts, p) - ydata )
    g(p) = sqrt_wt .* ( jacobian_model(xpts, p) )
    lmfit(f, g, p0, wt; kwargs...)
end

"""
    function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, sigma::Matrix, p0; kwargs...)

use `sigma` to construct a transformed cost function to perform General Least Squares, where `sigma` is a matrix of the covariance matrix of error, i.e. ϵ ~ N(0, Σ).
"""
function curve_fit(model::Function, xpts::AbstractArray, ydata::AbstractArray, sigma::Matrix, p0; kwargs...)
    warn("The `weight` argument has been deprecated. Please make sure that you're passing the covariance matrix of errors.")
    wt = inv(sigma)
    # Cholesky is effectively a sqrt of a matrix, which is what we want
    # to minimize in the least-squares of levenberg_marquardt()
    # This requires the matrix to be positive definite
    u = chol(wt)
    f(p) = u * ( model(xpts, p) - ydata )
    lmfit(f,p0,wt; kwargs...)
end

function curve_fit(model::Function, jacobian_model::Function,
                   xpts::AbstractArray, ydata::AbstractArray, sigma::Matrix, p0; kwargs...)
    warn("The `weight` argument has been deprecated. Please make sure that you're passing the covariance matrix of errors.")
    wt = inv(sigma)
    u = chol(wt)
    f(p) = u * ( model(xpts, p) - ydata )
    g(p) = u * ( jacobian_model(xpts, p) )
    lmfit(f, g, p0, wt; kwargs...)
end
