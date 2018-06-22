LsqFit.jl
===========

Basic least-squares fitting in pure Julia under an MIT license.

The basic functionality was originaly in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), before being separated into this library.  At this time, `LsqFit` only utilizes the Levenberg-Marquardt algorithm for non-linear fitting.

`LsqFit.jl` is part of the [JuliaNLSolvers](https://github.com/JuliaNLSolvers) family.

|Documentation|Package Evaluator|Build Status|
|:-----------:|:---------------:|:---:|
| [![][docs-latest-img]][docs-latest-url] |[![][pkg-0.5-img]][pkg-0.5-url] [![][pkg-0.6-img]][pkg-0.6-url] [![][pkg-0.7-img]][pkg-0.7-url] | [![][build-img]][build-url]

# Features
* Non-linear Least Squares Regression using Levenberg-Marquardt algorithm.
* Assign weights to squared residuals (Weighted and General Least Squares).
* Covariance inference about regression parameters.

# Non-linear Regression

```julia
using LsqFit

# a two-parameter exponential decay model
# x: array of independent variables
# p: array of model parameters
model(x, p) = p[1]*exp.(-x.*p[2])

# some example data
# xdata: independent variables
# ydata: dependent variable
xdata = linspace(0,10,20)
ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5]

fit = curve_fit(model, xdata, ydata, p0)
# fit is a composite type (LsqFitResult), with some interesting values:
#	fit.dof: degrees of freedom
#	fit.param: best fit parameters
#	fit.resid: residuals = vector of residuals
#	fit.jacobian: estimated Jacobian at solution

# We can estimate errors on the fit parameters,
# to get standard error of each parameter:
se = standard_error(fit)
# to get confidence interval of each parameter at 5% significance level:
confidence_interval = confidence_interval(fit, 0.05)

# The finite difference method is used above to approximate the Jacobian.
# Alternatively, a function which calculates it exactly can be supplied instead.
function jacobian_model(x,p)
    J = Array{Float64}(length(x),length(p))
    J[:,1] = exp.(-x.*p[2])    #dmodel/dp[1]
    J[:,2] = -x.*p[1].*J[:,1]  #dmodel/dp[2]
    J
end

fit = curve_fit(model, jacobian_model, xdata, ydata, p0)
```

# Documentation
For more details, see the latest [documentation][docs-latest-url].

# Installation

To install the package, run

```julia
Pkg.add("LsqFit")
```

`LsqFit.jl` is still in development, if you want the latest features, run

```julia
Pkg.checkout("LsqFit")
```

To use the package in your code

```julia
julia> using LsqFit
```

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://julianlsolvers.github.io/LsqFit.jl/latest

[build-img]: https://travis-ci.org/JuliaNLSolvers/LsqFit.jl.svg?branch=master
[build-url]: https://travis-ci.org/JuliaNLSolvers/LsqFit.jl

[pkg-0.5-img]: http://pkg.julialang.org/badges/LsqFit_0.5.svg
[pkg-0.5-url]: http://pkg.julialang.org/?pkg=LsqFit&ver=0.5
[pkg-0.6-img]: http://pkg.julialang.org/badges/LsqFit_0.6.svg
[pkg-0.6-url]: http://pkg.julialang.org/?pkg=LsqFit&ver=0.6
[pkg-0.7-img]: http://pkg.julialang.org/badges/LsqFit_0.7.svg
[pkg-0.7-url]: http://pkg.julialang.org/?pkg=LsqFit&ver=0.7
