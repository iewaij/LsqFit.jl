"""
    Base.show(io::IO, fit::LsqFitResult)

Show the output of `LsqFitResult`.
"""
function Base.show(io::IO, fit::LsqFitResult)
    print(io, """Results of Least Squares Fitting:
    * Algorithm: $(fit.algorithm)
    * Convergence: $(fit.converged)
    * Iterations: $(fit.iterations)
    * Estimated Parameters: $(fit.param)
    * Sample Size: $(fit.n)
    * Degrees of Freedom: $(fit.dof)
    * Weights: $(fit.wt)
    * Residual Sum of Squares: $(rss(fit))
    * σ²: $(estimate_sigma2(fit))
    * R²: $(r2(fit))
    """)
    return
end

"""
    rss(fit::LsqFitResult)

Calulate the residual sum of squares (RSS), also known as the sum of squared residuals (SSR) or the sum of squared errors (SSE).

```math
RSS =
```
"""
function rss(fit::LsqFitResult)
    rss = sum(abs2, fit.resid)
end

function tss(fit::LsqFitResult)
    tss = sum(abs2, ydata .- mean(ydata))
end

"""
    r2(fit::LsqFitResult)

Calulate the explained variance, also known as R².

```math
R^2 = \frac{\mathbf{Var}[m(\mathbf{X}, \boldsymbol{\gamma}^*)]}{\mathbf{Var}(Y)} = 1 - \frac{RSS}{TSS}
```
"""
function r2(fit::LsqFitResult)
    r2 = 1 - rss(fit)/tss(fit)
end

"""
    adj_r2(fit::LsqFitResult)
"""
function adj_r2(fit::LsqFitResult)
end

function mse(fit::LsqFitResult)
    mse = rss(fit) / fit.n
end

function rmse(fit::LsqFitResult)
    rmse = sqrt(mse(fit))
end

"""
    estimate_sigma2(fit::LsqFitResult)

Calulate the unbiased estimate of error term variance σ², assuming ϵ ~ N(0, σ²I).

```math
\widehat{\sigma^2} = \frac{RSS}{n-p}
```
"""
function estimate_sigma2(fit::LsqFitResult)
    sigma2 = rss(fit) / fit.dof
end

"""
    estimate_sigma(fit::LsqFitResult)

Calulate the unbiased estimate of error term standard deviation σ², assuming ϵ ~ N(0, σ²I).

```math
\widehat{\sigma} = \sqrt{\frac{RSS}{n-p}}
```
"""
function estimate_sigma(fit::LsqFitResult)
    sigma2 =  estimate_sigma2(fit)
    sigma = sqrt(sigma2)
end

function estimate_covar(fit::LsqFitResult)
    # computes covariance matrix of fit parameters
    J = fit.jacobian

    if isempty(fit.wt)
        sigma2 = estimate_sigma2(fit)
        # compute the covariance matrix from the QR decomposition
        Q,R = qr(J)
        Rinv = inv(R)
        covar = Rinv*Rinv'*sigma2
    elseif length(size(fit.wt)) == 1
        covar = inv(J'*Diagonal(fit.wt)*J)
    else
        covar = inv(J'*fit.wt*J)
    end

    return covar
end

function standard_error(fit::LsqFitResult; rtol::Real=NaN, atol::Real=0)
    # computes standard error of estimates from
    #   fit   : a LsqFitResult from a curve_fit()
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    covar = estimate_covar(fit)
    # then the standard errors are given by the sqrt of the diagonal
    vars = diag(covar)
    vratio = minimum(vars)/maximum(vars)
    if !isapprox(vratio, 0.0, atol=atol, rtol=isnan(rtol) ? Base.rtoldefault(vratio, 0.0, 0) : rtol) && vratio < 0.0
        error("Covariance matrix is negative for atol=$atol and rtol=$rtol")
    end
    return sqrt.(abs.(vars))
end

function margin_error(fit::LsqFitResult, alpha=0.05; rtol::Real=NaN, atol::Real=0)
    # computes margin of error at alpha significance level from
    #   fit   : a LsqFitResult from a curve_fit()
    #   alpha : significance level, e.g. alpha=0.05 for 95% confidence
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    std_errors = standard_error(fit; rtol=rtol, atol=atol)
    dist = TDist(fit.dof)
    critical_values = quantile(dist, 1 - alpha/2)
    # scale standard errors by quantile of the student-t distribution (critical values)
    return std_errors * critical_values
end

function confidence_interval(fit::LsqFitResult, alpha=0.05; rtol::Real=NaN, atol::Real=0)
    # computes confidence intervals at alpha significance level from
    #   fit   : a LsqFitResult from a curve_fit()
    #   alpha : significance level, e.g. alpha=0.05 for 95% confidence
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    std_errors = standard_error(fit; rtol=rtol, atol=atol)
    margin_of_errors = margin_error(fit, alpha; rtol=rtol, atol=atol)
    confidence_intervals = collect(zip(fit.param-margin_of_errors, fit.param+margin_of_errors))
end

@deprecate estimate_errors(fit::LsqFitResult, confidence=0.95; rtol::Real=NaN, atol::Real=0) margin_error(fit, 1-confidence; rtol=rtol, atol=atol)

function overview(fit::LsqFitResult)
end
