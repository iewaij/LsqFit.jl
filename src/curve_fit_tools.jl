"""
<<<<<<< HEAD
    Base.show(io, fit)
=======
    Base.show(io::IO, fit::LsqFitResult)
>>>>>>> 1af45a49bbd10fe26df21800024e9345b93bacd4

Show the output of `LsqFitResult`.
"""
function Base.show(io::IO, fit::LsqFitResult)
<<<<<<< HEAD
    print(io,
    """Results of Least Squares Fitting:
    * Algorithm: $(fit.algorithm)
    * Iterations: $(fit.iterations)
    * Converged: $(fit.converged)
=======
    print(io, """Results of Least Squares Fitting:
    * Algorithm: $(fit.algorithm)
    * Convergence: $(fit.converged)
    * Iterations: $(fit.iterations)
>>>>>>> 1af45a49bbd10fe26df21800024e9345b93bacd4
    * Estimated Parameters: $(fit.param)
    * Sample Size: $(fit.n)
    * Degrees of Freedom: $(fit.dof)
    * Weights: $(fit.wt)
<<<<<<< HEAD
    * Sum of Squared Errors: $(round(sse(fit), 4))
    * Mean Squared Errors: $(round(mse(fit), 4))
    * R²: $(round(r2(fit), 4))
    * Adjusted R²: $(round(adjr2(fit), 4))
=======
    * Residual Sum of Squares: $(rss(fit))
    * σ²: $(estimate_sigma2(fit))
    * R²: $(r2(fit))
>>>>>>> 1af45a49bbd10fe26df21800024e9345b93bacd4
    """)
    println(io, "\nVariance Inferences:")
    nc = 4
    nr = length(fit.param)
    outrows = Matrix{String}(nr+1, nc)
    outrows[1, :] = ["k", "value", "std error", "95% conf int"]

    for i in 1:nr
        outrows[i+1, :] = ["$i", "$(round(fit.param[i], 4))", "$(round(standard_error(fit)[i], 4))",
                           "$(round.(confidence_interval(fit)[i], 3))"]
    end

    colwidths = length.(outrows)
    max_colwidths = [maximum(view(colwidths, :, i)) for i in 1:nc]

    for r in 1:nr+1
        for c in 1:nc
            cur_cell = outrows[r, c]
            cur_cell_len = length(cur_cell)

            padding = " "^(max_colwidths[c]-cur_cell_len)
            if c > 1
                padding = " "*padding
            end

            print(io, padding)
            print(io, cur_cell)
        end
        print(io, "\n")
    end
end

"""
<<<<<<< HEAD
    sse(fit)

Return the residual sum of squares (RSS), also known as the sum of squared residuals (SSR) or the sum of squared errors (SSE).

```math
SSE = \sum_{i=1}^{n} [Y_i - m(\mathbf{x_i}, \boldsymbol{\gamma}^*)]^2
```

for the transformed version:

```math
SSE = r(\mathbf{X}, \boldsymbol{\gamma}^*)'Wr(\mathbf{X}, \boldsymbol{\gamma}^*)
```
"""
function sse(fit::LsqFitResult)
    sse = sum(abs2, fit.resid)
end

"""
    sst(fit)

Return the total sum of squares (SST).

```math
SST = \sum_{i=1}^{n} [Y_i - \bar{Y_i}]^2
```
"""
function sst(fit::LsqFitResult)
    sst = sum(abs2, fit.ydata .- mean(fit.ydata))
end

"""
    r2(fit)

Return the explained variance, also known as R².
=======
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
>>>>>>> 1af45a49bbd10fe26df21800024e9345b93bacd4

```math
R^2 = \frac{\mathbf{Var}[m(\mathbf{X}, \boldsymbol{\gamma}^*)]}{\mathbf{Var}(Y)} = 1 - \frac{RSS}{TSS}
```
"""
function r2(fit::LsqFitResult)
<<<<<<< HEAD
    r2 = 1 - sse(fit)/sst(fit)
end

"""
    adjr2(fit)

Return the adjusted R².

```math
R_{adj}^2 = 1 - (1-R^2)\frac{n-1}{n-p-1}
```
"""
function adjr2(fit::LsqFitResult)
    adjr2 = 1 - (1 - r2(fit))*(fit.n-1)/(fit.dof-1)
=======
    r2 = 1 - rss(fit)/tss(fit)
end

"""
    adj_r2(fit::LsqFitResult)
"""
function adj_r2(fit::LsqFitResult)
>>>>>>> 1af45a49bbd10fe26df21800024e9345b93bacd4
end

"""
    mse(fit)

Return the unbiased estimate of error term variance σ² assuming ϵ ~ N(0, σ²I), also known as MSE.

```math
MSE = \widehat{\sigma^2} = \frac{RSS}{n-p}
```
"""
function mse(fit::LsqFitResult)
<<<<<<< HEAD
    mse = sse(fit) / fit.dof
=======
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
>>>>>>> 1af45a49bbd10fe26df21800024e9345b93bacd4
end

"""
    rmse(fit)

Return the square root of MSE.
"""
function rmse(fit::LsqFitResult)
    rmse = sqrt(mse(fit))
end

"""
    estimate_covar(fit)

Return the covariance matrix of parameters.
"""
function estimate_covar(fit::LsqFitResult)
    J = fit.jacobian

    if isempty(fit.wt)
<<<<<<< HEAD
        sigma2 = mse(fit)
=======
        sigma2 = estimate_sigma2(fit)
>>>>>>> 1af45a49bbd10fe26df21800024e9345b93bacd4
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

"""
    standard_error(fit; rtol=NaN, atol=0)

Return the standard error of parameters.

# Arguments
fit::LsqFitResult: a LsqFitResult from a curve_fit()
atol::Real: absolute tolerance for approximate comparisson to 0.0 in negativity check
rtol::Real: relative tolerance for approximate comparisson to 0.0 in negativity check
"""
function standard_error(fit::LsqFitResult; rtol::Real=NaN, atol::Real=0)
    covar = estimate_covar(fit)
    # standard errors are given by the sqrt of the diagonal
    vars = diag(covar)
    vratio = minimum(vars)/maximum(vars)
    if !isapprox(vratio, 0.0, atol=atol, rtol=isnan(rtol) ? Base.rtoldefault(vratio, 0.0, 0) : rtol) && vratio < 0.0
        error("Covariance matrix is negative for atol=$atol and rtol=$rtol")
    end
    return sqrt.(abs.(vars))
end

"""
    margin_error(fit, alpha=0.05; rtol=NaN, atol=0)

Return margin of error at alpha significance level, which is the product of standard errors and quantile of the student-t distribution (critical values).

# Arguments
fit::LsqFitResult: a LsqFitResult from a curve_fit()
alpha::Real: significance level, e.g. alpha=0.05 for 95% confidence
atol::Real: absolute tolerance for approximate comparisson to 0.0 in negativity check
rtol::Real: relative tolerance for approximate comparisson to 0.0 in negativity check
"""
function margin_error(fit::LsqFitResult, alpha=0.05; rtol::Real=NaN, atol::Real=0)
    std_errors = standard_error(fit; rtol=rtol, atol=atol)
    dist = TDist(fit.dof)
    critical_values = quantile(dist, 1 - alpha/2)
    # scale standard errors by quantile of the student-t distribution (critical values)
    return std_errors * critical_values
end

"""
    margin_error(fit, alpha=0.05; rtol=NaN, atol=0)

Return confidence intervals at alpha significance level.

# Arguments
fit::LsqFitResult: a LsqFitResult from a curve_fit()
alpha::Real: significance level, e.g. alpha=0.05 for 95% confidence
atol::Real: absolute tolerance for approximate comparisson to 0.0 in negativity check
rtol::Real: relative tolerance for approximate comparisson to 0.0 in negativity check
"""
function confidence_interval(fit::LsqFitResult, alpha::Real=0.05; rtol::Real=NaN, atol::Real=0)
    std_errors = standard_error(fit; rtol=rtol, atol=atol)
    margin_of_errors = margin_error(fit, alpha; rtol=rtol, atol=atol)
    confidence_intervals = collect(zip(fit.param-margin_of_errors, fit.param+margin_of_errors))
end

@deprecate estimate_errors(fit::LsqFitResult, confidence=0.95; rtol::Real=NaN, atol::Real=0) margin_error(fit, 1-confidence; rtol=rtol, atol=atol)
