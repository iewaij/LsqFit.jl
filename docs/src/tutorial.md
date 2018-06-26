# Tutorial

## Introduction to Nonlinear Regression

Assume that, for the $i$th observation, the relationship between independent variable $\mathbf{x_i}=\begin{bmatrix} x_{1i},\, x_{2i},\, \ldots\, x_{pi}\ \end{bmatrix}'$ and dependent variable $Y_i$ follows:

```math
Y_i = m(\mathbf{x_i}, \boldsymbol{\gamma}) + \epsilon_i
```

where $m$ is a non-linear model function depends on the independent variable $\mathbf{x_i}$ and the parameter vector $\boldsymbol{\gamma}$. In order to find the parameter $\boldsymbol{\gamma}$ that "best" fit our data, we choose the parameter ${\boldsymbol{\gamma}}$ which minimizes the sum of squared residuals from our data, i.e. solves the problem:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma})= \sum_{i=1}^{n} [m(\mathbf{x_i}, \boldsymbol{\gamma}) - y_i]^2
```

Given that the function $m$ is non-linear, there's no analytical solution for the best $\boldsymbol{\gamma}$. We have to use computational tools, which is `LsqFit.jl` in this tutorial, to find the least squares solution.

One example of non-linear model is the exponential model, which takes a one-element predictor variable $t$. The model function is:

```math
m(t, \boldsymbol{\gamma}) = \gamma_1 \exp(\gamma_2 t)
```

and the model becomes:

```math
Y_i = \gamma_1 \exp(\gamma_2 t_i) + \epsilon_i
```

To fit data using `LsqFit.jl`, pass the defined model function (`m`), data (`tdata` and `ydata`) and the initial parameter value (`p_init`) to `curve_fit()`. For now, `LsqFit.jl` only supports the Levenberg Marquardt algorithm.

```julia
julia> # t: array of independent variables
julia> # p: array of model parameters
julia> m(t, p) = p[1] * exp.(p[2] * t)
julia> p_init = [0.5, 0.5]
julia> fit = curve_fit(m, tdata, ydata, p_init)
```

[`@.`](https://docs.julialang.org/en/stable/stdlib/arrays/#Base.Broadcast.@__dot__) macro can also be used to perform [element-by-element calculation](https://docs.julialang.org/en/stable/manual/functions/#man-vectorized-1).

```julia
julia> # t: array of independent variables
julia> # p: array of model parameters
julia> @. m(t, p) = p[1] * exp(p[2] * t)
julia> p_init = [0.5, 0.5]
julia> fit = curve_fit(m, tdata, ydata, p_init)
```

It will return a composite type `LsqFitResult`, with some interesting values:

*	`fit.dof`: degrees of freedom
*	`fit.param`: best fit parameters
*	`fit.resid`: vector of residuals
*	`fit.jacobian`: estimated Jacobian at the solution




## Jacobian Calculation

The Jacobian $J_f(\mathbf{x})$ of a vector function $f(\mathbf{x}): \mathbb{R}_m \to \mathbb{R}_n$ is deﬁned as the matrix with elements:

```math
[J_f(\mathbf{x})]_{ij} = \frac{\partial f_i(\mathbf{x})}{\partial x_j}
```
The matrix is therefore:

```math
J_f(\mathbf{x}) = \begin{bmatrix}
                \frac{\partial f_1}{\partial x_1}&\frac{\partial f_1}{\partial x_2}&\dots&\frac{\partial f_1}{\partial x_m}\\
                \frac{\partial f_2}{\partial x_1}&\frac{\partial f_2}{\partial x_2}&\dots&\frac{\partial f_2}{\partial x_m}\\
                \vdots&\vdots&\ddots&\vdots\\
                \frac{\partial f_n}{\partial x_1}&\frac{\partial f_n}{\partial x_2}&\dots&\frac{\partial f_n}{\partial x_m}\\
                \end{bmatrix}
```

The Jacobian of the exponential model function with respect to $\boldsymbol{\gamma}$ is:

```math
J_m(t, \boldsymbol{\gamma}) = \begin{bmatrix}
            \frac{\partial m}{\partial \gamma_1} &
            \frac{\partial m}{\partial \gamma_2} \\
            \end{bmatrix}
          = \begin{bmatrix}
            \exp(\gamma_2 t) &
            t \gamma_1 \exp(\gamma_2 t) \\
            \end{bmatrix}
```

By default, the finite difference method, `Calculus.jacobian()`, is used to approximate the Jacobian for the data fitting algorithm and covariance computation. Alternatively, a function which calculates the Jacobian can be supplied to `curve_fit()` for faster and/or more accurate results.

```Julia
function j_m(t,p)
    J = Array{Float64}(length(t),length(p))
    @. J[:,1] = exp(p[2] * t)       #dm/dp[1]
    @. J[:,2] = t * p[1] * J[:,1]   #dm/dp[2]
    J
end

fit = curve_fit(m, j_m, tdata, ydata, p0)
```

## Linear Approximation

The non-linear function $m$ can be approximated as a linear function by Talor expansion:

```math
m(\mathbf{x_i}, \boldsymbol{\gamma}+\boldsymbol{h}) \approx m(\mathbf{x_i}, \boldsymbol{\gamma}) +  \nabla m(\mathbf{x_i}, \boldsymbol{\gamma})'\boldsymbol{h}
```

where $\boldsymbol{\gamma}$ is a fixed vector, $\boldsymbol{h}$ is a very small-valued vector and $\nabla m(\mathbf{x_i}, \boldsymbol{\gamma})$ is the gradient at $\mathbf{x_i}$.

Consider the residual vector functon $r({\boldsymbol{\gamma}})=\begin{bmatrix}
                          r_1({\boldsymbol{\gamma}}) \\
                          r_2({\boldsymbol{\gamma}}) \\
                          \vdots\\
                          r_n({\boldsymbol{\gamma}})
                          \end{bmatrix}$
with entries:

```math
r_i({\boldsymbol{\gamma}}) = m(\mathbf{x_i}, {\boldsymbol{\gamma}}) - Y_i
```

Each entry's linear approximation can hence be written as:

```math
\begin{align}
r_i({\boldsymbol{\gamma}}+\boldsymbol{h}) &= m(\mathbf{x_i}, \boldsymbol{\gamma}+\boldsymbol{h}) - Y_i\\
&\approx m(\mathbf{x_i}, \boldsymbol{\gamma}) + \nabla m(\mathbf{x_i}, \boldsymbol{\gamma})'h - Y_i\\
&= r_i({\boldsymbol{\gamma}}) + \nabla m(\mathbf{x_i}, \boldsymbol{\gamma})'h
\end{align}
```

Since the $i$th row of $J(\boldsymbol{\gamma})$ equals the transpose of the gradient of $m(\mathbf{x_i}, \boldsymbol{\gamma})$, the vector function $r({\boldsymbol{\gamma}}+\boldsymbol{h})$ can be approximated as:

```math
r({\boldsymbol{\gamma}}+\boldsymbol{h}) \approx r({\boldsymbol{\gamma}}) + J(\boldsymbol{\gamma})h
```

which is a linear function on $\boldsymbol{h}$ since ${\boldsymbol{\gamma}}$ is a fixed vector.

## Goodness of Fit

The linear approximation of the non-linear least squares problem leads to the approximation of the covariance matrix of each parameter, from which we can perform regression analysis.

Consider a least squares solution $\boldsymbol{\gamma}^*$, which is a local minimizer of the non-linear problem:

```math
\boldsymbol{\gamma}^* = \underset{\boldsymbol{\gamma}}{\mathrm{arg\,min}} \ \sum_{i=1}^{n} [m(\mathbf{x_i}, \boldsymbol{\gamma}) - y_i]^2
```

Set $\boldsymbol{\gamma}^*$ as the fixed point in linear approximation, $r({\boldsymbol{\gamma^*}}) = r$ and $J(\boldsymbol{\gamma^*}) = J$. A parameter vector near $\boldsymbol{\gamma}^*$ can be expressed as $\boldsymbol{\gamma}=\boldsymbol{\gamma^*} + h$. The local approximation for the least squares problem is:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma})=s(\boldsymbol{\gamma}^*+\boldsymbol{h}) \approx [Jh + r]'[Jh + r]
```

which is essentially the linear least squares problem:

```math
\underset{\boldsymbol{\beta}}{\mathrm{min}} \quad [X\beta-Y]'[X\beta-Y]
```

where $X=J$, $\beta=\boldsymbol{h}$ and $Y=-r({\boldsymbol{\gamma}})$. Solve the equation where the partial derivatives equal to $0$, the analytical solution is:

```math
\hat{\boldsymbol{h}}=\hat{\boldsymbol{\gamma}}-\boldsymbol{\gamma}^*\approx-[J'J]^{-1}J'r
```

The covariance matrix for the analytical solution is:

```math
\mathbf{Cov}(\hat{\boldsymbol{\gamma}}) = \mathbf{Cov}(\boldsymbol{h}) = [J'J]^{-1}J'\mathbf{E}(rr')J[J'J]^{-1}
```

Note that $r$ is the residual vector at the best fit point $\boldsymbol{\gamma^*}$, with entries $r_i = Y_i - m(\mathbf{x_i}, \boldsymbol{\gamma^*})=\epsilon_i$. $\hat{\boldsymbol{\gamma}}$ is very close to $\boldsymbol{\gamma^*}$ and therefore can be replaced by $\boldsymbol{\gamma^*}$.

```math
\mathbf{Cov}(\boldsymbol{\gamma}^*) \approx \mathbf{Cov}(\hat{\boldsymbol{\gamma}})
```

Assume the errors in each sample are independent, normal distributed with zero mean and same variance, i.e. $\epsilon \sim N(0, \sigma^2I)$, the covariance matrix from the linear approximation is therefore:

```math
\mathbf{Cov}(\boldsymbol{\gamma}^*) = [J'J]^{-1}J'\mathbf{Cov}(\epsilon)J[J'J]^{-1} = \sigma^2[J'J]^{-1}
```

where $\sigma^2$ could be estimated as residual sum of squares devided by degrees of freedom:

```math
\hat{\sigma}^2=\frac{s(\boldsymbol{\gamma}^*)}{n-p}
```
In `LsqFit.jl`, the covariance matrix calculation uses QR decomposition to [be more computationally stable](http://www.seas.ucla.edu/~vandenbe/133A/lectures/ls.pdf), which has the form:

```math
\mathbf{Cov}(\boldsymbol{\gamma}^*) = \hat{\sigma}^2 \mathrm{R}^{-1}(\mathrm{R}^{-1})'
```

`estimate_covar()` computes the covariance matrix of fit:

```Julia
julia> cov = estimate_covar(fit)
2×2 Array{Float64,2}:
 0.000116545  0.000174633
 0.000174633  0.00258261
```

The standard error is then the square root of each diagonal elements of the covariance matrix. `standard_error()` returns the standard error of each parameter:

```Julia
julia> se = standard_error(fit)
2-element Array{Float64,1}:
 0.0114802
 0.0520416
```

`margin_error()` computes the product of standard error and the critical value of each parameter at a certain significance level (default is 5%) from t-distribution. The margin of error at 10% significance level can be computed by:

```Julia
julia> margin_of_error = margin_error(fit, 0.1)
2-element Array{Float64,1}:
 0.0199073
 0.0902435
```

`confidence_interval()` returns the confidence interval of each parameter at certain significance level, which is essentially the estimate value ± margin of error. To get the confidence interval at 10% significance level, run:

```Julia
julia> confidence_intervals = confidence_interval(fit, 0.1)
2-element Array{Tuple{Float64,Float64},1}:
 (0.976316, 1.01613)
 (1.91047, 2.09096)
```

## Weighted Least Squares

`curve_fit()` also accepts sigma parameter (`sigma`) to perform Weighted Least Squares and General Least Squares, where the parameter $\boldsymbol{\gamma}^*$ minimizes the transformed (weighted) residual sum of squares.

Assume the errors in each sample are independent, normal distributed with zero mean and **different** variances (heteroskedastic error), i.e. $\epsilon \sim N(0, \Sigma)$, where:

```math
\Sigma = \begin{bmatrix}
         \sigma_1^2    & 0      & \cdots & 0\\
         0      & \sigma_2^2    & \cdots & 0\\
         \vdots & \vdots & \ddots & \vdots\\
         0      & 0    & \cdots & \sigma_n^2\\
         \end{bmatrix}
```

To perform Weighted Least Squares, pass the array of **standard deviations of errors**, `[σ_1, σ_2, ..., σ_n]`, to `curve_fit()`.

```Julia
julia> fit = curve_fit(m, tdata, ydata, εstdd, p0)
julia> cov = estimate_covar(fit)
```

!!! warn
    The `weight` argument has been deprecated. Please make sure that you're passing the vector of the standard deviations of errors, which could be estimated by `abs(fit.resid)`.

!!! note
    `sigma` must be a vector. If `sigma` is a matrix, General Least Squares will be performed.

The weight will be calculated as the reciprocal of the standard deviation of errors:

```math
w_i = \frac{1}{\sigma_i^2}
```

in matrix form:

```math
\mathbf{W} =  \begin{bmatrix}
              w_1    & 0      & \cdots & 0\\
              0      & w_2    & \cdots & 0\\
              \vdots & \vdots & \ddots & \vdots\\
              0      & 0    & \cdots & w_n\\
              \end{bmatrix}
           =  \begin{bmatrix}
              1/\sigma_1^2    & 0      & \cdots & 0\\
              0      & 1/\sigma_2^2    & \cdots & 0\\
              \vdots & \vdots & \ddots & \vdots\\
              0      & 0    & \cdots & 1/\sigma_n^2\\
              \end{bmatrix}
```

The weighted least squares problem becomes:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma})= \sum_{i=1}^{n} w_i[m(\mathbf{x_i}, \boldsymbol{\gamma}) - Y_i]^2
```

in matrix form:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma})= r(\boldsymbol{\gamma})'Wr(\boldsymbol{\gamma})
```

where $r({\boldsymbol{\gamma}})=\begin{bmatrix}
                          r_1({\boldsymbol{\gamma}}),
                          r_2({\boldsymbol{\gamma}}),
                          \dots,
                          r_n({\boldsymbol{\gamma}})
                          \end{bmatrix}'$
is a residual vector function with entries:

```math
r_i({\boldsymbol{\gamma}}) = m(\mathbf{x_i}, {\boldsymbol{\gamma}}) - Y_i
```

The algorithm in `LsqFit.jl` will then provide a least squares solution $\boldsymbol{\gamma}^*$. Set $r({\boldsymbol{\gamma^*}}) = r$ and $J(\boldsymbol{\gamma^*}) = J$, the linear approximation of the weighted least squares problem is then:

```math
\underset{\boldsymbol{\gamma}}{\mathrm{min}} \quad s(\boldsymbol{\gamma}) = s(\boldsymbol{\gamma}^* + \boldsymbol{h}) \approx [J\boldsymbol{h}+r]'W[J\boldsymbol{h}+r]
```

The analytical solution to the linear approximation is:

```math
\hat{\boldsymbol{h}}=\hat{\boldsymbol{\gamma}}-\boldsymbol{\gamma}^*\approx-[J'WJ]^{-1}J'Wr
```

Since the weight matrix is the inverse of the variance, i.e. $W = \Sigma^{-1}$, the covariance matrix is now:

```math
{Cov}(\boldsymbol{\gamma}^*) \approx  [J'WJ]^{-1}J'W \Sigma W'J[J'W'J]^{-1} = [J'WJ]^{-1}
```

In most cases, the variances of errors are unknown. To perform Weighted Least Square, we need estimate the standard deviations of errors first, which is the absolute residual of the $i$th sample:

```math
\widehat{\mathbf{Var}(\epsilon_i)} = \widehat{\mathbf{E}(\epsilon_i^2)} = r_i(\boldsymbol{\gamma}^*)^2
```

```math
\widehat{\sigma_i} = \sqrt{\widehat{\mathbf{Var}(\epsilon_i)}} = |r_i(\boldsymbol{\gamma}^*)|
```

Unweighted fitting (OLS) will return the residuals we need, though the estimator is little bit biased. Then pass the the absolute residual to perform Weighted Least Squares:

```Julia
julia> fit = curve_fit(m, tdata, ydata, p0)
julia> fit_WLS = curve_fit(m, tdata, ydata, abs.(fit.resid), p0)
julia> cov = estimate_covar(fit_WLS)
```

If we only know **the relative ratio of standard deviations**, i.e. $\epsilon \sim N(0, k\Sigma)$, the covariance matrix of the estimated parameter will be:

```math
\mathbf{Cov}(\boldsymbol{\gamma}^*) = k[J'WJ]^{-1}
```

where $k$ is estimated as:

```math
\hat{k}=\frac{RSS}{n-p}
```

In this case, if we set $W = I$, the result will be the same as the unweighted version. However, `curve_fit()` currently **does not support** this implementation, i.e. the covariance of the estimated parameter is calculated as `covar = inv(J'*fit.wt*J)`. A workaround is to multiply the estimated covariance by `k`:

```Julia
julia> fit = curve_fit(m, tdata, ydata, εstdd_ratio, p0)
julia> k = sum(fit.resid) / fit.dof
julia> cov = k * estimate_covar(fit)
```

## General Least Squares
Assume the errors in each sample are **correlated**, normal distributed with zero mean and **different** variances (heteroskedastic and autocorrelated error), i.e. $\epsilon \sim N(0, \Sigma)$. Pass the covariance matrix of errors (`εcov`) to the function `curve_fit()` to perform General Least Squares:

```Julia
julia> fit = curve_fit(m, tdata, ydata, εcov, p0)
julia> cov = estimate_covar(fit)
```

!!! warn
    The `weight` argument has been deprecated. Please make sure that you're passing the covariance matrix of errors.

The transform matrix will be calculated as the inverse of the error covariance matrix, i.e. $W = \Sigma^{-1}$, we will get the parameter covariance matrix:

```math
\mathbf{Cov}(\boldsymbol{\gamma}^*) \approx  [J'WJ]^{-1}J'W \Sigma W'J[J'W'J]^{-1} = [J'WJ]^{-1}
```

!!! note
    In `LsqFit.jl`, the residual function passed to `levenberg_marquardt()` is in different format, if the weight is a vector:

    ```julia
    r(p) = sqrt_wt .* ( model(xpts, p) - ydata )
    lmfit(r, g, p0, wt; kwargs...)
    ```

    ```math
    r_i({\boldsymbol{\gamma}}) = \sqrt{w_i} \cdot [m(\mathbf{x_i}, {\boldsymbol{\gamma}}) - Y_i]
    ```

    Cholesky decomposition, which is effectively a square root of matrix, will be performed if the `sigma` is a matrix:

    ```julia
    u = chol(wt)
    r(p) = u * ( model(xpts, p) - ydata )
    lmfit(r, p0, wt; kwargs...)
    ```

    ```math
    r({\boldsymbol{\gamma}}) =  W^{\frac{1}{2}}\begin{bmatrix}
                                m(\mathbf{x_1}, {\boldsymbol{\gamma}}) - Y_1\\
                                \vdots\\
                                m(\mathbf{x_n}, {\boldsymbol{\gamma}}) - Y_n\\
                                \end{bmatrix}
    ```

    The solution will be the same as the least squares problem mentioned in the tutorial.

## References
Hansen, P. C., Pereyra, V. and Scherer, G. (2013) Least squares data fitting with applications. Baltimore, Md: Johns Hopkins University Press, p. 147-155.

Kutner, M. H. et al. (2005) Applied Linear statistical models.

Weisberg, S. (2014) Applied linear regression. Fourth edition. Hoboken, NJ: Wiley (Wiley series in probability and statistics).
