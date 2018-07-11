let
    # fitting noisy data to an exponential model
    # TODO: compability issues
    # exponential decay model
    @. model(t, p) = p[1] * exp(-p[2] * t)

    # some example data
    srand(666)
    xdata = linspace(0,10,20)
    ydata = model(xdata, [1.0, 2.0]) + 0.01*randn(length(xdata))
    p0 = [0.5, 0.5]
    fit = curve_fit(model, xdata, ydata, [0.5, 0.5])

    # can also get error estimates on the fit parameters
    covar = estimate_covar(fit)

    std_errors = standard_error(fit)
    @assert norm(std_errors - [0.009, 0.04]) < 0.01

    conf_int = margin_error(fit, 0.05)
    @assert norm(conf_int - [0.0189, 0.086]) < 0.01

    @assert norm(r2(fit)- 0.9986) < 0.01

    @assert norm(adjr2(fit) - 0.9984) < 0.01

    # sine function
    @. model(x,p) = p[1] * sin(p[2] * x)
    srand(66)
    xdata = linspace(-5,5,50)
    ydata = model(xdata, [1.0, 2.0]) + randn(50)
    p0 = [2., 2.]
    fit = curve_fit(model, xdata, ydata, p0)

    covar = estimate_covar(fit)
    @assert norm(covar - [0.0526098 0.000652803; 0.000652803 0.00407946]) < 0.0001

    std_errors = standard_error(fit)
    @assert norm(std_errors - [0.229368, 0.0638706]) < 0.0001

    @assert norm(sse(fit) - 60.29861860193) < 0.0001

    # TODO: test with weights
end
