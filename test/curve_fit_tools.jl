let
    # fitting noisy data to an exponential model
    # TODO: compability issues
    @. model(t, p) = p[1] * exp(-p[2] * t)

    # some example data
    srand(666)
    xdata = linspace(0,10,20)
    ydata = model(xdata, [1.0, 2.0]) + 0.01*randn(length(xdata))
    p0 = [0.5, 0.5]
    fit = curve_fit(model, xdata, ydata, [0.5, 0.5])
    # can also get error estimates on the fit parameters
    std_errors = standard_error(fit)
    @assert norm(std_errors - [0.009, 0.04]) < 0.01
    conf_int = margin_error(fit, 0.05)
    @assert norm(conf_int - [0.0189, 0.086]) < 0.01
    r2 = r2(fit)
    @assert norm(r2 - 0.9986) < 0.01
    adjr2 = adjr2(fit)
    @assert norm(adjr2 - 0.9984) < 0.01

    # TODO: test with weights
end
