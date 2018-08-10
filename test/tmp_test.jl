f(x, p) = p[1]*exp.(-x.*p[2])
initial_p = [0.7, 0.7]
xdata = range(0, stop=10, length=20)
ydata = f(xdata, [1.0, 2.0]) + 0.01 * randn(length(xdata))

function r!(F, p)
    F = f(xdata, p) - ydata
end

r(p) = f(xdata, p) - ydata

F = similar(ydata)
d = OnceDifferentiable(r!, [1.9, 9.0], F)

least_squares(r!, initial_p)

# curve_fit(f, xdata, ydata, initial_x)
