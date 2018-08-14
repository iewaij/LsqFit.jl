using NLSolversBase
using LsqFit

p_initial = [0.0, 0.0]
xdata = range(2, stop=12, length=10)
ydata = 12.5 * exp.(-xdata * 0.2) + 0.1 * randn(length(xdata))
function r!(F, p)
    F[:] = p[1]*exp.(-xdata.*p[2]) - ydata
end
F = zeros(10)
od = OnceDifferentiable(r!, [2321., 42.], F)
least_squares(od, p_initial)
# curve_fit(f, xdata, ydata, initial_x)
