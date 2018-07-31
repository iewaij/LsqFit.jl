#
# Correctness Tests
#

using LsqFit, Base.Test, Compat

tests = ["curve_fit.jl", "curve_fit_tools.jl", "levenberg_marquardt.jl"]

println("Running tests:")

# for test in tests
#     println(" * $(my_test)")
#     include(test)
# end

include("tmp_test.jl")
