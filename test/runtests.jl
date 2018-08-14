using LsqFit, Test, LinearAlgebra, Random, Compat

tests = ["tmp_test.jl"]

println("Running tests:")

for test in tests
    println(" * $(test)")
    include(test)
end