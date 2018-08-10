using LsqFit, Test, LinearAlgebra, Random
using OptimBase, Calculus

my_tests = ["tem_test.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end