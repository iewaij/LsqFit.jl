# LsqFit.jl

Basic least-squares fitting in pure Julia under an MIT license.

The basic functionality was originaly in [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl), before being separated into this library.  At this time, `LsqFit` only utilizes the Levenberg-Marquardt algorithm for non-linear fitting.

`LsqFit.jl` is part of the [JuliaNLSolvers](https://github.com/JuliaNLSolvers) family.

|Source|Package Evaluator|Build Status|
|:----:|:---------------:|:---:|
| [![](https://img.shields.io/badge/GitHub-source-green.svg)](https://github.com/JuliaNLSolvers/LsqFit.jl) |[![](http://pkg.julialang.org/badges/LsqFit_0.5.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.5) [![](http://pkg.julialang.org/badges/LsqFit_0.6.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.6) [![](http://pkg.julialang.org/badges/LsqFit_0.7.svg)](http://pkg.julialang.org/?pkg=LsqFit&ver=0.7) | [![](https://travis-ci.org/JuliaNLSolvers/LsqFit.jl.svg?branch=master)](https://travis-ci.org/JuliaNLSolvers/LsqFit.jl)

# Features
* Non-linear Least Squares Regression using Levenberg-Marquardt algorithm.
* Perform Weighted and General Least Squares.
* Covariance inference about regression parameters.

## Installation

To install the package, run

```julia
Pkg.add("LsqFit")
```

`LsqFit.jl` is still in development, if you want the latest features, run

```julia
Pkg.checkout("LsqFit")
```

To use the package in your code

```julia
julia> using LsqFit
```
