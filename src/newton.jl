immutable Newton <: Optimizer end
Base.summary(::Newton) = "Newton's Method"
"""
# Keyword arguments
* `tolX::Real=1e-8`: search tolerance in x
* `tolG::Real=1e-12`: search tolerance in gradient
* `maxIter::Integer=100`: maximum number of iterations
* `show_trace::Bool=false`: print a status summary on each iteration if true
"""
function newton{T}(r::Function, j::Function, initial_x::AbstractVector{T};
    tolX::Real = 1e-8, tolG::Real = 1e-12, maxIter::Integer = 100, show_trace::Bool = false)

    converged = false
    x_converged = false
    g_converged = false
    need_jacobian = true
    iterCt = 0
    x = copy(initial_x)
    delta_x = copy(initial_x)
    f_calls = 0
    g_calls = 0

    fcur = f(x)
    f_calls += 1

    # Create buffers
    n = length(x)
    JJ = Matrix{T}(n, n)
    n_buffer = Vector{T}(n)

    # Maintain a trace of the system.
    tr = OptimizationTrace{LevenbergMarquardt}()
    if show_trace
        d = Dict("lambda" => lambda)
        os = OptimizationState{LevenbergMarquardt}(iterCt, sum(abs2, fcur), NaN, d)
        push!(tr, os)
        println(os)
    end

    local J
    while (~converged && iterCt < maxIter)
        if need_jacobian
            J = g(x)
            g_calls += 1
            need_jacobian = false
        end
        # we want to solve:
        #    argmin 0.5*||J(x)*delta_x + f(x)||^2 + lambda*||diagm(J'*J)*delta_x||^2
        # Solving for the minimum gives:
        #    (J'*J + lambda*diagm(DtD)) * delta_x == -J' * f(x), where DtD = sum(abs2, J,1)
        # Where we have used the equivalence: diagm(J'*J) = diagm(sum(abs2, J,1))
        # It is additionally useful to bound the elements of DtD below to help
        # prevent "parameter evaporation".
        DtD = vec(sum(abs2, J, 1))
        for i in 1:length(DtD)
            if DtD[i] <= MIN_DIAGONAL
                DtD[i] = MIN_DIAGONAL
            end
        end
        # delta_x = ( J'*J + lambda * Diagonal(DtD) ) \ ( -J'*fcur )
        At_mul_B!(JJ, J, J)
        @simd for i in 1:n
            @inbounds JJ[i, i] += lambda * DtD[i]
        end
        At_mul_B!(n_buffer, J, fcur)
        scale!(n_buffer, -1)
        delta_x = JJ \ n_buffer

        # apply box constraints
        if !isempty(lower)
            @simd for i in 1:n
               @inbounds delta_x[i] = max(x[i] + delta_x[i], lower[i]) - x[i]
            end
        end
        if !isempty(upper)
            @simd for i in 1:n
               @inbounds delta_x[i] = min(x[i] + delta_x[i], upper[i]) - x[i]
            end
        end

        # if the linear assumption is valid, our new residual should be:
        predicted_residual = sum(abs2, J*delta_x + fcur)

        # try the step and compute its quality
        trial_f = f(x + delta_x)
        f_calls += 1
        trial_residual = sum(abs2, trial_f)
        # step quality = residual change / predicted residual change
        rho = (trial_residual - residual) / (predicted_residual - residual)
        if rho > min_step_quality
            x += delta_x
            fcur = trial_f
            residual = trial_residual
            if rho > good_step_quality
                # increase trust region radius
                lambda = max(lambda_decrease*lambda, MIN_LAMBDA)
            end
            need_jacobian = true
        else
            # decrease trust region radius
            lambda = min(lambda_increase*lambda, MAX_LAMBDA)
        end
        iterCt += 1

        # show state
        if show_trace
            g_norm = norm(J' * fcur, Inf)
            d = Dict("g(x)" => g_norm, "dx" => delta_x, "lambda" => lambda)
            os = OptimizationState{LevenbergMarquardt}(iterCt, sum(abs2, fcur), g_norm, d)
            push!(tr, os)
            println(os)
        end

        # check convergence criteria:
        # 1. Small gradient: norm(J^T * fcur, Inf) < tolG
        # 2. Small step size: norm(delta_x) < tolX
        if norm(J' * fcur, Inf) < tolG
            g_converged = true
        elseif norm(delta_x) < tolX*(tolX + norm(x))
            x_converged = true
        end
        converged = g_converged | x_converged
    end

    MultivariateOptimizationResults(
        LevenbergMarquardt(),    # method
        initial_x,             # initial_x
        x,                     # minimizer
        sum(abs2, fcur),       # minimum
        iterCt,                # iterations
        !converged,            # iteration_converged
        x_converged,           # x_converged
        0.0,                   # x_tol
        0.0,
        false,                 # f_converged
        0.0,                   # f_tol
        0.0,
        g_converged,           # g_converged
        tolG,                  # g_tol
        0.0,
        false,                 # f_increased
        tr,                    # trace
        f_calls,               # f_calls
        g_calls,               # g_calls
        0                      # h_calls
    )
end
