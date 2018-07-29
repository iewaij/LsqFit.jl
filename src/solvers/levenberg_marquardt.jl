immutable LevenbergMarquardt{T<:Real, V<:Vector} <: AbstractOptimizer
    min_step_quality::T
    max_step_quality::T
    initial_lambda::T
    lambda_increase::T
    lambda_decrease::T
    lower::V
    upper::V
end

Base.summary(::LevenbergMarquardt) = "Levenberg-Marquardt"

"""
    LevenbergMarquardt(; min_step_quality=1e-3,
                         max_step_quality=1e-3,
                         initial_lambda = 10.0,
                         lambda_increase = 10.0,
                         lambda_decrease = 0.1,
                         lower = [],
                         upper = [])

Implements box constraints as described in Kanzow, Yamashita, Fukushima (2004; J
Comp & Applied Math).
"""
function LevenbergMarquardt(; min_step_quality=1e-3,
                     max_step_quality=1e-3,
                     initial_lambda = 10.0,
                     lambda_increase = 10.0,
                     lambda_decrease = 0.1,
                     lower = zeros(0),
                     upper = zeros(0))
    LevenbergMarquardt(min_step_quality, max_step_quality, initial_lambda,lambda_increase, lambda_decrease, lower, upper)
end

mutable struct LevenbergMarquardtState{Tx, T, M, V, L} <: AbstractOptimizerState
    x::Tx
    previous_x::Tx
    previous_f_x::T
    JJ::M
    n_buffer::V
    lambda::L
end

function initial_state(d, initial_x, method::LevenbergMarquardt, options)
    # check parameters
    @assert (isempty(lower) || length(lower)==length(initial_x)) && (isempty(upper) || length(upper)==length(initial_x)) "Bounds must either be empty or of the same length as the number of parameters."
    @assert (isempty(lower) || all(initial_x .>= lower)) && (isempty(upper) || all(initial_x .<= upper)) "Initial guess must be within bounds."
    @assert 0 <= min_step_quality < 1 "0 <= min_step_quality < 1 must hold."
    @assert 0 < good_step_quality <= 1 "0 < good_step_quality <= 1 must hold."
    @assert min_step_quality < good_step_quality "min_step_quality < good_step_quality must hold."

    T = eltype(initial_x)
    n = length(initial_x)
    value_jacobian!!(d, initial_x)

    LevenbergMarquardtState(initial_x,
                            similar(initial_x),
                            T(NaN),
                            Matrix{T}(undef, n, n),
                            Vector{T}(undef, n),
                            method.initial_lambda)
end

function update_state!(state::LevenbergMarquardtState, d, method::LevenbergMarquardt)
    # we want to solve:
    #    argmin 0.5*||J(x)*delta_x + r(x)||^2 + lambda*||diagm(J'*J)*delta_x||^2
    # Solving for the minimum gives:
    #    (J'*J + lambda*diagm(DtD)) * delta_x == -J' * r(p), where DtD = sum(abs2, J, 1)
    # Where we have used the equivalence: diagm(J'*J) = diagm(sum(abs2, J, 1))
    # It is additionally useful to bound the elements of DtD below to help
    # prevent "parameter evaporation".

    # constants
    MAX_LAMBDA = 1e16
    MIN_LAMBDA = 1e-16
    MIN_DIAGONAL = 1e-6

    DtD = vec(sum(abs2, J, 1))

    for i in 1:length(DtD)
        if DtD[i] <= MIN_DIAGONAL
            DtD[i] = MIN_DIAGONAL
        end
    end

    J = jacobian(d)
    f_x = value(d)
    residual = sum(abs2, f_x)

    # delta_x = (J'*J + lambda * Diagonal(DtD) ) \ (-J'*f_x)
    mul!(state.JJ, transpose(J), J)
    @simd for i in 1:n
        @inbounds state.JJ[i, i] += state.lambda * DtD[i]
    end
    mul!(state.n_buffer, transpose(J), f_x)
    rmul!(state.n_buffer, -1)
    delta_x = state.JJ \ state.n_buffer

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
    predicted_residual = sum(abs2, J * delta_x + f_x)

    # try the step and compute its quality
    trial_f = value(d, x + delta_x)
    trial_residual = sum(abs2, trial_f)

    # step quality = residual change / predicted residual change
    rho = (trial_residual - residual) / (predicted_residual - residual)

    if rho > min_step_quality
        state.previous_x = state.x
        state.previous_f_x = f_x
        state.x += delta_x
        if rho > good_step_quality
            # increase trust region radius
            state.lambda = max(method.lambda_decrease * state.lambda, MIN_LAMBDA)
        end
        value_jacobian!!(d, state.x)
    else
        # decrease trust region radius
        state.lambda = min(method.lambda_increase * state.lambda, MAX_LAMBDA)
    end
end

function assess_convergence(d, options, state::LevenbergMarquardtState)
    # check convergence criteria:
    # 1. Small gradient: norm(J^T * f_x, Inf) < g_tol
    # 2. Small step size: norm(delta_x) < x_tol * x
    f_x = value(d)
    J = jacobian(d)
    delta_x = state.x - state.previous_x
    delta_f_x = f_x - state.f_x_previous

    if norm(delta_x) < options.x_tol * (options.x_tol + norm(state.x))
        x_converged = true
    end

    if abs(delta_f_x) <=  options.f_tol * abs(f_x)
        f_converged = true
    end

    if delta_f_x > 0
        f_increased = true
    end

    if norm(J' * f_x, Inf) < options.g_tol
        g_converged = true
    end

    converged = x_converged || g_converged

    x_converged, f_converged, f_increased, g_converged, converged
end

function trace!(tr, d, state, iteration, method::LevenbergMarquardt, options)
    dt = Dict()
    if options.extended_trace
        dt["p"] = copy(state.x)
        dt["J(p)"] = copy(value(d))
    end
    J_norm = vecnorm(value(d), Inf)
    update!(tr,
            iteration,
            value(d),
            J_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end
