struct GaussNewton <: AbstractOptimizer end

Base.summary(::GaussNewton) = "Gauss-Newton"

mutable struct GaussNewtonState{Tx, T, L, M, V} <: AbstractOptimizerState
    x::Tx
    previous_x::Tx
    previous_f_x::T
end

function initial_state(d, initial_x, method::GaussNewton, options)
    initial_x = copy(initial_x)
    value_gradient!!(d, initial_x)
    SteepestDescentState(initial_x, # Maintain current state in state.x
                         similar(initial_x), # Maintain previous state in state.x_previous
                         real(T(NaN))) # Store previous f in state.f_x_previous
end

function update_state!(state::GaussNewtonState, d, method::GaussNewton)
    # values from objective
    J = jacobian(d)
    f_x = value(d)
    residual_sum = sum(abs2, f_x)
    # delta_x = (J'*J)^(-1) * J * f_x
    delta_x = inv(J' * J) * J * f_x
    state.x += delta_x
end

function assess_convergence(d, options, state::GaussNewtonState)
    # check convergence criteria:
    # 1. Small jacobian: norm(J^T * f_x, Inf) < g_tol
    # 2. Small step size: norm(delta_x) < x_tol * x
    f_x = value(d)
    J = jacobian(d)
    delta_x = state.x - state.previous_x
    delta_f_x = f_x - state.previous_f_x

    if norm(delta_x) < options.x_tol * (options.x_tol + norm(state.x))
        x_converged = true
    end

    if abs.(delta_f_x) .<=  options.f_tol * abs.(f_x)
        f_converged = true
    end

    if delta_f_x > 0
        f_increased = true
    end

    if norm(J' * f_x, Inf) < options.g_tol
        g_converged = true
    end

    converged = x_converged | g_converged

    x_converged, f_converged, f_increased, g_converged, converged
end

function trace!(tr, d, state, iteration, method::GaussNewton, options)
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
