mutable struct METTSLogger
    energies::Vector{Any}
    Sz_all::Vector{Any}
    Sx_all::Vector{Any}
end

function METTSLogger()
    METTSLogger([], [], [])
end

function log_observables!(logger::METTSLogger, E0, Sz_sites, Sx_sites)
    push!(logger.energies, real(E0))
    push!(logger.Sz_all, Sz_sites)
    push!(logger.Sx_all, Sx_sites)
end


function run_METTS(mps, H, projectors_1, projectors_2, beta, dt=0.1, N_samples=20, time_evol_alg=:tdvp, 
    gate=nothing, gate_half=nothing, trotter_order=2; trscheme=nothing, verbose=1, logger=nothing)

    if time_evol_alg == :tebd || time_evol_alg == :tdvp2
        if isnothing(trscheme)
            error("trscheme must be provided for time_evol_alg :tebd or :tdvp2")
        end
    end

    if time_evol_alg == :tebd
        if isnothing(gate)
            error("gate must be provided for time_evol_alg :tebd")
        end
        if trotter_order == 2 && isnothing(gate_half)
            error("gate_half must be provided for time_evol_alg :tebd with trotter_order=2")
        end
    end

    if verbose > 0
        println("Running METTS for beta=$beta, N_samples=$N_samples")
        println("\t Time-evolution algorithm: $time_evol_alg, dt=$dt")
        if time_evol_alg == :tebd
            println("\t\t Trotter order: $trotter_order")
        end
    end
    
    if time_evol_alg == :tdvp
        alg = TDVP()
    elseif time_evol_alg == :tdvp2
        alg = TDVP2(; trscheme=trscheme)
    end
    
    beta_half = beta/2
    t_span = collect(0:dt:beta_half)
    n_steps_timeevol = length(t_span)

    if isnothing(logger)
        energies = []
        Sz_all = []
        Sx_all = []
    end

    envs = environments(mps, H)

    for i in 1:N_samples
        println("METTS step $(i)")
        if i%2 == 0
            projectors = projectors_1
        else
            projectors =  projectors_2
        end
        
        # Imaginary time-evolution
        if time_evol_alg == :tdvp || time_evol_alg == :tdvp2
            mps, envs = time_evolve(mps, H, t_span, alg, envs; imaginary_evolution=true, verbosity=1)
            # for t in 1:n_steps_timeevol
            #     mps, envs = timestep(mps, H, 0, dt, alg, envs; imaginary_evolution=true)
            #     if t%1 == 0 
            #         normalize!(mps)
            #     end
            #     # @show norm(mps)
            # end
            # @show norm(mps)
        elseif time_evol_alg == :tebd
            if trotter_order == 1
                mps = tebd!(mps, physical_space, dt, n_steps_timeevol, gate; imaginary_time=true, trotter_order=1, trscheme=trscheme)
            elseif trotter_order == 2
                mps = tebd!(mps, physical_space, dt, n_steps_timeevol, gate, gate_half; imaginary_time=true, trotter_order=2, trscheme=trscheme)
            end
        end
        
        # Measure expectation values
        E0 = expectation_value(mps, H)
        Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
        Sx_sites = [real(expectation_value(mps, i=>S_x())) for i in 1:L]
        Sz_tot = sum(Sz_sites)/L
    

        if !isnothing(logger)
            log_observables!(logger, E0, Sz_sites, Sx_sites)
        else
            push!(energies, real(E0))
            push!(Sz_all, Sz_sites)
            push!(Sx_all, Sx_sites)
        end
    
        # Collapse onto a new product state
        collapse_to_cps!(mps, projectors)
    end
    if !isnothing(logger)
        return logger
    else
        return energies, Sz_all, Sx_all
    end

end