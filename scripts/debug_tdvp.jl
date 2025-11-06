using MPSKit, MPSKitModels

function debug_imaginary_time(; L=10, J=1.0, h=0.5, dt=0.1, num_steps=50)
    # Setup
    V = ComplexSpace(2)
    V_virt = ComplexSpace(D)
    ψ = FiniteMPS(rand, ComplexF64, L, V, V_virt)
    normalize!(ψ)
    
    chain = FiniteChain(L)
    # H = transverse_field_ising(; L=L, J=J, h=h)
    H = heisenberg_XXX(chain; J=J, spin=1//2)
    
    max_dim = D
    atol = 1e-8
    combined_trunc = truncrank(max_dim) & trunctol(; atol)
    
    # Check Hamiltonian
    E_initial = real(expectation_value(ψ, H))
    println("Initial energy: $E_initial")
    println("Initial norm: $(norm(ψ))")
    println()
    
    # Try both conventions
    println("=== Testing +im * dt ===")
    ψ_plus = copy(ψ)
    for step in 1:num_steps
        ψ_plus, _ = timestep(ψ_plus, H, 0, im * dt, TDVP2(; trscheme=combined_trunc))
        E = real(expectation_value(ψ_plus, H))
        n = norm(ψ_plus)
        println("Step $step: E = $E, norm = $n")
    end
    
    println("\n=== Testing -im * dt ===")
    ψ_minus = copy(ψ)
    for step in 1:num_steps
        ψ_minus, _ = timestep(ψ_minus, H, 0, -im * dt, TDVP2(; trscheme=combined_trunc))
        E = real(expectation_value(ψ_minus, H))
        n = norm(ψ_minus)
        println("Step $step: E = $E, norm = $n")
    end
    
    println("\n=== Testing -im * dt with normalization ===")
    ψ_norm = copy(ψ)
    for step in 1:num_steps
        ψ_norm, _ = timestep(ψ_norm, H, 0, -im * dt, TDVP2(; trscheme=combined_trunc))
        normalize!(ψ_norm)
        E = real(expectation_value(ψ_norm, H))
        println("Step $step: E = $E, norm = $(norm(ψ_norm))")
    end
    
    println("\n=== Testing dt with imaginary_evolution=true ===")
    ψ_imt = copy(ψ)
    for step in 1:num_steps
        ψ_imt, _ = timestep(ψ_imt, H, 0, dt, TDVP2(; trscheme=combined_trunc); imaginary_evolution=true)
        # normalize!(ψ_imt)
        E = real(expectation_value(ψ_imt, H))
        println("Step $step: E = $E, norm = $(norm(ψ_imt))")
    end
    
    println("\n=== Testing time_evolve with imaginary_evolution=true ===")
    ψ_imt = copy(ψ)
    t_span = collect(0:dt:num_steps*dt)
    ψ_imt, _ = time_evolve(ψ_imt, H, t_span, TDVP2(; trscheme=combined_trunc); imaginary_evolution=true)
    E = real(expectation_value(ψ_imt, H))
    println("Step $num_steps: E = $E, norm = $(norm(ψ_imt))")
    
    
    return ψ_minus, ψ_plus, ψ_norm, ψ_imt
end

# Run diagnostics
debug_imaginary_time();