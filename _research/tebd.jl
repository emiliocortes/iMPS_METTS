using MPSKit
using TensorKit
using LinearAlgebra

"""
Construct a two-site time evolution gate for the Heisenberg model
H = Σᵢ (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
"""
function heisenberg_gate(V::VectorSpace, dt::Number; imaginary_time=false)
    # Create Pauli matrices
    Sx = TensorMap(zeros, ComplexF64, V ← V)
    Sy = TensorMap(zeros, ComplexF64, V ← V)
    Sz = TensorMap(zeros, ComplexF64, V ← V)
    
    Sx[1, 2] = 0.5
    Sx[2, 1] = 0.5
    Sy[1, 2] = -0.5im
    Sy[2, 1] = 0.5im
    Sz[1, 1] = 0.5
    Sz[2, 2] = -0.5
    
    # Two-site Hamiltonian
    @tensor H_bond[a, b, c, d] := Sx[a, c] * Sx[b, d] + Sy[a, c] * Sy[b, d] + Sz[a, c] * Sz[b, d]
    
    # Reshape to matrix for exponentiation
    # H_matrix = reshape(H_bond, 4, 4)
    H_matrix = reshape(H_bond.data, 4, 4)
    
    # Time evolution operator
    if imaginary_time
        U_matrix = exp(-dt * H_matrix)
    else
        U_matrix = exp(-im * dt * H_matrix)
    end
    
    # Reshape back to tensor
    U = TensorMap(U_matrix, V ⊗ V ← V ⊗ V)
    
    return U
end

"""
Apply a two-site gate to an MPS at bond (site, site+1) and truncate
"""
function apply_two_site_gate!(ψ::FiniteMPS, gate::TensorMap, site::Int; trunc_params...)
    # # Move orthogonality center to the bond
    # ψ = MPSKit.center_position(ψ, site)

    atol = get(trunc_params, :truncerr, 1e-8)
    maxdim = get(trunc_params, :truncdim, 10)
    combined_trunc = truncrank(maxdim) & trunctol(; atol)
    
    # Get the two-site tensor
    @tensor θ[a, b, c, d] := ψ.AC[site][a, b, e] * ψ.AR[site+1][e, c, d]
    
    # Apply gate
    @tensor θ_new[a, b, c, d] := gate[b, c, e, f] * θ[a, e, f, d]
    # SVD to split back into MPS form
    # U, S, V = tsvd(θ_new, ((1, 2), (3, 4)); trunc=truncerror(; rtol=get(trunc_params, :tol, 1e-10)))
    U, S, V = tsvd(θ_new, ((1, 2), (3, 4)); trunc=combined_trunc)


    # Form new AC tensors
    # AC[site] = U * sqrt(S)
    # AC[site+1] = sqrt(S) * V
    S_sqrt = sqrt(S)
    
    @tensor AC_site[a, b, c] := U[a, b, d] * S_sqrt[d, c]
    @tensor AC_site1[a, b, c] := S_sqrt[a, d] * V[d, b, c]
    
    # Fix structure with permute
    AC_site = permute(AC_site, ((1, 2), (3,)))
    AC_site1 = permute(AC_site1, ((1, 2), (3,)))
    
    # Update the MPS
    ψ.AC[site] = AC_site
    ψ.AC[site+1] = AC_site1
    
    # # Update MPS tensors
    # # Left tensor (AL form)
    # AL_new = permute(U, ((1, 2), (3,)))
    # # Center tensor  
    # @tensor AC_new[a, b, c] := S[a, d] * V[d, b, c]
    
    # # # Right tensor (AR form) 
    # # @tensor AR_new[a, b, c] := S[a, d] * V[d, b, c]
    
    # # Update the state (this is simplified - full implementation needs gauge fixing)
    # # ψ.AC[site] = AL_new * S
    # # ψ.AR[site+1] = AR_new
    # # ψ.AL[site] = AL_new
    # ψ.AC[site+1] = AC_new
    
    # 
    
    # Properly regauge
    # ψ = MPSKit.recalculate!(ψ)
    
    return ψ
end

"""
Perform one TEBD sweep (even bonds, then odd bonds)
"""
function tebd_sweep!(ψ::FiniteMPS, gates_even::Vector, gates_odd::Vector; kwargs...)
    L = length(ψ)
    
    # Apply gates to even bonds: (1,2), (3,4), (5,6), ...
    for i in 1:2:L-1
        gate_idx = div(i-1, 2) + 1
        if gate_idx <= length(gates_even)
            ψ = apply_two_site_gate!(ψ, gates_even[gate_idx], i; kwargs...)
        end
    end
    
    # Apply gates to odd bonds: (2,3), (4,5), (6,7), ...
    for i in 2:2:L-1
        gate_idx = div(i-2, 2) + 1
        if gate_idx <= length(gates_odd)
            ψ = apply_two_site_gate!(ψ, gates_odd[gate_idx], i; kwargs...)
        end
    end
    
    return ψ
end

"""
Full TEBD time evolution
"""
function tebd!(ψ::FiniteMPS, V::VectorSpace, dt::Number, num_steps::Int; 
               imaginary_time=false, trotter_order=2, truncerr=1e-8, truncdim=10)
    
    L = length(ψ)
    
    # For second-order Trotter: exp(-iHdt) ≈ exp(-iH_odd*dt/2) exp(-iH_even*dt) exp(-iH_odd*dt/2)
    if trotter_order == 2
        dt_half = dt / 2
        dt_full = dt
    else
        dt_half = dt
        dt_full = dt
    end
    
    # Create evolution gates
    gate_full = heisenberg_gate(V, dt_full; imaginary_time=imaginary_time)
    gate_half = heisenberg_gate(V, dt_half; imaginary_time=imaginary_time)
    
    # Prepare gates for even and odd bonds
    num_even = div(L-1+1, 2)  # bonds (1,2), (3,4), ...
    num_odd = div(L-1, 2)      # bonds (2,3), (4,5), ...
    
    gates_even_full = [gate_full for _ in 1:num_even]
    gates_odd_full = [gate_full for _ in 1:num_odd]
    gates_even_half = [gate_half for _ in 1:num_even]
    gates_odd_half = [gate_half for _ in 1:num_odd]
    
    # Time evolution loop
    for step in 1:num_steps
        if trotter_order == 2
            # S2: U_odd(dt/2) U_even(dt) U_odd(dt/2)
            ψ = tebd_sweep_split!(ψ, gates_odd_half, gates_even_full, gates_odd_half; truncerr=truncerr)
        else
            # S1: U_even(dt) U_odd(dt)
            ψ = tebd_sweep!(ψ, gates_even_full, gates_odd_full; truncerr=truncerr, truncdim=truncdim)
        end
        
        # Optionally: normalize and compute energy/observables
        if step % 1 == 0
            println("Step $step/$num_steps")
        end
    end
    
    return ψ
end

"""
Second-order Trotter sweep helper
"""
function tebd_sweep_split!(ψ::FiniteMPS, gates_odd_half, gates_even, gates_odd_half2; kwargs...)
    L = length(ψ)
    
    # Odd bonds (half step)
    for i in 2:2:L-1
        gate_idx = div(i-2, 2) + 1
        if gate_idx <= length(gates_odd_half)
            ψ = apply_two_site_gate!(ψ, gates_odd_half[gate_idx], i; kwargs...)
        end
    end
    
    # Even bonds (full step)
    for i in 1:2:L-1
        gate_idx = div(i-1, 2) + 1
        if gate_idx <= length(gates_even)
            ψ = apply_two_site_gate!(ψ, gates_even[gate_idx], i; kwargs...)
        end
    end
    
    # Odd bonds (half step)
    for i in 2:2:L-1
        gate_idx = div(i-2, 2) + 1
        if gate_idx <= length(gates_odd_half2)
            ψ = apply_two_site_gate!(ψ, gates_odd_half2[gate_idx], i; kwargs...)
        end
    end
    
    return ψ
end

# ============= Example Usage =============

function main()
    # Parameters
    L = 10          # Chain length
    χ = 20          # Bond dimension
    dt = 0.1        # Time step
    num_steps = 50  # Number of steps
    
    # Physical space (spin-1/2)
    V = ComplexSpace(2)
    
    # Virtual space
    D = ComplexSpace(χ)
    
    # Initialize random MPS
    ψ = FiniteMPS(rand, ComplexF64, L, V, D)
    
    println("Initial MPS created with length $L and bond dimension $χ")
    println("Starting TEBD evolution...")
    
    # Run TEBD (imaginary time for ground state search)
    ψ = tebd!(ψ, V, dt, num_steps; imaginary_time=true, trotter_order=2, tol=1e-10)
    
    println("\nTEBD evolution complete!")
    println("Final bond dimensions: ", [dim(space(ψ.AC[i], 3)) for i in 1:L-1])
    
    return ψ
end

# Run example
# ψ_final = main()
