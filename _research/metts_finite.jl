

d = 2 # physical dimension
D = 6 # virtual dimension
L = 20 # number of sites

chain = FiniteChain(L)
J = 1.0
g = 0.5

H = transverse_field_ising(chain; J, g)


physical_space = ComplexSpace(d)
virtual_space = ComplexSpace(D)
# mps = FiniteMPS(L, physical_space, virtual_space) # random MPS with maximal bond dimension D
# find_groundstate!(mps, H, DMRG(; maxiter=10))
# E0 = expectation_value(mps, H)
# println("<mps|H|mps> = $(real(E0))")


# # Testing imaginary time evolution
# # For large beta, energies match with DRMG
# beta = 1.0
# beta_half = beta/2
# t_span = Vector(0:0.1:beta_half)

# envs = environments(mps, H)

# mps = FiniteMPS(L, physical_space, virtual_space) # random MPS with maximal bond dimension D
# # mps, envs = time_evolve(mps, H, t_span, TDVP(), envs; imaginary_evolution=true, verbosity=1)
# mps, envs = time_evolve(mps, H, t_span, TDVP(), envs; imaginary_evolution=true, verbosity=1)

# E0 = expectation_value(mps, H)
# println("<mps|H|mps> = $(real(E0))")



#### Sampling
println("Starting METTS sampling")
N_samples = 100


# Initialize product state by collapsing a random state
mps = FiniteMPS(L, physical_space, virtual_space) # random MPS with maximal bond dimension D
# envs = environments(mps, H)
projectors_Z = [P_up, P_dn]
projectors_X = [P_pl, P_mn]

collapse_to_cps!(mps,  projectors_Z)


function run_METTS(mps, H, projectors_1, projectors_2, beta, N_samples)
    beta_half = beta/2
    dt = 0.1
    t_span = Vector(0:dt:beta_half)
    n_steps_timeevol = length(t_span)

    energies = []
    Sz_all = []
    
    envs = environments(mps, H)

    for i in 1:N_samples
        println("METTS step $(i)")
        if i%2 == 0
            projectors = projectors_1
        else
            projectors =  projectors_2
        end
        
        # Imaginary time-evolution
        # mps, envs = time_evolve(mps, H, t_span, TDVP(), envs; imaginary_evolution=true, verbosity=1)


        mps = tebd!(mps, physical_space, dt, n_steps_timeevol; imaginary_time=true, trotter_order=1, truncerr=1e-8, truncdim=20)


        # Measure expectation values
        E0 = expectation_value(mps, H)
        Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
        Sz_tot = sum(Sz_sites)/L
    
        push!(energies, real(E0))
        push!(Sz_all, Sz_sites)
    
        # Collapse onto a new product state
        collapse_to_cps!(mps, projectors)
        # Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
        # print("Sz of the Product state")
        # @show Sz_sites

    end
    return energies, Sz_all

end


