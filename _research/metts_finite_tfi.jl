using MPSKitModels
using Plots

d = 2 # physical dimension
D = 10 # virtual dimension
L = 20 # number of sites

chain = FiniteChain(L)
J = 1.0
g = 0.5
# g = 0.0

H = transverse_field_ising(chain; J, g)
# H = heisenberg_XXX(chain; J, spin=1//2)



# Truncation parameters
atol = 1e-7 # Truncate singular values smaller than this
max_dim = D # Max bond dimension
combined_trunc = truncrank(max_dim) & trunctol(; atol)

 
#########################################################
# Imaginary time evolution parameters
# For large beta, energies match with DRMG
beta = 20.0
beta_half = beta/2 # evolve METTS up to beta/2
dt = 0.1
t_span = Vector(0:dt:beta/2)
n_steps_timeevol = length(t_span)

physical_space = ComplexSpace(d)
virtual_space = ComplexSpace(D)
mps_init = FiniteMPS(L, physical_space, virtual_space) # random MPS with maximal bond dimension D


#########################################################
# DMRG single-site
println("##############################################################")
println("Initializing DMRG calculation")
mps = deepcopy(mps_init)
find_groundstate!(mps, H, DMRG(; maxiter=100, tol=atol))
E0 = expectation_value(mps, H)
println("E (DMRG) = $(real(E0))")
Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
Sx_sites = [real(expectation_value(mps, i=>S_x())) for i in 1:L]
plot(Sz_sites, marker=:o, label="DMRG")
plot!(Sx_sites, marker=:o)



#########################################################
# TEBD
println("##############################################################")
println("Initializing TEBD calculation")
mps = copy(mps_init)
envs = environments(mps, H)
@show norm(mps)
trotter_order = 2

gate = tfim_gate(physical_space, dt, J, g; imaginary_time=true)
if trotter_order == 1
    mps = tebd!(mps, physical_space, dt, n_steps_timeevol, gate; imaginary_time=true, trotter_order=1, truncerr=atol, truncdim=max_dim)
elseif trotter_order == 2
    gate_half = tfim_gate(physical_space, dt/2, J, g; imaginary_time=true)
    mps = tebd!(mps, physical_space, dt, n_steps_timeevol, gate, gate_half; imaginary_time=true, trotter_order=2, truncerr=atol, truncdim=max_dim)
end



E0 = expectation_value(mps, H)
println("E (TEBD) = $(real(E0))")
Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
Sx_sites = [real(expectation_value(mps, i=>S_x())) for i in 1:L]
plot!(Sz_sites, marker=:s, label="TEBD")
plot!(Sx_sites, marker=:s)


#### Sampling
println("##############################################################")
N_samples = 100
println("Starting METTS sampling for $(N_samples) samples")


# Initialize product state by collapsing a random state
mps = copy(mps_init)
@show norm(mps)
# envs = environments(mps, H)
projectors_Z = [P_up, P_dn]
projectors_X = [P_pl, P_mn]

collapse_to_cps!(mps,  projectors_Z)
@show norm(mps)


function run_METTS(mps, H, projectors_1, projectors_2, beta, N_samples)
    beta_half = beta/2
    dt = 0.05
    t_span = Vector(0:dt:beta_half)
    n_steps_timeevol = length(t_span)
    trotter_order = 2

    energies = []
    Sz_all = []
    Sx_all = []
    
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


        gate = tfim_gate(physical_space, dt, J, g; imaginary_time=true)
        if trotter_order == 1
            mps = tebd!(mps, physical_space, dt, n_steps_timeevol, gate; imaginary_time=true, trotter_order=1, truncerr=atol, truncdim=max_dim)
        elseif trotter_order == 2
            gate_half = tfim_gate(physical_space, dt/2, J, g; imaginary_time=true)
            mps = tebd!(mps, physical_space, dt, n_steps_timeevol, gate, gate_half; imaginary_time=true, trotter_order=2, truncerr=atol, truncdim=max_dim)
        end

        # Measure expectation values
        E0 = expectation_value(mps, H)
        Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
        Sx_sites = [real(expectation_value(mps, i=>S_x())) for i in 1:L]
        Sz_tot = sum(Sz_sites)/L
    
        push!(energies, real(E0))
        push!(Sz_all, Sz_sites)
        push!(Sx_all, Sx_sites)
    
        # Collapse onto a new product state
        collapse_to_cps!(mps, projectors)
        # Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
        # print("Sz of the Product state")
        # @show Sz_sites

    end
    return energies, Sz_all, Sx_all

end


