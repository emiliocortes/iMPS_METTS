using MPSKitModels
using Plots

include("../src/collapse_to_cps.jl")
include("../src/tebd.jl")
include("../src/run_METTS.jl")


d = 2 # physical dimension
D = 20 # virtual dimension
L = 10 # number of sites

chain = FiniteChain(L)
J = 1.0

# H = heisenberg_XXX(chain; J, spin=1)
H = heisenberg_XXX(chain; J, spin=1//2)



# Truncation parameters
atol = 1e-8 # Truncate singular values smaller than this
max_dim = D # Max bond dimension
combined_trunc = truncrank(max_dim) & trunctol(; atol)

 
physical_space = ComplexSpace(d)
virtual_space = ComplexSpace(D)
mps_init = FiniteMPS(L, physical_space, virtual_space) # random MPS with maximal bond dimension D

p1 = plot()

#########################################################
# DMRG single-site
println("##############################################################")
println("Initializing DMRG2 calculation")
mps = deepcopy(mps_init)
# find_groundstate!(mps, H, DMRG(; maxiter=100, tol=atol))
find_groundstate!(mps, H, DMRG2(; maxiter=100, tol=1e-6, trscheme=combined_trunc))
E0 = expectation_value(mps, H)
println("E (DMRG) = $(real(E0))")
Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
Sx_sites = [real(expectation_value(mps, i=>S_x())) for i in 1:L]
plot!(Sz_sites, marker=:o, label="DMRG")
plot!(Sx_sites, marker=:o)



#########################################################
# TEBD
println("##############################################################")
println("Initializing TEBD calculation")
beta = 20.0 # high beta to project to gs
dt = 0.1
t_span = collect(0:dt:beta)
n_steps_timeevol = length(t_span)

mps = copy(mps_init)
envs = environments(mps, H)
@show norm(mps)
trotter_order = 2

gate = heisenberg_gate(physical_space, dt; imaginary_time=true)
if trotter_order == 1
    mps = tebd!(mps, physical_space, dt, n_steps_timeevol, gate; imaginary_time=true, trotter_order=1, trscheme=combined_trunc)
elseif trotter_order == 2
    gate_half = heisenberg_gate(physical_space, dt/2; imaginary_time=true)
    mps = tebd!(mps, physical_space, dt, n_steps_timeevol, gate, gate_half; imaginary_time=true, trotter_order=2, trscheme=combined_trunc)
end



E0 = expectation_value(mps, H)
println("E (TEBD) = $(real(E0))")
Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
Sx_sites = [real(expectation_value(mps, i=>S_x())) for i in 1:L]
plot!(Sz_sites, marker=:s, label="TEBD")
plot!(Sx_sites, marker=:s)

#########################################################
# Imaginary time evolution parameters
# For large beta, energies match with DRMG
beta = 10.0
beta_half = beta/2 # evolve METTS up to beta/2
dt = 0.1
t_span = collect(0:dt:beta/2)
n_steps_timeevol = length(t_span)


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

p2 = plot()

mps_t = deepcopy(mps)
energies, Sz_all, Sx_all = run_METTS(mps_t, H, projectors_Z, projectors_X, beta, dt, N_samples, :tebd, gate, gate_half, 2; trscheme=combined_trunc)
mps_t = deepcopy(mps)
energies_2, Sz_all_2, Sx_all_2 = run_METTS(mps_t, H, projectors_Z, projectors_X, beta, dt, N_samples, :tdvp)
mps_t = deepcopy(mps)
energies_3, Sz_all_3, Sx_all_3 = run_METTS(mps_t, H, projectors_Z, projectors_X, beta, dt, N_samples, :tdvp2; trscheme=combined_trunc)
plot!(energies, label="TEBD")
plot!(energies_2, label="TDVP")
plot!(energies_3, label="TDVP2")

