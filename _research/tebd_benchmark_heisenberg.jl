
d = 2 # physical dimension
D = 30 # virtual dimension
L = 10 # number of sites

chain = FiniteChain(L)
J = 1.0

# H = transverse_field_ising(chain; J, g)
H = heisenberg_XXX(chain; J, spin=1//2)


physical_space = ComplexSpace(d)
virtual_space = ComplexSpace(D)
mps_init = FiniteMPS(L, physical_space, virtual_space) # random MPS with maximal bond dimension D


# Truncation parameters
atol = 1e-7 # Truncate singular values smaller than this
max_dim = D # Max bond dimension
combined_trunc = truncrank(max_dim) & trunctol(; atol)

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
# plot!(Sx_sites, marker=:o)

# #########################################################
# # DMRG two-site
println("##############################################################")
println("Initializing DMRG2 calculation")
mps_init = FiniteMPS(L, physical_space, virtual_space) # random MPS with maximal bond dimension D
mps = deepcopy(mps_init)
find_groundstate!(mps, H, DMRG2(; maxiter=100, tol=atol, trscheme=combined_trunc))
E0 = expectation_value(mps, H)
println("E (DMRG) = $(real(E0))")
Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
Sx_sites = [real(expectation_value(mps, i=>S_x())) for i in 1:L]
plot!(Sz_sites, marker=:o, label="DMRG2")
# plot!(Sx_sites, marker=:o)


#########################################################
# Imaginary time evolution parameters
# For large beta, energies match with DRMG
beta = 20.0
dt = 0.05
t_span = Vector(0:dt:beta)
n_steps_timeevol = length(t_span)


# #########################################################
# # TDVP single-site
println("##############################################################")
println("Initializing TDVP calculation")
mps = deepcopy(mps_init)
envs = environments(mps, H)
mps, envs = time_evolve(mps, H, t_span, TDVP(), envs; imaginary_evolution=true, verbosity=1)
E0 = expectation_value(mps, H)
println("E (TDVP) = $(real(E0))")
Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
Sx_sites = [real(expectation_value(mps, i=>S_x())) for i in 1:L]
plot!(Sz_sites, marker=:x, label="TDVP")
# plot!(Sx_sites, marker=:x)

# #########################################################
# # TDVP two-site
println("##############################################################")
println("Initializing TDVP2 calculation")
mps = deepcopy(mps_init)
envs = environments(mps, H)
mps, envs = time_evolve(mps, H, t_span, TDVP2(; trscheme=combined_trunc), envs; imaginary_evolution=true, verbosity=1)
E0 = expectation_value(mps, H)
println("E (TDVP) = $(real(E0))")
Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
Sx_sites = [real(expectation_value(mps, i=>S_x())) for i in 1:L]
plot!(Sz_sites, marker=:x, label="TDVP2")
# plot!(Sx_sites, marker=:x)


#########################################################
# TEBD
println("##############################################################")
println("Initializing TEBD calculation")
mps = copy(mps_init)
envs = environments(mps, H)
@show norm(mps)
trotter_order = 2

gate = heisenberg_gate(physical_space, dt; imaginary_time=true)
if trotter_order == 1
    mps = tebd!(mps, physical_space, dt, n_steps_timeevol, gate; imaginary_time=true, trotter_order=1, truncerr=atol, truncdim=max_dim)
elseif trotter_order == 2
    gate_half = heisenberg_gate(physical_space, dt/2; imaginary_time=true)
    mps = tebd!(mps, physical_space, dt, n_steps_timeevol, gate, gate_half; imaginary_time=true, trotter_order=2, truncerr=atol, truncdim=max_dim)
end
E0 = expectation_value(mps, H)
println("E (TEBD) = $(real(E0))")
Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
Sx_sites = [real(expectation_value(mps, i=>S_x())) for i in 1:L]
plot!(Sz_sites, marker=:s, label="TEBD")
# plot!(Sx_sites, marker=:s)

