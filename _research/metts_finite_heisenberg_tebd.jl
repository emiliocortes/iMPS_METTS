using MPSKitModels
using Statistics
using StatsPlots

include("../src/collapse_to_cps.jl")
include("../src/tebd.jl")
include("../src/run_METTS.jl")


d = 2 # physical dimension
D = 20 # virtual dimension
L = 100 # number of sites

chain = FiniteChain(L)
J = 1.0

# H = heisenberg_XXX(chain; J, spin=1)
H = heisenberg_XXX(chain; J, spin=1//2)

projectors_Z = [P_up, P_dn]
projectors_X = [P_pl, P_mn]


# Truncation parameters
atol = 1e-10 # Truncate singular values smaller than this
max_dim = D # Max bond dimension
combined_trunc = truncrank(max_dim) & trunctol(; atol)

 
physical_space = ComplexSpace(d)
virtual_space = ComplexSpace(D)


gate = heisenberg_gate(physical_space, dt; imaginary_time=true)
if trotter_order == 2
    gate_half = heisenberg_gate(physical_space, dt/2; imaginary_time=true)
end

#########################################################
# Imaginary time evolution parameters
beta = 4.0
beta_half = beta/2 # evolve METTS up to beta/2
dt = 0.1
t_span = collect(0:dt:beta/2)
n_steps_timeevol = length(t_span)


println("##############################################################")
N_samples = 10
N_parchains = 100

data_chains = []

for k in 1:N_parchains
    println("Starting METTS chain $k")
    logger = METTSLogger()

    # Each chain starts with a random MPS
    mps_init = FiniteMPS(L, physical_space, virtual_space) # random MPS with maximal bond dimension D
    
    # Initialize product state by collapsing a random state
    mps = deepcopy(mps_init)
    collapse_to_cps!(mps,  projectors_Z)
    
    run_METTS(mps, H, projectors_Z, projectors_X, beta, dt, N_samples, :tebd, gate, gate_half, 2; trscheme=combined_trunc, logger=logger)
    
    push!(data_chains, logger)
end

p1 = plot()
for data in data_chains
    println("Processing data chain")
    energies = data.energies
    # plot!(energies)
    scatter!(energies, ms=1, msw=0.5, msalpha=0, label=nothing)
end
energies = stack([data.energies for data in data_chains])
energies = Matrix{Float64}(energies)
errorline!(1:N_samples, energies, errorstyle=:stick, groupcolor=:blue, lw=2, label="TEBD")