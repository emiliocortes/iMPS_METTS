using TensorOperations
using TensorKit
using MPSKit
using LinearAlgebra: norm
using Plots

d = 2 # physical dimension
D = 10 # virtual dimension
L = 50 # number of sites

mps = FiniteMPS(L, ComplexSpace(d), ComplexSpace(D)) # random MPS with maximal bond dimension D



J = 1.0
# g = 0.9
lattice = fill(ComplexSpace(2), L)
X = TensorMap(ComplexF64[0 1; 1 0], ComplexSpace(2), ComplexSpace(2))
Z = TensorMap(ComplexF64[1 0; 0 -1], space(X))
# H = FiniteMPOHamiltonian(lattice, (i, i+1) => -J * X ⊗ X for i in 1:length(lattice)-1) +
#     FiniteMPOHamiltonian(lattice, (i,) => - g * Z for i in 1:length(lattice))
# find_groundstate!(mps, H, DMRG(; maxiter=10))
# E0 = expectation_value(mps, H)
# println("<mps|H|mps> = $real(E0)")


energies = []
Sz_all = []

for g in 0.1:0.1:2
    # H = FiniteMPOHamiltonian(lattice, (i, i+1) => -J * X ⊗ X for i in 1:length(lattice)-1) +
    #     FiniteMPOHamiltonian(lattice, (i,) => - g * Z for i in 1:length(lattice))
    H = transverse_field_ising(; g=g)
    H = periodic_boundary_conditions(H, L)
    find_groundstate!(mps, H, DMRG(; maxiter=10))
    E0 = expectation_value(mps, H)
    Sz_sites = [real(expectation_value(mps, i=>S_z())) for i in 1:L]
    println("<mps|H|mps> = $real(E0)")
    push!(energies, real(E0))
    push!(Sz_all, Sz_sites)
end



plot(sum.(Sz_all)/L)

# plot(energies)
