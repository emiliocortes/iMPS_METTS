using TensorOperations
using TensorKit
using MPSKit
using LinearAlgebra: norm

using Random

seed = 1234
Random.seed!(seed)


d = 2 # physical dimension
D = 5 # virtual dimension
mps = InfiniteMPS(d, D) # random MPS


J = 1.0
g = 0.5
lattice = PeriodicVector([ComplexSpace(2)])
X = TensorMap(ComplexF64[0 1; 1 0], ComplexSpace(2), ComplexSpace(2))
Z = TensorMap(ComplexF64[1 0; 0 -1], space(X))
H = InfiniteMPOHamiltonian(lattice, (1, 2) => -J * X ⊗ X, (1,) => - g * Z)
# mps, = find_groundstate(mps, H, IDMRG(; maxiter=100))
mps, = find_groundstate(mps, H, VUMPS(; maxiter=100))
E0 = expectation_value(mps, H)
println("<mps|H|mps> = $(sum(real(E0)) / length(mps))")




