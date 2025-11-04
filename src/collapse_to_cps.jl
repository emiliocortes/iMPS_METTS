using TensorKit
using MPSKit

d = 2
V = ComplexSpace(d)

# Create a projector onto |↑⟩ (first basis state)
P_up = TensorMap(zeros, ComplexF64, V ← V)
P_up[1, 1] = 1.0

# Create a projector onto |↓⟩ (second basis state)
P_dn = TensorMap(zeros, ComplexF64, V ← V)
P_dn[2, 2] = 1.0

# Create a projector onto |+x⟩ (first basis state)
P_pl = TensorMap(zeros, ComplexF64, V ← V)
# P_pl[1, 2] = 1.0
# P_pl[2, 1] = 1.0
P_pl[1,1] = P_pl[2,2] = 0.5
P_pl[1,2] = P_pl[2,1] = 0.5

# Create a projector onto |-x⟩ (second basis state)
P_mn = TensorMap(zeros, ComplexF64, V ← V)
# P_mn[1, 2] = 1.0im
# P_mn[2, 1] = -1.0im
P_mn[1,1] = P_mn[2,2] = 0.5
P_mn[1,2] = P_mn[2,1] = -0.5
#

# Sz = 1/2 * [1 0; 0 -1]
# Sp = [0 1; 0 0]
# Sm = [0 0; 1 0]
# Sx = 1/2 * [0 1; 1 0]
# Sy = 1/2 * [0 -1im; 1im 0]
# Id = [1 0; 0 1]

# P_plus_site = Id/2 + Sz
# P_minus_site = Id/2 - Sz



# P_pl[:,:] = Id/2 + Sx
# P_mn[:,:] = Id/2 - Sx

# P_up[:,:] = Id/2 + Sz
# P_dn[:,:] = Id/2 - Sz



function apply_projector(AC::TensorMap, P::TensorMap)
    # AC has structure (χL ⊗ V_in) ← χR
    # P has structure V_out ← V_in
    # Want result: (χL ⊗ V_out) ← χR
    
    χL = space(AC, 1)
    χR = space(AC, 3)
    # V_in = domain(P, 1)
    # V_out = codomain(P, 1)
    V_in = domain(P)
    V_out = codomain(P)
    
    # Do contraction (result will have wrong structure)
    @tensor temp[χL, p_out, χR] := P[p_out, p_in] * AC[χL, p_in, χR]
    
    # # Fix the structure using permute
    # result = permute(temp, (1, 2), (3,))

    # Then fix structure:
    # result = TensorMap(temp.data, χL ⊗ codomain(P) ← χR)
    result = TensorMap(temp.data, codomain(AC,1) ⊗ codomain(P) ← domain(AC))
    # result = result/norm(result)
    return result
end


function project_site!(mps::FiniteMPS, P::TensorMap, site::Int)
    # todo: mode orthogonality center
    # not necessary? (done automatically)
    AC = mps.AC[site]
    AC_proj = apply_projector(AC, P)
    AC_proj = AC_proj/norm(AC_proj) # normalize to mantain mps normalized
    mps.AC[site] = AC_proj
    return mps
end

function projection_probability(mps::FiniteMPS, P::TensorMap, site::Int)
    # The probability to project onto specific state is given by the expectation value of the projector
    p = real(expectation_value(mps, site=>P))
    return p
end

function collapse_to_cps!(mps::FiniteMPS, projectors::Vector{<:TensorMap})
    L = length(mps)
    for i in 1:L
        r = rand()
        P1, P2 = projectors

        prob = projection_probability(mps, P1, i)
        if r <= prob
            project_site!(mps, P1, i)
        else
            project_site!(mps, P2, i)
        end
    end
    return mps
end
