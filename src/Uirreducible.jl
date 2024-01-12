# irreducible vertex in singlet channel
function calc_T(
    w   :: MatsubaraFrequency{Boson}, 
    v   :: MatsubaraFrequency{Fermion}, 
    vp  :: MatsubaraFrequency{Fermion},
    η_D :: P_t,
    λ_D :: λ_t,
    η_M :: P_t,
    λ_M :: λ_t,
    F0  :: Array{ComplexF64, 4},
        :: Type{ch_S}
    )   :: Array{ComplexF64, 4}

    # bare contribution 
    T = -2.0 * copy(F0)

    # SBE contributions
    w1      = w - vp - v
    η1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(η_D, 1))
    λ1_idx1 = MatsubaraFunctions.grid_index_extrp(w1, grids(λ_D, 1))
    λ1_idx2 = MatsubaraFunctions.grid_index_extrp( v, grids(λ_D, 2))
    λ1_idx3 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))
    η_D_c   = view(η_D, η1_idx, :, :, :, :)
    η_M_c   = view(η_M, η1_idx, :, :, :, :)
    λ_D_l   = view(λ_D, λ1_idx1, λ1_idx2, :, :, :, :)
    λ_M_l   = view(λ_M, λ1_idx1, λ1_idx2, :, :, :, :)
    λ_D_r   = view(λ_D, λ1_idx1, λ1_idx3, :, :, :, :)
    λ_M_r   = view(λ_M, λ1_idx1, λ1_idx3, :, :, :, :)

    @tullio T[x1p, x1, x2p, x2] += 0.5 * λ_D_l[x1p, x1, x3, x4] * η_D_c[x4, x3, x5, x6] * λ_D_r[x2, x2p, x5, x6] - 
        1.5 * λ_M_l[x1p, x1, x3, x4] * η_M_c[x4, x3, x5, x6] * λ_M_r[x2, x2p, x5, x6]

    w2      = vp - v
    v2      = w - vp
    η2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(η_D, 1))
    λ2_idx1 = MatsubaraFunctions.grid_index_extrp(w2, grids(λ_D, 1))
    λ2_idx2 = MatsubaraFunctions.grid_index_extrp(v2, grids(λ_D, 2))
    η_D_c   = view(η_D, η2_idx, :, :, :, :)
    η_M_c   = view(η_M, η2_idx, :, :, :, :)
    λ_D_l   = view(λ_D, λ2_idx1, λ1_idx2, :, :, :, :)
    λ_M_l   = view(λ_M, λ2_idx1, λ1_idx2, :, :, :, :)
    λ_D_r   = view(λ_D, λ2_idx1, λ2_idx2, :, :, :, :)
    λ_M_r   = view(λ_M, λ2_idx1, λ2_idx2, :, :, :, :)

    @tullio T[x1p, x1, x2p, x2] += 0.5 * λ_D_l[x1p, x2, x3, x4] * η_D_c[x4, x3, x5, x6] * λ_D_r[x1, x2p, x5, x6] - 
        1.5 * λ_M_l[x1p, x2, x3, x4] * η_M_c[x4, x3, x5, x6] * λ_M_r[x1, x2p, x5, x6]

    return T 
end

# irreducible vertex in triplet channel
function calc_T(
    w   :: MatsubaraFrequency{Boson}, 
    v   :: MatsubaraFrequency{Fermion}, 
    vp  :: MatsubaraFrequency{Fermion},
    η_D :: P_t,
    λ_D :: λ_t,
    η_M :: P_t,
    λ_M :: λ_t,
    F0  :: Array{ComplexF64, 4},
        :: Type{ch_T}
    )   :: Array{ComplexF64, 4}

    # bare contribution 
    T = -2.0 * copy(F0)

    # SBE contributions
    w1      = w - vp - v
    η1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(η_D, 1))
    λ1_idx1 = MatsubaraFunctions.grid_index_extrp(w1, grids(λ_D, 1))
    λ1_idx2 = MatsubaraFunctions.grid_index_extrp( v, grids(λ_D, 2))
    λ1_idx3 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))
    η_D_c   = view(η_D, η1_idx, :, :, :, :)
    η_M_c   = view(η_M, η1_idx, :, :, :, :)
    λ_D_l   = view(λ_D, λ1_idx1, λ1_idx2, :, :, :, :)
    λ_M_l   = view(λ_M, λ1_idx1, λ1_idx2, :, :, :, :)
    λ_D_r   = view(λ_D, λ1_idx1, λ1_idx3, :, :, :, :)
    λ_M_r   = view(λ_M, λ1_idx1, λ1_idx3, :, :, :, :)

    @tullio T[x1p, x1, x2p, x2] += 0.5 * λ_D_l[x1p, x1, x3, x4] * η_D_c[x4, x3, x5, x6] * λ_D_r[x2, x2p, x5, x6] + 
        0.5 * λ_M_l[x1p, x1, x3, x4] * η_M_c[x4, x3, x5, x6] * λ_M_r[x2, x2p, x5, x6]

    w2      = vp - v
    v2      = w - vp
    η2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(η_D, 1))
    λ2_idx1 = MatsubaraFunctions.grid_index_extrp(w2, grids(λ_D, 1))
    λ2_idx2 = MatsubaraFunctions.grid_index_extrp(v2, grids(λ_D, 2))
    η_D_c   = view(η_D, η2_idx, :, :, :, :)
    η_M_c   = view(η_M, η2_idx, :, :, :, :)
    λ_D_l   = view(λ_D, λ2_idx1, λ1_idx2, :, :, :, :)
    λ_M_l   = view(λ_M, λ2_idx1, λ1_idx2, :, :, :, :)
    λ_D_r   = view(λ_D, λ2_idx1, λ2_idx2, :, :, :, :)
    λ_M_r   = view(λ_M, λ2_idx1, λ2_idx2, :, :, :, :)

    @tullio T[x1p, x1, x2p, x2] += -0.5 * λ_D_l[x1p, x2, x3, x4] * η_D_c[x4, x3, x5, x6] * λ_D_r[x1, x2p, x5, x6] - 
        0.5 * λ_M_l[x1p, x2, x3, x4] * η_M_c[x4, x3, x5, x6] * λ_M_r[x1, x2p, x5, x6]

    return T 
end

# irreducible vertex in density channel
function calc_T(
    w   :: MatsubaraFrequency{Boson}, 
    v   :: MatsubaraFrequency{Fermion}, 
    vp  :: MatsubaraFrequency{Fermion},
    η_S :: P_t,
    λ_S :: λ_t,
    η_T :: P_t,
    λ_T :: λ_t,
    η_D :: P_t,
    λ_D :: λ_t,
    η_M :: P_t,
    λ_M :: λ_t,
    F0  :: Array{ComplexF64, 4},
        :: Type{ch_D}
    )   :: Array{ComplexF64, 4}

    # bare contribution 
    T = -2.0 * copy(F0)

    # SBE contributions
    w1      = w + v + vp
    η1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(η_D, 1))
    λ1_idx1 = MatsubaraFunctions.grid_index_extrp(w1, grids(λ_D, 1))
    λ1_idx2 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))
    λ1_idx3 = MatsubaraFunctions.grid_index_extrp( v, grids(λ_D, 2))
    η_S_c   = view(η_S, η1_idx, :, :, :, :)
    η_T_c   = view(η_T, η1_idx, :, :, :, :)
    λ_S_l   = view(λ_S, λ1_idx1, λ1_idx2, :, :, :, :)
    λ_T_l   = view(λ_T, λ1_idx1, λ1_idx2, :, :, :, :)
    λ_S_r   = view(λ_S, λ1_idx1, λ1_idx3, :, :, :, :)
    λ_T_r   = view(λ_T, λ1_idx1, λ1_idx3, :, :, :, :)

    @tullio T[x1p, x1, x2p, x2] += 0.5 * λ_S_l[x3, x1, x4, x2] * η_S_c[x5, x3, x6, x4] * λ_S_r[x6, x2p, x5, x1p] + 
        1.5 * λ_T_l[x3, x1, x4, x2] * η_T_c[x5, x3, x6, x4] * λ_T_r[x6, x2p, x5, x1p]

    w2      = vp - v
    v2      = w + v
    η2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(η_D, 1))
    λ2_idx1 = MatsubaraFunctions.grid_index_extrp(w2, grids(λ_D, 1))
    λ2_idx2 = MatsubaraFunctions.grid_index_extrp(v2, grids(λ_D, 2))
    η_D_c   = view(η_D, η2_idx, :, :, :, :)
    η_M_c   = view(η_M, η2_idx, :, :, :, :)
    λ_D_l   = view(λ_D, λ2_idx1, λ1_idx3, :, :, :, :)
    λ_M_l   = view(λ_M, λ2_idx1, λ1_idx3, :, :, :, :)
    λ_D_r   = view(λ_D, λ2_idx1, λ2_idx2, :, :, :, :)
    λ_M_r   = view(λ_M, λ2_idx1, λ2_idx2, :, :, :, :)

    @tullio T[x1p, x1, x2p, x2] += -0.5 * λ_D_l[x1p, x2, x3, x4] * η_D_c[x4, x3, x5, x6] * λ_D_r[x1, x2p, x5, x6] - 
        1.5 * λ_M_l[x1p, x2, x3, x4] * η_M_c[x4, x3, x5, x6] * λ_M_r[x1, x2p, x5, x6]

    return T 
end

# irreducible vertex in magnetic channel
function calc_T(
    w   :: MatsubaraFrequency{Boson}, 
    v   :: MatsubaraFrequency{Fermion}, 
    vp  :: MatsubaraFrequency{Fermion},
    η_S :: P_t,
    λ_S :: λ_t,
    η_T :: P_t,
    λ_T :: λ_t,
    η_D :: P_t,
    λ_D :: λ_t,
    η_M :: P_t,
    λ_M :: λ_t,
    F0  :: Array{ComplexF64, 4},
        :: Type{ch_M}
    )   :: Array{ComplexF64, 4}

    # bare contribution 
    T = -2.0 * copy(F0)

    # SBE contributions
    w1      = w + v + vp
    η1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(η_D, 1))
    λ1_idx1 = MatsubaraFunctions.grid_index_extrp(w1, grids(λ_D, 1))
    λ1_idx2 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))
    λ1_idx3 = MatsubaraFunctions.grid_index_extrp( v, grids(λ_D, 2))
    η_S_c   = view(η_S, η1_idx, :, :, :, :)
    η_T_c   = view(η_T, η1_idx, :, :, :, :)
    λ_S_l   = view(λ_S, λ1_idx1, λ1_idx2, :, :, :, :)
    λ_T_l   = view(λ_T, λ1_idx1, λ1_idx2, :, :, :, :)
    λ_S_r   = view(λ_S, λ1_idx1, λ1_idx3, :, :, :, :)
    λ_T_r   = view(λ_T, λ1_idx1, λ1_idx3, :, :, :, :)

    @tullio T[x1p, x1, x2p, x2] += -0.5 * λ_S_l[x3, x1, x4, x2] * η_S_c[x5, x3, x6, x4] * λ_S_r[x6, x2p, x5, x1p] + 
        0.5 * λ_T_l[x3, x1, x4, x2] * η_T_c[x5, x3, x6, x4] * λ_T_r[x6, x2p, x5, x1p]

    w2      = vp - v
    v2      = w + v
    η2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(η_D, 1))
    λ2_idx1 = MatsubaraFunctions.grid_index_extrp(w2, grids(λ_D, 1))
    λ2_idx2 = MatsubaraFunctions.grid_index_extrp(v2, grids(λ_D, 2))
    η_D_c   = view(η_D, η2_idx, :, :, :, :)
    η_M_c   = view(η_M, η2_idx, :, :, :, :)
    λ_D_l   = view(λ_D, λ2_idx1, λ1_idx3, :, :, :, :)
    λ_M_l   = view(λ_M, λ2_idx1, λ1_idx3, :, :, :, :)
    λ_D_r   = view(λ_D, λ2_idx1, λ2_idx2, :, :, :, :)
    λ_M_r   = view(λ_M, λ2_idx1, λ2_idx2, :, :, :, :)

    @tullio T[x1p, x1, x2p, x2] += -0.5 * λ_D_l[x1p, x2, x3, x4] * η_D_c[x4, x3, x5, x6] * λ_D_r[x1, x2p, x5, x6] + 
        0.5 * λ_M_l[x1p, x2, x3, x4] * η_M_c[x4, x3, x5, x6] * λ_M_r[x1, x2p, x5, x6]

    return T 
end