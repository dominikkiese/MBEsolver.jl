# irreducible vertex in singlet channel
function calc_T(
    w   :: MatsubaraFrequency, 
    v   :: MatsubaraFrequency, 
    vp  :: MatsubaraFrequency,
    η_S :: MatsubaraFunction{1, 1, 2, Float64},
    λ_S :: MatsubaraFunction{2, 1, 3, Float64},
    η_D :: MatsubaraFunction{1, 1, 2, Float64},
    λ_D :: MatsubaraFunction{2, 1, 3, Float64},
    η_M :: MatsubaraFunction{1, 1, 2, Float64},
    λ_M :: MatsubaraFunction{2, 1, 3, Float64},
    M_S :: MatsubaraFunction{3, 1, 4, Float64},
    M_T :: MatsubaraFunction{3, 1, 4, Float64},
    M_D :: MatsubaraFunction{3, 1, 4, Float64},
    M_M :: MatsubaraFunction{3, 1, 4, Float64},
    U   :: Float64,
        :: Type{ch_S}
    )   :: Float64

    # bare contribution 
    T = -4.0 * U 

    # SBE contributions
    w1      = w - vp - v
    η1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(η_D, 1))
    λ1_idx1 = MatsubaraFunctions.grid_index_extrp(w1, grids(λ_D, 1))
    λ1_idx2 = MatsubaraFunctions.grid_index_extrp( v, grids(λ_D, 2))
    λ1_idx3 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))

    w2      = vp - v
    v2      = w - vp
    η2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(η_D, 1))
    λ2_idx1 = MatsubaraFunctions.grid_index_extrp(w2, grids(λ_D, 1))
    λ2_idx2 = MatsubaraFunctions.grid_index_extrp(v2, grids(λ_D, 2))

    T += +0.5 * λ_D[λ1_idx1, λ1_idx2, 1] * η_D[η1_idx, 1] * λ_D[λ1_idx1, λ1_idx3, 1];
    T += -1.5 * λ_M[λ1_idx1, λ1_idx2, 1] * η_M[η1_idx, 1] * λ_M[λ1_idx1, λ1_idx3, 1]; 
    T += +0.5 * λ_D[λ2_idx1, λ1_idx2, 1] * η_D[η2_idx, 1] * λ_D[λ2_idx1, λ2_idx2, 1];
    T += -1.5 * λ_M[λ2_idx1, λ1_idx2, 1] * η_M[η2_idx, 1] * λ_M[λ2_idx1, λ2_idx2, 1];

    # MBE contributions
    w_idx  = MatsubaraFunctions.grid_index_extrp( w, grids(M_S, 1))
    v_idx  = MatsubaraFunctions.grid_index_extrp( v, grids(M_S, 2))
    vp_idx = MatsubaraFunctions.grid_index_extrp(vp, grids(M_S, 2))

    w1_idx = MatsubaraFunctions.grid_index_extrp(w1, grids(M_S, 1))
    w2_idx = MatsubaraFunctions.grid_index_extrp(w2, grids(M_S, 1))
    v2_idx = MatsubaraFunctions.grid_index_extrp(v2, grids(M_S, 2))

    T += M_S[w_idx, v_idx, vp_idx, 1]
    T += +0.5 * M_D[w1_idx, v_idx, vp_idx, 1]
    T += -1.5 * M_M[w1_idx, v_idx, vp_idx, 1]
    T += +0.5 * M_D[w2_idx, v_idx, v2_idx, 1]
    T += -1.5 * M_M[w2_idx, v_idx, v2_idx, 1]

    return T 
end

# irreducible vertex in triplet channel
function calc_T(
    w   :: MatsubaraFrequency, 
    v   :: MatsubaraFrequency, 
    vp  :: MatsubaraFrequency,
    η_S :: MatsubaraFunction{1, 1, 2, Float64},
    λ_S :: MatsubaraFunction{2, 1, 3, Float64},
    η_D :: MatsubaraFunction{1, 1, 2, Float64},
    λ_D :: MatsubaraFunction{2, 1, 3, Float64},
    η_M :: MatsubaraFunction{1, 1, 2, Float64},
    λ_M :: MatsubaraFunction{2, 1, 3, Float64},
    M_S :: MatsubaraFunction{3, 1, 4, Float64},
    M_T :: MatsubaraFunction{3, 1, 4, Float64},
    M_D :: MatsubaraFunction{3, 1, 4, Float64},
    M_M :: MatsubaraFunction{3, 1, 4, Float64},
    U   :: Float64,
        :: Type{ch_T}
    )   :: Float64

    # bare contribution 
    T = 0.0

    # SBE contributions
    w1      = w - vp - v
    η1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(η_D, 1))
    λ1_idx1 = MatsubaraFunctions.grid_index_extrp(w1, grids(λ_D, 1))
    λ1_idx2 = MatsubaraFunctions.grid_index_extrp( v, grids(λ_D, 2))
    λ1_idx3 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))

    w2      = vp - v
    v2      = w - vp
    η2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(η_D, 1))
    λ2_idx1 = MatsubaraFunctions.grid_index_extrp(w2, grids(λ_D, 1))
    λ2_idx2 = MatsubaraFunctions.grid_index_extrp(v2, grids(λ_D, 2))

    T += +0.5 * λ_D[λ1_idx1, λ1_idx2, 1] * η_D[η1_idx, 1] * λ_D[λ1_idx1, λ1_idx3, 1]
    T += +0.5 * λ_M[λ1_idx1, λ1_idx2, 1] * η_M[η1_idx, 1] * λ_M[λ1_idx1, λ1_idx3, 1]
    T += -0.5 * λ_D[λ2_idx1, λ1_idx2, 1] * η_D[η2_idx, 1] * λ_D[λ2_idx1, λ2_idx2, 1]
    T += -0.5 * λ_M[λ2_idx1, λ1_idx2, 1] * η_M[η2_idx, 1] * λ_M[λ2_idx1, λ2_idx2, 1]

    # MBE contributions
    w_idx  = MatsubaraFunctions.grid_index_extrp( w, grids(M_S, 1))
    v_idx  = MatsubaraFunctions.grid_index_extrp( v, grids(M_S, 2))
    vp_idx = MatsubaraFunctions.grid_index_extrp(vp, grids(M_S, 2))

    w1_idx = MatsubaraFunctions.grid_index_extrp(w1, grids(M_S, 1))
    w2_idx = MatsubaraFunctions.grid_index_extrp(w2, grids(M_S, 1))
    v2_idx = MatsubaraFunctions.grid_index_extrp(v2, grids(M_S, 2))

    T += M_T[w_idx, v_idx, vp_idx, 1]
    T += +0.5 * M_D[w1_idx, v_idx, vp_idx, 1]
    T += +0.5 * M_M[w1_idx, v_idx, vp_idx, 1]
    T += -0.5 * M_D[w2_idx, v_idx, v2_idx, 1]
    T += -0.5 * M_M[w2_idx, v_idx, v2_idx, 1]

    return T 
end

# inplace calculation of irreducible vertices in pp channel
function calc_T_pp!(
    T_S :: MatsubaraFunction{3, 1, 4, Float64},
    T_T :: MatsubaraFunction{3, 1, 4, Float64},
    η_S :: MatsubaraFunction{1, 1, 2, Float64},
    λ_S :: MatsubaraFunction{2, 1, 3, Float64},
    η_D :: MatsubaraFunction{1, 1, 2, Float64},
    λ_D :: MatsubaraFunction{2, 1, 3, Float64},
    η_M :: MatsubaraFunction{1, 1, 2, Float64},
    λ_M :: MatsubaraFunction{2, 1, 3, Float64},
    M_S :: MatsubaraFunction{3, 1, 4, Float64},
    M_T :: MatsubaraFunction{3, 1, 4, Float64},
    M_D :: MatsubaraFunction{3, 1, 4, Float64},
    M_M :: MatsubaraFunction{3, 1, 4, Float64},
    U   :: Float64
    )   :: Nothing

    Threads.@threads for vp in grids(T_S, 3)
        λ1_idx3 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))
        vp_idx  = MatsubaraFunctions.grid_index_extrp(vp, grids(M_S, 2))

        for v in grids(T_S, 2)
            w2      = vp - v
            λ1_idx2 = MatsubaraFunctions.grid_index_extrp( v, grids(λ_D, 2))
            η2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(η_D, 1))
            λ2_idx1 = MatsubaraFunctions.grid_index_extrp(w2, grids(λ_D, 1))
            v_idx   = MatsubaraFunctions.grid_index_extrp( v, grids(M_S, 2))
            w2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(M_S, 1))

            for w in grids(T_S, 1)
                w1      = w - vp - v
                v2      = w - vp
                η1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(η_D, 1))
                λ1_idx1 = MatsubaraFunctions.grid_index_extrp(w1, grids(λ_D, 1))
                λ2_idx2 = MatsubaraFunctions.grid_index_extrp(v2, grids(λ_D, 2))
                w_idx   = MatsubaraFunctions.grid_index_extrp( w, grids(M_S, 1))
                w1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(M_S, 1))
                v2_idx  = MatsubaraFunctions.grid_index_extrp(v2, grids(M_S, 2))

                # compute SBE vertices
                p1 = λ_D[λ1_idx1, λ1_idx2, 1] * η_D[η1_idx, 1] * λ_D[λ1_idx1, λ1_idx3, 1]
                p2 = λ_M[λ1_idx1, λ1_idx2, 1] * η_M[η1_idx, 1] * λ_M[λ1_idx1, λ1_idx3, 1]
                p3 = λ_D[λ2_idx1, λ1_idx2, 1] * η_D[η2_idx, 1] * λ_D[λ2_idx1, λ2_idx2, 1]
                p4 = λ_M[λ2_idx1, λ1_idx2, 1] * η_M[η2_idx, 1] * λ_M[λ2_idx1, λ2_idx2, 1]

                # compute MBE vertices
                m1 = M_D[w1_idx, v_idx, vp_idx, 1]
                m2 = M_M[w1_idx, v_idx, vp_idx, 1]
                m3 = M_D[w2_idx, v_idx, v2_idx, 1]
                m4 = M_M[w2_idx, v_idx, v2_idx, 1]

                T_S[w, v, vp] = -4.0 * U + M_S[w_idx, v_idx, vp_idx, 1] + 0.5 * (p1 + m1 + p3 + m3) - 1.5 * (p2 + m2 + p4 + m4)
                T_T[w, v, vp] = M_T[w_idx, v_idx, vp_idx, 1] + 0.5 * (m1 + p1 + m2 + p2) - 0.5 * (m3 + p3 + m4 + p4)
            end
        end
    end

    return nothing 
end

# irreducible vertex in density channel
function calc_T(
    w   :: MatsubaraFrequency, 
    v   :: MatsubaraFrequency, 
    vp  :: MatsubaraFrequency,
    η_S :: MatsubaraFunction{1, 1, 2, Float64},
    λ_S :: MatsubaraFunction{2, 1, 3, Float64},
    η_D :: MatsubaraFunction{1, 1, 2, Float64},
    λ_D :: MatsubaraFunction{2, 1, 3, Float64},
    η_M :: MatsubaraFunction{1, 1, 2, Float64},
    λ_M :: MatsubaraFunction{2, 1, 3, Float64},
    M_S :: MatsubaraFunction{3, 1, 4, Float64},
    M_T :: MatsubaraFunction{3, 1, 4, Float64},
    M_D :: MatsubaraFunction{3, 1, 4, Float64},
    M_M :: MatsubaraFunction{3, 1, 4, Float64},
    U   :: Float64,
        :: Type{ch_D}
    )   :: Float64

    # bare contribution
    T = -2.0 * U 

    # SBE contributions
    w1      = w + v + vp
    η1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(η_D, 1))
    λ1_idx1 = MatsubaraFunctions.grid_index_extrp(w1, grids(λ_D, 1))
    λ1_idx2 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))
    λ1_idx3 = MatsubaraFunctions.grid_index_extrp( v, grids(λ_D, 2))

    w2      = vp - v
    v2      = w + v
    η2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(η_D, 1))
    λ2_idx1 = MatsubaraFunctions.grid_index_extrp(w2, grids(λ_D, 1))
    λ2_idx2 = MatsubaraFunctions.grid_index_extrp(v2, grids(λ_D, 2))

    T += +0.5 * λ_S[λ1_idx1, λ1_idx2, 1] * η_S[η1_idx, 1] * λ_S[λ1_idx1, λ1_idx3, 1]
    T += -0.5 * λ_D[λ2_idx1, λ1_idx3, 1] * η_D[η2_idx, 1] * λ_D[λ2_idx1, λ2_idx2, 1]
    T += -1.5 * λ_M[λ2_idx1, λ1_idx3, 1] * η_M[η2_idx, 1] * λ_M[λ2_idx1, λ2_idx2, 1]

    # MBE contributions
    w_idx  = MatsubaraFunctions.grid_index_extrp( w, grids(M_S, 1))
    v_idx  = MatsubaraFunctions.grid_index_extrp( v, grids(M_S, 2))
    vp_idx = MatsubaraFunctions.grid_index_extrp(vp, grids(M_S, 2))

    w1_idx = MatsubaraFunctions.grid_index_extrp(w1, grids(M_S, 1))
    w2_idx = MatsubaraFunctions.grid_index_extrp(w2, grids(M_S, 1))
    v2_idx = MatsubaraFunctions.grid_index_extrp(v2, grids(M_S, 2))

    T += M_D[w_idx, v_idx, vp_idx, 1]
    T += +0.5 * M_S[w1_idx, v_idx, vp_idx, 1]
    T += +1.5 * M_T[w1_idx, v_idx, vp_idx, 1]
    T += -0.5 * M_D[w2_idx, v_idx, v2_idx, 1]
    T += -1.5 * M_M[w2_idx, v_idx, v2_idx, 1]

    return T 
end

# irreducible vertex in magnetic channel
function calc_T(
    w   :: MatsubaraFrequency, 
    v   :: MatsubaraFrequency, 
    vp  :: MatsubaraFrequency,
    η_S :: MatsubaraFunction{1, 1, 2, Float64},
    λ_S :: MatsubaraFunction{2, 1, 3, Float64},
    η_D :: MatsubaraFunction{1, 1, 2, Float64},
    λ_D :: MatsubaraFunction{2, 1, 3, Float64},
    η_M :: MatsubaraFunction{1, 1, 2, Float64},
    λ_M :: MatsubaraFunction{2, 1, 3, Float64},
    M_S :: MatsubaraFunction{3, 1, 4, Float64},
    M_T :: MatsubaraFunction{3, 1, 4, Float64},
    M_D :: MatsubaraFunction{3, 1, 4, Float64},
    M_M :: MatsubaraFunction{3, 1, 4, Float64},
    U   :: Float64,
        :: Type{ch_M}
    )   :: Float64

    # bare contribution
    T = +2.0 * U 

    # SBE contributions
    w1      = w + v + vp
    η1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(η_D, 1))
    λ1_idx1 = MatsubaraFunctions.grid_index_extrp(w1, grids(λ_D, 1))
    λ1_idx2 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))
    λ1_idx3 = MatsubaraFunctions.grid_index_extrp( v, grids(λ_D, 2))

    w2      = vp - v
    v2      = w + v
    η2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(η_D, 1))
    λ2_idx1 = MatsubaraFunctions.grid_index_extrp(w2, grids(λ_D, 1))
    λ2_idx2 = MatsubaraFunctions.grid_index_extrp(v2, grids(λ_D, 2))

    T += -0.5 * λ_S[λ1_idx1, λ1_idx2, 1] * η_S[η1_idx, 1] * λ_S[λ1_idx1, λ1_idx3, 1]
    T += -0.5 * λ_D[λ2_idx1, λ1_idx3, 1] * η_D[η2_idx, 1] * λ_D[λ2_idx1, λ2_idx2, 1]
    T += +0.5 * λ_M[λ2_idx1, λ1_idx3, 1] * η_M[η2_idx, 1] * λ_M[λ2_idx1, λ2_idx2, 1]

    # MBE contributions
    w_idx  = MatsubaraFunctions.grid_index_extrp( w, grids(M_S, 1))
    v_idx  = MatsubaraFunctions.grid_index_extrp( v, grids(M_S, 2))
    vp_idx = MatsubaraFunctions.grid_index_extrp(vp, grids(M_S, 2))

    w1_idx = MatsubaraFunctions.grid_index_extrp(w1, grids(M_S, 1))
    w2_idx = MatsubaraFunctions.grid_index_extrp(w2, grids(M_S, 1))
    v2_idx = MatsubaraFunctions.grid_index_extrp(v2, grids(M_S, 2))

    T += M_M[w_idx, v_idx, vp_idx, 1]
    T += -0.5 * M_S[w1_idx, v_idx, vp_idx, 1]
    T += +0.5 * M_T[w1_idx, v_idx, vp_idx, 1]
    T += -0.5 * M_D[w2_idx, v_idx, v2_idx, 1]
    T += +0.5 * M_M[w2_idx, v_idx, v2_idx, 1]

    return T 
end

# inplace calculation of irreducible vertices in ph channel
function calc_T_ph!(
    T_D :: MatsubaraFunction{3, 1, 4, Float64},
    T_M :: MatsubaraFunction{3, 1, 4, Float64},
    η_S :: MatsubaraFunction{1, 1, 2, Float64},
    λ_S :: MatsubaraFunction{2, 1, 3, Float64},
    η_D :: MatsubaraFunction{1, 1, 2, Float64},
    λ_D :: MatsubaraFunction{2, 1, 3, Float64},
    η_M :: MatsubaraFunction{1, 1, 2, Float64},
    λ_M :: MatsubaraFunction{2, 1, 3, Float64},
    M_S :: MatsubaraFunction{3, 1, 4, Float64},
    M_T :: MatsubaraFunction{3, 1, 4, Float64},
    M_D :: MatsubaraFunction{3, 1, 4, Float64},
    M_M :: MatsubaraFunction{3, 1, 4, Float64},
    U   :: Float64
    )   :: Nothing

    Threads.@threads for vp in grids(T_D, 3)
        λ1_idx2 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))
        vp_idx  = MatsubaraFunctions.grid_index_extrp(vp, grids(M_S, 2))

        for v in grids(T_D, 2)
            w2      = vp - v
            λ1_idx3 = MatsubaraFunctions.grid_index_extrp( v, grids(λ_D, 2))
            η2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(η_D, 1))
            λ2_idx1 = MatsubaraFunctions.grid_index_extrp(w2, grids(λ_D, 1))
            v_idx   = MatsubaraFunctions.grid_index_extrp( v, grids(M_S, 2))
            w2_idx  = MatsubaraFunctions.grid_index_extrp(w2, grids(M_S, 1))

            for w in grids(T_D, 1)
                w1      = w + v + vp
                v2      = w + v
                η1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(η_D, 1))
                λ1_idx1 = MatsubaraFunctions.grid_index_extrp(w1, grids(λ_D, 1))
                λ2_idx2 = MatsubaraFunctions.grid_index_extrp(v2, grids(λ_D, 2))
                w_idx   = MatsubaraFunctions.grid_index_extrp( w, grids(M_S, 1))
                w1_idx  = MatsubaraFunctions.grid_index_extrp(w1, grids(M_S, 1))
                v2_idx  = MatsubaraFunctions.grid_index_extrp(v2, grids(M_S, 2))

                # compute SBE vertices
                p1 = λ_S[λ1_idx1, λ1_idx2, 1] * η_S[η1_idx, 1] * λ_S[λ1_idx1, λ1_idx3, 1]
                p2 = λ_D[λ2_idx1, λ1_idx3, 1] * η_D[η2_idx, 1] * λ_D[λ2_idx1, λ2_idx2, 1]
                p3 = λ_M[λ2_idx1, λ1_idx3, 1] * η_M[η2_idx, 1] * λ_M[λ2_idx1, λ2_idx2, 1]

                # compute MBE vertices
                m1 = M_S[w1_idx, v_idx, vp_idx, 1]
                m2 = M_T[w1_idx, v_idx, vp_idx, 1]
                m3 = M_D[w2_idx, v_idx, v2_idx, 1]
                m4 = M_M[w2_idx, v_idx, v2_idx, 1]

                T_D[w, v, vp] = -2.0 * U + M_D[w_idx, v_idx, vp_idx, 1] + 0.5 * (p1 + m1 - p2 - m3) + 1.5 * (m2 - p3 - m4) 
                T_M[w, v, vp] = +2.0 * U + M_M[w_idx, v_idx, vp_idx, 1] - 0.5 * (p1 + m1 + p2 + m3) + 0.5 * (m2 + p3 + m4)
            end
        end
    end

    return nothing 
end