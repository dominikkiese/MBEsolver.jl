# symmetries of the multiboson vertex in the particle-particle channel
function s1_M_p(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[2], w[1] - w[3]), (x[1],), MatsubaraOperation()
end 

function s2_M_p(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[1] - w[2], w[3]), (x[1],), MatsubaraOperation()
end 

function s3_M_p(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (-w[1], -w[3], -w[2]), (x[1],), MatsubaraOperation(false, true)
end 

function s4_M_p(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[3], w[2]), (x[1],), MatsubaraOperation()
end 



# symmetries of the multiboson vertex in the particle-hole channel 
function s1_M_d(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (-w[1], w[1] + w[3], w[1] + w[2]), (x[1],), MatsubaraOperation()
end 

function s2_M_d(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (-w[1], -w[3], -w[2]), (x[1],), MatsubaraOperation(false, true)
end 

function s3_M_d(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[3], w[2]), (x[1],), MatsubaraOperation()
end 



# multiboson vertex in singlet channel
function calc_M!(
    M   :: MatsubaraFunction{3, 1, 4, Float64},
    G   :: MatsubaraFunction{1, 1, 2, Float64}, 
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
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_S}
    )   :: Nothing

    g = MatsubaraGrid(temperature(G), 2 * N(grids(M, 1)) + N(grids(M, 2)) + N(grids(η_S, 1)), Fermion)

    # model the diagram 
    function f(wtpl, xtpl)

        w, v, vp = wtpl 
        val      = 0.0

        for vpp in g
            T_L  = calc_T(w, vpp, vp, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_S)
            T_R  = calc_T(w, v, w - vpp, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_S)
            val += (T_L - M_S(w, vpp, vp; extrp = (true, 0.0))) * G(vpp; extrp = (true, 0.0)) * G(w - vpp; extrp = (true, 0.0)) * T_R
        end 

        return 0.5 * temperature(M) * val
    end 

    # compute multiboson vertex 
    SG(M, MatsubaraInitFunction{3, 1, Float64}(f); mode = :hybrid)

    return nothing 
end   

# multiboson vertex in triplet channel
function calc_M!(
    M   :: MatsubaraFunction{3, 1, 4, Float64},
    G   :: MatsubaraFunction{1, 1, 2, Float64}, 
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
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_T}
    )   :: Nothing

    g = MatsubaraGrid(temperature(G), 2 * N(grids(M, 1)) + N(grids(M, 2)) + N(grids(η_S, 1)), Fermion)

    # model the diagram 
    function f(wtpl, xtpl)

        w, v, vp = wtpl 
        val      = 0.0

        for vpp in g
            T_L  = calc_T(w, vpp, vp, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_T)
            T_R  = calc_T(w, v, w - vpp, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_T)
            val += (T_L - M_T(w, vpp, vp; extrp = (true, 0.0))) * G(vpp; extrp = (true, 0.0)) * G(w - vpp; extrp = (true, 0.0)) * T_R
        end 

        return 0.5 * temperature(M) * val
    end 

    # compute multiboson vertex 
    SG(M, MatsubaraInitFunction{3, 1, Float64}(f); mode = :hybrid)

    return nothing 
end   

# multiboson vertex in density channel
function calc_M!(
    M   :: MatsubaraFunction{3, 1, 4, Float64},
    G   :: MatsubaraFunction{1, 1, 2, Float64}, 
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
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_D}
    )   :: Nothing

    g = MatsubaraGrid(temperature(G), N(grids(M, 1)) + N(grids(M, 2)) + N(grids(η_S, 1)), Fermion)

    # model the diagram 
    function f(wtpl, xtpl)

        w, v, vp = wtpl 
        val      = 0.0

        for vpp in g
            T_L  = calc_T(w, v, vpp, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_D)
            T_R  = calc_T(w, vpp, vp, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_D)
            val -= (T_L - M_D(w, v, vpp; extrp = (true, 0.0))) * G(w + vpp; extrp = (true, 0.0)) * G(vpp; extrp = (true, 0.0)) * T_R
        end 

        return temperature(M) * val
    end 

    # compute multiboson vertex 
    SG(M, MatsubaraInitFunction{3, 1, Float64}(f); mode = :hybrid)

    return nothing 
end 

# multiboson vertex in magnetic channel
function calc_M!(
    M   :: MatsubaraFunction{3, 1, 4, Float64},
    G   :: MatsubaraFunction{1, 1, 2, Float64}, 
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
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_M}
    )   :: Nothing

    g = MatsubaraGrid(temperature(G), N(grids(M, 1)) + N(grids(M, 2)) + N(grids(η_S, 1)), Fermion)

    # model the diagram 
    function f(wtpl, xtpl)

        w, v, vp = wtpl 
        val      = 0.0

        for vpp in g
            T_L  = calc_T(w, v, vpp, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_M)
            T_R  = calc_T(w, vpp, vp, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_M)
            val -= (T_L - M_M(w, v, vpp; extrp = (true, 0.0))) * G(w + vpp; extrp = (true, 0.0)) * G(vpp; extrp = (true, 0.0)) * T_R
        end 

        return temperature(M) * val
    end 

    # compute multiboson vertex 
    SG(M, MatsubaraInitFunction{3, 1, Float64}(f); mode = :hybrid)

    return nothing 
end   