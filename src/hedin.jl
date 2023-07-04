# symmetries of the Hedin vertex in the particle-particle channel
function s1_λ_p(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (-w[1], -w[2]), (x[1],), MatsubaraOperation(false, true)
end 

function s2_λ_p(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[1] - w[2]), (x[1],), MatsubaraOperation()
end 



# symmetries of the Hedin vertex in the particle-hole channel
function s1_λ_d(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (-w[1], -w[2]), (x[1],), MatsubaraOperation(false, true)
end 

function s2_λ_d(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (-w[1], w[1] + w[2]), (x[1],), MatsubaraOperation()
end 



# Hedin vertex in singlet channel
function calc_λ!(
    λ   :: MatsubaraFunction{2, 1, 3, Float64},
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

    g = MatsubaraGrid(temperature(G), N(grids(λ, 1)) + N(grids(λ, 2)) + N(grids(η_S, 1)), Fermion)

    # model the diagram
    function f(wtpl, xtpl)

        w, v = wtpl
        val  = 0.0

        for vp in g
            T_S  = calc_T(w, vp, v, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_S)
            val += T_S * G(vp; extrp = (true, 0.0)) * G(w - vp; extrp = (true, 0.0))
        end

        return 1.0 + 0.5 * temperature(λ) * val
    end

    # compute Hedin vertex
    SG(λ, MatsubaraInitFunction{2, 1, Float64}(f); mode = :hybrid)

    return nothing 
end

# Hedin vertex in density channel
function calc_λ!(
    λ   :: MatsubaraFunction{2, 1, 3, Float64},
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

    g = MatsubaraGrid(temperature(G), N(grids(λ, 1)) + N(grids(λ, 2)) + N(grids(η_S, 1)), Fermion)

    # model the diagram
    function f(wtpl, xtpl)

        w, v = wtpl
        val  = 0.0

        for vp in g
            T_D  = calc_T(w, v, vp, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_D)
            val -= T_D * G(w + vp; extrp = (true, 0.0)) * G(vp; extrp = (true, 0.0))
        end

        return 1.0 + temperature(λ) * val
    end

    # compute Hedin vertex
    SG(λ, MatsubaraInitFunction{2, 1, Float64}(f); mode = :hybrid)

    return nothing 
end

# Hedin vertex in magnetic channel
function calc_λ!(
    λ   :: MatsubaraFunction{2, 1, 3, Float64},
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

    g = MatsubaraGrid(temperature(G), N(grids(λ, 1)) + N(grids(λ, 2)) + N(grids(η_S, 1)), Fermion)

    # model the diagram
    function f(wtpl, xtpl)

        w, v = wtpl
        val  = 0.0

        for vp in g
            T_M  = calc_T(w, v, vp, η_S, λ_S, η_D, λ_D, η_M, λ_M, M_S, M_T, M_D, M_M, U, ch_M)
            val -= T_M * G(w + vp; extrp = (true, 0.0)) * G(vp; extrp = (true, 0.0))
        end

        return 1.0 + temperature(λ) * val
    end

    # compute Hedin vertex
    SG(λ, MatsubaraInitFunction{2, 1, Float64}(f); mode = :hybrid)
    
    return nothing 
end