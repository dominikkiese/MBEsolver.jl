# symmetries of the Hedin vertex in the particle-particle channel
function s1_λ_p(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (-w[1], -w[2]), x, MatsubaraOperation(false, true)
end 

function s2_λ_p(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[1] - w[2]), x, MatsubaraOperation()
end 



# symmetries of the Hedin vertex in the particle-hole channel
function s1_λ_d(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (-w[1], -w[2]), x, MatsubaraOperation(false, true)
end 

function s2_λ_d(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (-w[1], w[1] + w[2]), x, MatsubaraOperation()
end 



# Hedin vertex in singlet channel
function calc_λ!(
    λ  :: MatsubaraFunction{2, 1, 3, Float64},
    G  :: MatsubaraFunction{1, 1, 2, Float64},
    T  :: MatsubaraFunction{3, 1, 4, Float64},
    SG :: MatsubaraSymmetryGroup,
       :: Type{ch_S}
    )  :: Nothing

    # model the diagram
    function f(wtpl, xtpl)

        w, v    = wtpl
        val     = 0.0
        ex      = (true, 0.0)
        T_slice = view(T, w, :, v)

        for i in eachindex(grids(T, 2))
            vp   = grids(T, 2)[i]
            val += T_slice[i] * G(vp; extrp = ex) * G(w - vp; extrp = ex)
        end

        return 1.0 + 0.5 * temperature(λ) * val
    end

    # compute Hedin vertex
    SG(λ, MatsubaraInitFunction{2, 1, Float64}(f); mode = :threads)

    return nothing 
end

# Hedin vertex in density channel
function calc_λ!(
    λ  :: MatsubaraFunction{2, 1, 3, Float64},
    G  :: MatsubaraFunction{1, 1, 2, Float64},
    T  :: MatsubaraFunction{3, 1, 4, Float64},
    SG :: MatsubaraSymmetryGroup,
       :: Type{ch_D}
    )  :: Nothing

    # model the diagram
    function f(wtpl, xtpl)

        w, v    = wtpl
        val     = 0.0
        ex      = (true, 0.0)
        T_slice = view(T, w, v, :)

        for i in eachindex(grids(T, 3))
            vp   = grids(T, 3)[i]
            val -= T_slice[i] * G(w + vp; extrp = ex) * G(vp; extrp = ex)
        end

        return 1.0 + temperature(λ) * val
    end

    # compute Hedin vertex
    SG(λ, MatsubaraInitFunction{2, 1, Float64}(f); mode = :threads)

    return nothing 
end

# Hedin vertex in magnetic channel
function calc_λ!(
    λ  :: MatsubaraFunction{2, 1, 3, Float64},
    G  :: MatsubaraFunction{1, 1, 2, Float64},
    T  :: MatsubaraFunction{3, 1, 4, Float64},
    SG :: MatsubaraSymmetryGroup,
       :: Type{ch_M}
    )  :: Nothing

    return calc_λ!(λ, G, T, SG, ch_D)  
end