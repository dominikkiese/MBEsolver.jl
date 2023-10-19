# symmetries of the Hedin vertex in the particle-particle channel
function s1_λ_p(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (-w[1], -w[2]), x, MatsubaraOperation(false, true)
end 

function s2_λ_p(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (w[1], w[1] - w[2]), x, MatsubaraOperation()
end 

# symmetries of the Hedin vertex in the particle-hole channel
function s1_λ_d(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (-w[1], -w[2]), x, MatsubaraOperation(false, true)
end 

function s2_λ_d(
    w :: NTuple{2, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (-w[1], w[1] + w[2]), x, MatsubaraOperation()
end 

# Hedin vertex in singlet channel
function calc_λ!(λ :: MF2, Π :: MF2, T :: MF3, SG :: MSG2, :: Type{ch_S}) :: Nothing

    function f(wtpl, xtpl)

        w, v    = wtpl
        val     = 0.0
        v1, v2  = grids(Π, 2)(grids(T, 2)[1]), grids(Π, 2)(grids(T, 2)[end])
        Π_slice = view(Π, w, v1 : v2)
        T_slice = view(T, w, :, v)

        for i in eachindex(Π_slice)
            val += T_slice[i] * Π_slice[i]
        end

        return 1.0 + 0.5 * temperature(λ) * val
    end

    SG(λ, MIF2(f); mode = :threads)
    return nothing 
end

# Hedin vertex in density channel
function calc_λ!(λ :: MF2, Π :: MF2, T :: MF3, SG :: MSG2, :: Type{ch_D}) :: Nothing

    function f(wtpl, xtpl)

        w, v    = wtpl
        val     = 0.0
        v1, v2  = grids(Π, 2)(grids(T, 3)[1]), grids(Π, 2)(grids(T, 3)[end])
        Π_slice = view(Π, w, v1 : v2)
        T_slice = view(T, w, v, :)

        for i in eachindex(Π_slice)
            val -= T_slice[i] * Π_slice[i]
        end

        return 1.0 + temperature(λ) * val
    end

    SG(λ, MIF2(f); mode = :threads)
    return nothing 
end

# Hedin vertex in magnetic channel
calc_λ!(λ :: MF2, Π :: MF2, T :: MF3, SG :: MSG2, :: Type{ch_M}) :: Nothing = calc_λ!(λ, Π, T, SG, ch_D)  