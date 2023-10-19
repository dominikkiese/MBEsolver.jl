# symmetries of the multiboson vertex in the particle-particle channel
# note: crossing symmetry in triplet channel yields an additional minus sign
function s1_M_S(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (w[1], w[2], w[1] - w[3]), x, MatsubaraOperation()
end 

function s1_M_T(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (w[1], w[2], w[1] - w[3]), x, MatsubaraOperation(true, false)
end 

function s2_M_S(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (w[1], w[1] - w[2], w[3]), x, MatsubaraOperation()
end 

function s2_M_T(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (w[1], w[1] - w[2], w[3]), x, MatsubaraOperation(true, false)
end 

function s3_M_p(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (-w[1], -w[3], -w[2]), x, MatsubaraOperation(false, true)
end 

function s4_M_p(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (w[1], w[3], w[2]), x, MatsubaraOperation()
end 

# symmetries of the multiboson vertex in the particle-hole channel 
function s1_M_d(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (-w[1], w[1] + w[3], w[1] + w[2]), x, MatsubaraOperation()
end 

function s2_M_d(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (-w[1], -w[3], -w[2]), x, MatsubaraOperation(false, true)
end 

function s3_M_d(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{}, MatsubaraOperation}

    return (w[1], w[3], w[2]), x, MatsubaraOperation()
end 

# multiboson vertex in singlet channel
function calc_M!(M :: MF3, Π :: MF2, T :: MF3, M_S :: MF3, SG :: MSG3, :: Type{ch_S}) :: Nothing

    function f(wtpl, xtpl)

        w, v, vp  = wtpl 
        val       = 0.0
        v1, v2    = grids(Π, 2)(grids(T, 2)[1]), grids(Π, 2)(grids(T, 2)[end])
        Π_slice   = view(Π, w, v1 : v2)
        M_S_slice = view(M_S, w, :, vp)
        T_L_slice = view(T, w, :, vp) 
        T_R_slice = view(T, w, :, v)

        vl = grids(T, 2)(grids(M_S, 2)[1])
        vr = grids(T, 2)(grids(M_S, 2)[end])

        for i in 1 : vl - 1
            val += (T_L_slice[i] - M_S_slice[1]) * Π_slice[i] * T_R_slice[i]
        end 

        for i in vl : vr
            val += (T_L_slice[i] - M_S_slice[i - vl + 1]) * Π_slice[i] * T_R_slice[i]
        end

        for i in vr + 1 : length(T_L_slice)
            val += (T_L_slice[i] - M_S_slice[end]) * Π_slice[i] * T_R_slice[i]
        end

        return 0.5 * temperature(M) * val
    end 

    SG(M, MIF3(f); mode = :hybrid)
    return nothing 
end   

# multiboson vertex in triplet channel
function calc_M!(M :: MF3, Π :: MF2, T :: MF3, M_T :: MF3, SG :: MSG3, :: Type{ch_T}) :: Nothing

    function f(wtpl, xtpl)

        w, v, vp  = wtpl 
        val       = 0.0
        v1, v2    = grids(Π, 2)(grids(T, 2)[1]), grids(Π, 2)(grids(T, 2)[end])
        Π_slice   = view(Π, w, v1 : v2)
        M_T_slice = view(M_T, w, :, vp)
        T_L_slice = view(T, w, :, vp) 
        T_R_slice = view(T, w, :, v)

        # additional minus sign from use of crossing symmetry
        vl = grids(T, 2)(grids(M_T, 2)[1])
        vr = grids(T, 2)(grids(M_T, 2)[end])

        for i in 1 : vl - 1
            val -= (T_L_slice[i] - M_T_slice[1]) * Π_slice[i] * T_R_slice[i]
        end 

        for i in vl : vr
            val -= (T_L_slice[i] - M_T_slice[i - vl + 1]) * Π_slice[i] * T_R_slice[i]
        end

        for i in vr + 1 : length(T_L_slice)
            val -= (T_L_slice[i] - M_T_slice[end]) * Π_slice[i] * T_R_slice[i]
        end

        return 0.5 * temperature(M) * val
    end 

    SG(M, MIF3(f); mode = :hybrid)
    return nothing 
end   

# multiboson vertex in density channel
function calc_M!(M :: MF3, Π :: MF2, T :: MF3, M_D :: MF3, SG :: MSG3, :: Type{ch_D}) :: Nothing

    # model the diagram 
    function f(wtpl, xtpl)

        w, v, vp  = wtpl 
        val       = 0.0
        v1, v2    = grids(Π, 2)(grids(T, 3)[1]), grids(Π, 2)(grids(T, 3)[end])
        Π_slice   = view(Π, w, v1 : v2)
        M_D_slice = view(M_D, w, v, :)
        T_L_slice = view(T, w, v, :)
        T_R_slice = view(T, w, vp, :)

        vl = grids(T, 3)(grids(M_D, 3)[1])
        vr = grids(T, 3)(grids(M_D, 3)[end])

        for i in 1 : vl - 1
            val -= (T_L_slice[i] - M_D_slice[1]) * Π_slice[i] * T_R_slice[i]
        end 

        for i in vl : vr
            val -= (T_L_slice[i] - M_D_slice[i - vl + 1]) * Π_slice[i] * T_R_slice[i]
        end

        for i in vr + 1 : length(T_L_slice)
            val -= (T_L_slice[i] - M_D_slice[end]) * Π_slice[i] * T_R_slice[i]
        end

        return temperature(M) * val
    end 

    SG(M, MIF3(f); mode = :hybrid)
    return nothing 
end 

# multiboson vertex in magnetic channel
calc_M!(M :: MF3, Π :: MF2, T :: MF3, M_M :: MF3, SG :: MSG3, :: Type{ch_M}) :: Nothing = calc_M!(M, Π, T, M_M, SG, ch_D)  