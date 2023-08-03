# symmetries of the multiboson vertex in the particle-particle channel
# note: crossing symmetry in triplet channel yields an additional minus sign
function s1_M_S(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[2], w[1] - w[3]), x, MatsubaraOperation()
end 

function s1_M_T(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[2], w[1] - w[3]), x, MatsubaraOperation(true, false)
end 

function s2_M_S(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[1] - w[2], w[3]), x, MatsubaraOperation()
end 

function s2_M_T(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[1] - w[2], w[3]), x, MatsubaraOperation(true, false)
end 

function s3_M_p(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (-w[1], -w[3], -w[2]), x, MatsubaraOperation(false, true)
end 

function s4_M_p(
    w :: NTuple{3, MatsubaraFrequency},
    x :: Tuple{Int64}
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Tuple{Int64}, MatsubaraOperation}

    return (w[1], w[3], w[2]), x, MatsubaraOperation()
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
    Π   :: MatsubaraFunction{2, 1, 3, Float64}, 
    T   :: MatsubaraFunction{3, 1, 4, Float64}, 
    M_S :: MatsubaraFunction{3, 1, 4, Float64}, 
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_S}
    )   :: Nothing

    # model the diagram 
    function f(wtpl, xtpl)

        w, v, vp  = wtpl 
        val       = 0.0
        v1, v2    = grids(Π, 2)(grids(T, 2)[1]), grids(Π, 2)(grids(T, 2)[end])
        Πslice    = view(Π, w, v1 : v2)
        M_S_slice = view(M_S, w, :, vp)
        T_L_slice = view(T, w, :, vp) 
        T_R_slice = view(T, w, :, v)

        # piecewise vectorization 
        vl = grids(T, 2)(grids(M_S, 2)[1])
        vr = grids(T, 2)(grids(M_S, 2)[end])

        @turbo for i in 1 : vl - 1
            val += (T_L_slice[i] - M_S_slice[1]) * Πslice[i] * T_R_slice[i]
        end 

        @turbo for i in vl : vr
            val += (T_L_slice[i] - M_S_slice[i - vl + 1]) * Πslice[i] * T_R_slice[i]
        end

        @turbo for i in vr + 1 : length(T_L_slice)
            val += (T_L_slice[i] - M_S_slice[vr - vl + 1]) * Πslice[i] * T_R_slice[i]
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
    Π   :: MatsubaraFunction{2, 1, 3, Float64}, 
    T   :: MatsubaraFunction{3, 1, 4, Float64}, 
    M_T :: MatsubaraFunction{3, 1, 4, Float64}, 
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_T}
    )   :: Nothing

    # model the diagram 
    function f(wtpl, xtpl)

        w, v, vp  = wtpl 
        val       = 0.0
        v1, v2    = grids(Π, 2)(grids(T, 2)[1]), grids(Π, 2)(grids(T, 2)[end])
        Πslice    = view(Π, w, v1 : v2)
        M_T_slice = view(M_T, w, :, vp)
        T_L_slice = view(T, w, :, vp) 
        T_R_slice = view(T, w, :, v)

        # piecewise vectorization, additional minus sign from use of crossing symmetry
        vl = grids(T, 2)(grids(M_T, 2)[1])
        vr = grids(T, 2)(grids(M_T, 2)[end])

        @turbo for i in 1 : vl - 1
            val -= (T_L_slice[i] - M_T_slice[1]) * Πslice[i] * T_R_slice[i]
        end 

        @turbo for i in vl : vr
            val -= (T_L_slice[i] - M_T_slice[i - vl + 1]) * Πslice[i] * T_R_slice[i]
        end

        @turbo for i in vr + 1 : length(T_L_slice)
            val -= (T_L_slice[i] - M_T_slice[vr - vl + 1]) * Πslice[i] * T_R_slice[i]
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
    Π   :: MatsubaraFunction{2, 1, 3, Float64}, 
    T   :: MatsubaraFunction{3, 1, 4, Float64}, 
    M_D :: MatsubaraFunction{3, 1, 4, Float64}, 
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_D}
    )   :: Nothing

    # model the diagram 
    function f(wtpl, xtpl)

        w, v, vp  = wtpl 
        val       = 0.0
        v1, v2    = grids(Π, 2)(grids(T, 3)[1]), grids(Π, 2)(grids(T, 3)[end])
        Πslice    = view(Π, w, v1 : v2)
        M_D_slice = view(M_D, w, v, :)
        T_L_slice = view(T, w, v, :)
        T_R_slice = view(T, w, vp, :)

        # piecewise vectorization 
        vl = grids(T, 3)(grids(M_D, 3)[1])
        vr = grids(T, 3)(grids(M_D, 3)[end])

        @turbo for i in 1 : vl - 1
            val -= (T_L_slice[i] - M_D_slice[1]) * Πslice[i] * T_R_slice[i]
        end 

        @turbo for i in vl : vr
            val -= (T_L_slice[i] - M_D_slice[i - vl + 1]) * Πslice[i] * T_R_slice[i]
        end

        @turbo for i in vr + 1 : length(T_L_slice)
            val -= (T_L_slice[i] - M_D_slice[vr - vl + 1]) * Πslice[i] * T_R_slice[i]
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
    Π   :: MatsubaraFunction{2, 1, 3, Float64}, 
    T   :: MatsubaraFunction{3, 1, 4, Float64}, 
    M_M :: MatsubaraFunction{3, 1, 4, Float64}, 
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_M}
    )   :: Nothing

    return calc_M!(M, Π, T, M_M, SG, ch_D)
end   