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
    T   :: MatsubaraFunction{3, 1, 4, Float64}, 
    M_S :: MatsubaraFunction{3, 1, 4, Float64}, 
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_S}
    )   :: Nothing

    # model the diagram 
    function f(wtpl, xtpl)

        w, v, vp  = wtpl 
        val       = 0.0
        ex        = (true, 0.0)
        T_L_slice = view(T, w, :, vp) 
        T_R_slice = view(T, w, :, w - v)

        for i in eachindex(grids(T, 2))
            vpp  = grids(T, 2)[i]
            val += (T_L_slice[i] - M_S(w, vpp, vp)) * G(vpp; extrp = ex) * G(w - vpp; extrp = ex) * T_R_slice[i]
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
    T   :: MatsubaraFunction{3, 1, 4, Float64}, 
    M_T :: MatsubaraFunction{3, 1, 4, Float64}, 
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_T}
    )   :: Nothing

    return calc_M!(M, G, T, M_T, SG, ch_S)
end   

# multiboson vertex in density channel
function calc_M!(
    M   :: MatsubaraFunction{3, 1, 4, Float64},
    G   :: MatsubaraFunction{1, 1, 2, Float64}, 
    T   :: MatsubaraFunction{3, 1, 4, Float64}, 
    M_D :: MatsubaraFunction{3, 1, 4, Float64}, 
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_D}
    )   :: Nothing

    # model the diagram 
    function f(wtpl, xtpl)

        w, v, vp  = wtpl 
        val       = 0.0
        ex        = (true, 0.0)
        T_L_slice = view(T, w, v, :)
        T_R_slice = view(T, w, vp, :)

        for i in eachindex(grids(T, 3))
            vpp  = grids(T, 3)[i]
            val -= (T_L_slice[i] - M_D(w, v, vpp)) * G(w + vpp; extrp = ex) * G(vpp; extrp = ex) * T_R_slice[i]
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
    T   :: MatsubaraFunction{3, 1, 4, Float64}, 
    M_M :: MatsubaraFunction{3, 1, 4, Float64}, 
    SG  :: MatsubaraSymmetryGroup,
        :: Type{ch_M}
    )   :: Nothing

    return calc_M!(M, G, T, M_M, SG, ch_D)
end   