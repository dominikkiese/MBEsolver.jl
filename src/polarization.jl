# polarization in singlet channel
function calc_P(λ :: MF2, Π :: MF2, num_w :: Int64, :: Type{ch_S}) :: MF1

    # generate container for P
    T  = temperature(Π)
    P  = MatsubaraFunction(MatsubaraGrid(T, num_w, Boson); data_t = Float64)
    L  = grids_shape(P, 1)
    vl = grids(Π, 2)(grids(λ, 2)[1])
    vr = grids(Π, 2)(grids(λ, 2)[end])
    set!(P, 0.0)
 
    @batch per = thread for w_idx in 1 : L
        w       = grids(P, 1)[w_idx]
        w_λ     = grids(λ, 1)[MatsubaraFunctions.grid_index_extrp(w, grids(λ, 1))]
        Π_slice = view(Π, w, :)
        λ_slice = view(λ, w_λ, :)
        val1    = 0.0
        val2    = 0.0
        val3    = 0.0

        for i in 1 : vl - 1
            val1 += Π_slice[i]
        end

        for i in vl : vr
            val2 += Π_slice[i] * λ_slice[i - vl + 1]
        end

        for i in vr + 1 : length(Π_slice)
            val3 += Π_slice[i]
        end

        val1 *= λ_slice[1]
        val3 *= λ_slice[end]
        P[w]  = val1 + val2 + val3
    end 

    mult!(P, 0.5 * T)
    return P
end 

# polarization in density channel
function calc_P(λ :: MF2, Π :: MF2, num_w :: Int64, :: Type{ch_D}) :: MF1

    # generate container for P
    T  = temperature(Π)
    P  = MatsubaraFunction(MatsubaraGrid(T, num_w, Boson); data_t = Float64)
    L  = grids_shape(P, 1)
    vl = grids(Π, 2)(grids(λ, 2)[1])
    vr = grids(Π, 2)(grids(λ, 2)[end])
    set!(P, 0.0)

    @batch per = thread for w_idx in 1 : L
        w       = grids(P, 1)[w_idx]
        w_λ     = grids(λ, 1)[MatsubaraFunctions.grid_index_extrp(w, grids(λ, 1))]
        Π_slice = view(Π, w, :)
        λ_slice = view(λ, w_λ, :)
        val1    = 0.0
        val2    = 0.0
        val3    = 0.0

        for i in 1 : vl - 1
            val1 -= Π_slice[i]
        end

        for i in vl : vr
            val2 -= Π_slice[i] * λ_slice[i - vl + 1]
        end

        for i in vr + 1 : length(Π_slice)
            val3 -= Π_slice[i]
        end

        val1 *= λ_slice[1]
        val3 *= λ_slice[end]
        P[w]  = val1 + val2 + val3
    end

    mult!(P, T)
    return P
end

# polarization in magnetic channel
calc_P(λ :: MF2, Π :: MF2, num_w :: Int64, :: Type{ch_M}) :: MF1 = calc_P(λ, Π, num_w, ch_D)