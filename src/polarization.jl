# polarization in singlet channel
function calc_P(
    λ     :: MatsubaraFunction{2, 1, 3, Float64},
    Π     :: MatsubaraFunction{2, 1, 3, Float64},
    num_w :: Int64,
          :: Type{ch_S}
    )     :: MatsubaraFunction{1, 1, 2, Float64}

    # generate container for P
    T = temperature(Π)
    P = MatsubaraFunction(MatsubaraGrid(T, num_w, Boson), 1, Float64)
    L = grids_shape(P, 1)
    set!(P, 0.0)
 
    @batch per = thread for w_idx in 1 : L
        w       = grids(P, 1)[w_idx]
        w_λ     = grids(λ, 1)[MatsubaraFunctions.grid_index_extrp(w, grids(λ, 1))]
        Π_slice = view(Π, w, :)
        λ_slice = view(λ, w_λ, :)

        # piecewise vectorization
        vl = grids(Π, 2)(grids(λ, 2)[1])
        vr = grids(Π, 2)(grids(λ, 2)[end])

        val1 = 0.0

        @turbo for i in 1 : vl - 1
            val1 += Π_slice[i]
        end

        val1 *= λ_slice[1]
        val2  = 0.0

        @turbo for i in vl : vr
            val2 += Π_slice[i] * λ_slice[i - vl + 1]
        end

        val3 = 0.0

        @turbo for i in vr + 1 : length(Π_slice)
            val3 += Π_slice[i]
        end

        val3 *= λ_slice[vr - vl + 1]
        P[w]  = val1 + val2 + val3
    end 

    mult!(P, 0.5 * T)
    return P
end 

# polarization in density channel
function calc_P(
    λ     :: MatsubaraFunction{2, 1, 3, Float64},
    Π     :: MatsubaraFunction{2, 1, 3, Float64},
    num_w :: Int64,
          :: Type{ch_D}
    )     :: MatsubaraFunction{1, 1, 2, Float64}

    # generate container for P
    T = temperature(Π)
    P = MatsubaraFunction(MatsubaraGrid(T, num_w, Boson), 1, Float64)
    L = grids_shape(P, 1)
    set!(P, 0.0)

    @batch per = thread for w_idx in 1 : L
        w       = grids(P, 1)[w_idx]
        w_λ     = grids(λ, 1)[MatsubaraFunctions.grid_index_extrp(w, grids(λ, 1))]
        Π_slice = view(Π, w, :)
        λ_slice = view(λ, w_λ, :)

        # piecewise vectorization
        vl = grids(Π, 2)(grids(λ, 2)[1])
        vr = grids(Π, 2)(grids(λ, 2)[end])

        val1 = 0.0

        @turbo for i in 1 : vl - 1
            val1 -= Π_slice[i]
        end

        val1 *= λ_slice[1]
        val2  = 0.0

        @turbo for i in vl : vr
            val2 -= Π_slice[i] * λ_slice[i - vl + 1]
        end

        val3 = 0.0

        @turbo for i in vr + 1 : length(Π_slice)
            val3 -= Π_slice[i]
        end

        val3 *= λ_slice[vr - vl + 1]
        P[w]  = val1 + val2 + val3
    end

    mult!(P, T)
    return P
end

# polarization in magnetic channel
function calc_P(
    λ     :: MatsubaraFunction{2, 1, 3, Float64},
    Π     :: MatsubaraFunction{2, 1, 3, Float64},
    num_w :: Int64,
          :: Type{ch_M}
    )     :: MatsubaraFunction{1, 1, 2, Float64}

    return calc_P(λ, Π, num_w, ch_D)
end