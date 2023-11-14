# polarization in singlet channel
function calc_P(λ :: MF2, Π :: MF2, num_w :: Int64, :: Type{ch_S}) :: MF1

    # generate container for P
    T  = temperature(Π)
    P  = MatsubaraFunction(MatsubaraGrid(T, num_w, Boson); data_t = Float64)
    set!(P, 0.0)
    vl_Π = firstindex(grids(Π, 2))
    vr_Π =  lastindex(grids(Π, 2))
    vl_λ = firstindex(grids(λ, 2))
    vr_λ =  lastindex(grids(λ, 2))

    w_grid = grids(P, 1)
    wl = firstindex(w_grid)
    wr =  lastindex(w_grid)
 
    @batch per = thread for iw in wl:wr
        w = w_grid[iw]
        w_λ     = grids(λ, 1)[MatsubaraFunctions.grid_index_extrp(w, grids(λ, 1))]
        Π_slice_l = view(Π, w, vl_Π     : vl_λ-1)
        Π_slice_c = view(Π, w, vl_λ     : vr_λ)
        Π_slice_r = view(Π, w, vr_λ+1   : vr_Π)
        λ_slice = view(λ, w_λ, vl_λ     : vr_λ)

        val1 = sum(Π_slice_l) * λ_slice[1]
        val2 = mapreduce(*, +, Π_slice_c, λ_slice)
        val3 = sum(Π_slice_r) * λ_slice[end]

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
    set!(P, 0.0)
    vl_Π = firstindex(grids(Π, 2))
    vr_Π =  lastindex(grids(Π, 2))
    vl_λ = firstindex(grids(λ, 2))
    vr_λ =  lastindex(grids(λ, 2))

    w_grid = grids(P, 1)
    wl = firstindex(w_grid)
    wr =  lastindex(w_grid)

    @batch per = thread for iw in wl:wr
        w = w_grid[iw]
        w_λ     = grids(λ, 1)[MatsubaraFunctions.grid_index_extrp(w, grids(λ, 1))]
        Π_slice_l = view(Π, w, vl_Π     : vl_λ-1)
        Π_slice_c = view(Π, w, vl_λ     : vr_λ)
        Π_slice_r = view(Π, w, vr_λ+1   : vr_Π)
        λ_slice = view(λ, w_λ, vl_λ     : vr_λ)

        val1 = sum(Π_slice_l) * λ_slice[1]
        val2 = mapreduce(*, +, Π_slice_c, λ_slice)
        val3 = sum(Π_slice_r) * λ_slice[end]

        P[w]  = val1 + val2 + val3
    end 

    mult!(P, -T)
    return P
end

# polarization in magnetic channel
calc_P(λ :: MF2, Π :: MF2, num_w :: Int64, :: Type{ch_M}) :: MF1 = calc_P(λ, Π, num_w, ch_D)