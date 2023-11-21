# Schwinger-Dyson equation for Σ
function calc_Σ(
    G     :: MF1,
    η_D   :: MF1,
    λ_D   :: MF2,
    η_M   :: MF1,
    λ_M   :: MF2,
    U     :: Float64,
    num_v :: Int64
    )     :: MF1

    # generate container for Σ
    T = temperature(G)
    g = MatsubaraGrid(T, 4 * num_v, Fermion)
    Σ = MatsubaraFunction(MatsubaraGrid(T, num_v, Fermion); data_t = Float64)
    set!(Σ, 0.0)

    v_grid = grids(Σ, 1)

    @batch per = thread for iv in axes(v_grid)
        v = v_grid[iv]

        for vp in g
            Σ[v] -= (0.25 * η_D(v - vp; extrp = +U) * λ_D(v - vp, vp) + 
                     0.75 * η_M(v - vp; extrp = -U) * λ_M(v - vp, vp) + 0.5 * U) * G(vp) * T
        end 
    end

    return Σ 
end