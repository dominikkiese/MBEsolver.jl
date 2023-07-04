# Schwinger-Dyson equation for Σ
function calc_Σ(
    G     :: MatsubaraFunction{1, 1, 2, Float64},
    η_D   :: MatsubaraFunction{1, 1, 2, Float64},
    λ_D   :: MatsubaraFunction{2, 1, 3, Float64},
    η_M   :: MatsubaraFunction{1, 1, 2, Float64},
    λ_M   :: MatsubaraFunction{2, 1, 3, Float64},
    U     :: Float64,
    num_v :: Int64
    )     :: MatsubaraFunction{1, 1, 2, Float64}

    # generate container for Σ
    T = temperature(G)
    g = MatsubaraGrid(T, 4 * num_v, Fermion)
    Σ = MatsubaraFunction(MatsubaraGrid(T, num_v, Fermion), 1, Float64)
    set!(Σ, 0.0)

    Threads.@threads for v_idx in eachindex(grids(Σ, 1))
        v = grids(Σ, 1)[v_idx]

        for vp in g
            Σ[v] -= (0.25 * η_D(v - vp; extrp = (true, +U)) * λ_D(v - vp, vp; extrp = (true, 0.0)) + 
                     0.75 * η_M(v - vp; extrp = (true, -U)) * λ_M(v - vp, vp; extrp = (true, 0.0)) + 0.5 * U) * G(vp; extrp = (true, 0.0)) * T
        end 
    end

    return Σ 
end