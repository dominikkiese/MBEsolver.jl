# Schwinger-Dyson equation for Σ
function calc_Σ!(
    Σ    :: G_t,
    G    :: G_t,
    η_D  :: P_t,
    λ_D  :: λ_t,
    η_M  :: P_t,
    λ_M  :: λ_t,
    F0_D :: Array{ComplexF64, 4},
    F0_M :: Array{ComplexF64, 4}
    )    :: Nothing

    T = temperature(Σ)
    δ = SMatrix{2, 2, ComplexF64}(1, 0, 0, 1)
    set!(Σ, 0.0)
   
    Threads.@threads for v in grids(Σ, 1)
        Σ_v = view(Σ, v, :, :)

        # summation of 1/iν Hartree tail
        @tullio Σ_v[x1, x1p] = -(0.25 * F0_D[x3, x1, x1p, x3] + 0.75 * F0_M[x3, x1, x1p, x3])

        for vp in grids(G, 1)
            inv_vp = 1.0 / (im * value(vp))
            G_vp   = view(G, vp, :, :)
            η_idx  = MatsubaraFunctions.grid_index_extrp(v - vp, grids(η_D, 1))
            λ_idx1 = MatsubaraFunctions.grid_index_extrp(v - vp, grids(λ_D, 1))
            λ_idx2 = MatsubaraFunctions.grid_index_extrp(vp, grids(λ_D, 2))
            η_D_vp = view(η_D, η_idx, :, :, :, :)
            η_M_vp = view(η_M, η_idx, :, :, :, :)
            λ_D_vp = view(λ_D, λ_idx1, λ_idx2, :, :, :, :)
            λ_M_vp = view(λ_M, λ_idx1, λ_idx2, :, :, :, :)
            
            # Hartree contribution
            @tullio Σ_v[x1, x1p] += -T * (0.5 * F0_D[x3, x1, x1p, x4] + 1.5 * F0_M[x3, x1, x1p, x4]) * 
               (G_vp[x3, x4] - inv_vp * δ[x3, x4])

            # vertex corrections
            @tullio Σ_v[x1, x1p] += -T * (0.25 * (2.0 * η_D_vp[x3, x1, x5, x6] - F0_D[x3, x1, x5, x6]) * λ_D_vp[x4, x1p, x5, x6] + 
                0.75 * (2.0 * η_M_vp[x3, x1, x5, x6] - F0_M[x3, x1, x5, x6]) * λ_M_vp[x4, x1p, x5, x6]) * G_vp[x3, x4]
        end 
    end
end