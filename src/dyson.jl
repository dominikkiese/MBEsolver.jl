# fermionic Dyson equation for G (no extrapolation for Σ)
function calc_G!(G :: G_t, G0 :: G_t, Σ :: G_t) :: Nothing
    Threads.@threads for v in grids(G, 1)
        G0_ij = SMatrix{2, 2, ComplexF64}(view(G0, v, :, :))
        G_ij  = view(G, v, :, :)

        if is_inbounds(v, grids(Σ, 1))
            Σ_ij  = SMatrix{2, 2, ComplexF64}(view(Σ, v, :, :))
            G_ij .= inv(inv(G0_ij) - Σ_ij)
        else 
            G_ij .= G0_ij 
        end
    end 
end

# bosonic Dyson equation in particle-particle channel
function calc_η_pp!(η :: P_t, P :: P_t, F0 :: Array{ComplexF64, 4}) :: Nothing
    ηp = MatsubaraFunction(η)
    set!(η, 0.0)

    Threads.@threads for w in grids(η, 1)
        η_w  = view(η, w, :, :, :, :)
        P_w  = view(P, w, :, :, :, :)
        ηp_w = view(ηp, w, :, :, :, :)

        # calculate tensor contraction
        η_w .= F0
        Tullio.@einsum η_w[x1p, x1, x2p, x2] += F0[x3, x1, x4, x2] * P_w[x5, x3, x6, x4] * ηp_w[x1p, x5, x2p, x6]
    end 
end

# bosonic Dyson equation in particle-hole channel
function calc_η_ph!(η :: P_t, P :: P_t, F0 :: Array{ComplexF64, 4}) :: Nothing
    ηp = MatsubaraFunction(η)
    set!(η, 0.0)

    Threads.@threads for w in grids(η, 1)
        η_w  = view(η, w, :, :, :, :)
        P_w  = view(P, w, :, :, :, :)
        ηp_w = view(ηp, w, :, :, :, :)

        # calculate tensor contraction
        η_w .= F0
        Tullio.@einsum η_w[x1p, x1, x2p, x2] += F0[x1p, x1, x3, x4] * P_w[x4, x3, x5, x6] * ηp_w[x6, x5, x2p, x2]
    end 
end