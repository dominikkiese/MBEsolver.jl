# polarization in particle-particle channel
function calc_P_pp!(P :: P_t, G :: G_t, λ :: λ_t) :: Nothing 
    set!(P, 0.0)

    Threads.@threads for w in grids(P, 1)
        P_w   = view(P, w, :, :, :, :)
        idx_w = MatsubaraFunctions.grid_index_extrp(w, grids(λ, 1))

        for v in grids(G, 1)
            G1_v  = view(G, v, :, :)
            G2_v  = slice_extrp(G, w - v)
            idx_v = MatsubaraFunctions.grid_index_extrp(w - v, grids(λ, 2))
            λ_v   = view(λ, idx_w, idx_v, :, :, :, :)

            # calculate tensor contraction
            Tullio.@einsum P_w[x1p, x1, x2p, x2] += G1_v[x1, x3] * G2_v[x2, x4] * λ_v[x1p, x3, x2p, x4]
        end
    end 

    mult!(P, -0.5 * temperature(P))
    return nothing 
end

# polarization in particle-hole channel
function calc_P_ph!(P :: P_t, G :: G_t, λ :: λ_t) :: Nothing 
    set!(P, 0.0)
    
    Threads.@threads for w in grids(P, 1)
        P_w   = view(P, w, :, :, :, :)
        idx_w = MatsubaraFunctions.grid_index_extrp(w, grids(λ, 1))

        for v in grids(G, 1)
            G1_v  = slice_extrp(G, w + v)
            G2_v  = view(G, v, :, :)
            idx_v = MatsubaraFunctions.grid_index_extrp(v, grids(λ, 2))
            λ_v   = view(λ, idx_w, idx_v, :, :, :, :)

            # calculate tensor contraction
            Tullio.@einsum P_w[x1p, x1, x2p, x2] += G1_v[x1, x3] * G2_v[x4, x1p] * λ_v[x4, x3, x2p, x2]
        end
    end 

    mult!(P, temperature(P))
    return nothing 
end