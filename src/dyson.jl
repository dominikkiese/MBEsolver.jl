# fermionic Dyson equation for G
function calc_G(
    G0 :: MatsubaraFunction{1, 1, 2, Float64},
    Σ  :: MatsubaraFunction{1, 1, 2, Float64}
    )  :: MatsubaraFunction{1, 1, 2, Float64}

    G = copy(G0)

    for v in grids(G, 1)
        # positive sign for Σ since (-i was factored out)
        G[v] = 1.0 / (1.0 / G0[v] + Σ(v))
    end 

    return G
end

# bosonic Dyson equation for η
function calc_η(
    P   :: MatsubaraFunction{1, 1, 2, Float64},
    η   :: MatsubaraFunction{1, 1, 2, Float64},
    val :: Float64
    )   :: MatsubaraFunction{1, 1, 2, Float64}

    ηp = copy(P)

    for w in grids(P, 1)
        ηp[w] = val * (1.0 + P[w] * η[w])
    end 

    return ηp 
end

# bubble functions Π
function calc_Π!(
    Π_pp :: MatsubaraFunction{2, 1, 3, Float64},
    Π_ph :: MatsubaraFunction{2, 1, 3, Float64},
    G    :: MatsubaraFunction{1, 1, 2, Float64}
    )    :: Nothing 
    
    L = grids_shape(Π_pp, 2)

    @batch per = thread for v_idx in 1 : L
        v = grids(Π_pp, 2)[v_idx]
        g = G(v)
        
        for w in grids(Π_pp, 1)
            Π_pp[w, v] = g * G(w - v)
            Π_ph[w, v] = g * G(w + v)
        end
    end

    return nothing 
end