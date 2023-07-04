# fermionic Dyson equation for G
function calc_G(
    G0 :: MatsubaraFunction{1, 1, 2, Float64},
    Σ  :: MatsubaraFunction{1, 1, 2, Float64}
    )  :: MatsubaraFunction{1, 1, 2, Float64}

    G = deepcopy(G0)

    for v in grids(G, 1)
        # positive sign for Σ since (-i was factored out)
        G[v] = 1.0 / (1.0 / G0[v] + Σ(v; extrp = (true, 0.0)))
    end 

    return G
end

# bosonic Dyson equation for η
function calc_η(
    P   :: MatsubaraFunction{1, 1, 2, Float64},
    η   :: MatsubaraFunction{1, 1, 2, Float64},
    val :: Float64
    )   :: MatsubaraFunction{1, 1, 2, Float64}

    ηp = deepcopy(P)

    for w in grids(P, 1)
        ηp[w] = val * (1.0 + P[w] * η[w])
    end 

    return ηp 
end