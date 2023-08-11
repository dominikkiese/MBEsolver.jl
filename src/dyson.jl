# fermionic Dyson equation for G
function calc_G(
    G0 :: MatsubaraFunction{1, 1, 2, Float64},
    Σ  :: MatsubaraFunction{1, 1, 2, Float64}
    )  :: MatsubaraFunction{1, 1, 2, Float64}

    G = copy(G0)

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

    ex = (true, 0.0)
    
    for w in grids(Π_pp, 1), v in grids(Π_pp, 2)
        Π_pp[w, v] = G(v; extrp = ex) * G(w - v; extrp = ex)
        Π_ph[w, v] = G(w + v; extrp = ex) * G(v; extrp = ex)
    end

    return nothing 
end