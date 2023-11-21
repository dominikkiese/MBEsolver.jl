# fermionic Dyson equation for G
function calc_G(G0 :: MF1, Σ :: MF1) :: MF1

    G = copy(G0)

    for v in grids(G, 1)
        # positive sign for Σ since (-i was factored out)
        G[v] = 1.0 / (1.0 / G0[v] + Σ(v))
    end 

    return G
end

# bosonic Dyson equation for η (iterative form empirically more stable)
function calc_η(P :: MF1, η :: MF1, val :: Float64) :: MF1

    ηp = copy(P)

    for w in grids(P, 1)
        ηp[w] = val * (1.0 + P[w] * η[w])
    end 

    return ηp 
end

# bubble functions Π
function calc_Π!(Π_pp :: MF2, Π_ph :: MF2, G :: MF1) :: Nothing 
    
    v_grid = grids(Π_pp, 2)

    @batch per = thread for iv in axes(v_grid)
        v = v_grid[iv]
        g = G(v)
        
        for w in grids(Π_pp, 1)
            Π_pp[w, v] = g * G(w - v)
            Π_ph[w, v] = g * G(w + v)
        end
    end

    return nothing 
end