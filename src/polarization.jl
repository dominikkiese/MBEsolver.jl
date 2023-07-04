# polarization in singlet channel
function calc_P(
    λ     :: MatsubaraFunction{2, 1, 3, Float64},
    G     :: MatsubaraFunction{1, 1, 2, Float64},
    num_w :: Int64,
          :: Type{ch_S}
    )     :: MatsubaraFunction{1, 1, 2, Float64}

    # generate container for P
    T = temperature(G)
    g = MatsubaraGrid(T, 4 * num_w, Fermion)
    P = MatsubaraFunction(MatsubaraGrid(T, num_w, Boson), 1, Float64)
    set!(P, 0.0)
 
    Threads.@threads for w_idx in eachindex(grids(P, 1))
        w = grids(P, 1)[w_idx]

        for v in g
            P[w] += 0.5 * T * G(v; extrp = (true, 0.0)) * G(w - v; extrp = (true, 0.0)) * λ(w, v; extrp = (true, 0.0))
        end 
    end 

    return P
end 

# polarization in density channel
function calc_P(
    λ     :: MatsubaraFunction{2, 1, 3, Float64},
    G     :: MatsubaraFunction{1, 1, 2, Float64},
    num_w :: Int64,
          :: Type{ch_D}
    )     :: MatsubaraFunction{1, 1, 2, Float64}

    # generate container for P
    T = temperature(G)
    g = MatsubaraGrid(T, 4 * num_w, Fermion)
    P = MatsubaraFunction(MatsubaraGrid(T, num_w, Boson), 1, Float64)
    set!(P, 0.0)

    Threads.@threads for w_idx in eachindex(grids(P, 1))
        w = grids(P, 1)[w_idx]

        for v in g
            P[w] -= T * G(w + v; extrp = (true, 0.0)) * G(v; extrp = (true, 0.0)) * λ(w, v; extrp = (true, 0.0))
        end 
    end

    return P
end

# polarization in magnetic channel
function calc_P(
    λ     :: MatsubaraFunction{2, 1, 3, Float64},
    G     :: MatsubaraFunction{1, 1, 2, Float64},
    num_w :: Int64,
          :: Type{ch_M}
    )     :: MatsubaraFunction{1, 1, 2, Float64}

    return calc_P(λ, G, num_w, ch_D)
end