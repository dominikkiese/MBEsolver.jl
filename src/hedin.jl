# Hedin vertex in singlet channel
function calc_λ!(
    λ   :: λ_t,
    G   :: G_t, 
    η_D :: P_t,
    λ_D :: λ_t,
    η_M :: P_t,
    λ_M :: λ_t,
    F0  :: Array{ComplexF64, 4},
        :: Type{ch_S}
    )   :: Nothing

    T = temperature(G)
    δ = SMatrix{2, 2, ComplexF64}(1, 0, 0, 1)
    set!(λ, 0.0)

    Threads.@threads for w in grids(λ, 1)
        for v in grids(λ, 2)
            λ_w_v = view(λ, (w, v), :, :, :, :)

            # initialize λ
            @tullio λ_w_v[x1p, x1, x2p, x2] = δ[x1p, x1] * δ[x2p, x2]

            for vp in grids(G, 1)
                G1_v = view(G, vp, :, :) 
                G2_v = slice_extrp(G, w - vp)
                T_S  = calc_T(w, vp, v, η_D, λ_D, η_M, λ_M, F0, ch_S)

                # calculate tensor contraction
                @tullio λ_w_v[x1p, x1, x2p, x2] += -0.5 * T * T_S[x3, x1, x4, x2] * G1_v[x3, x1p] * G2_v[x4, x2p]
            end 
        end 
    end 

    return nothing 
end

# Hedin vertex in triplet channel
function calc_λ!(
    λ   :: λ_t,
    G   :: G_t, 
    η_D :: P_t,
    λ_D :: λ_t,
    η_M :: P_t,
    λ_M :: λ_t,
    F0  :: Array{ComplexF64, 4},
        :: Type{ch_T}
    )   :: Nothing

    T = temperature(G)
    δ = SMatrix{2, 2, ComplexF64}(1, 0, 0, 1)
    set!(λ, 0.0)

    Threads.@threads for w in grids(λ, 1)
        for v in grids(λ, 2)
            λ_w_v = view(λ, (w, v), :, :, :, :)

            # initialize λ
            @tullio λ_w_v[x1p, x1, x2p, x2] = δ[x1p, x1] * δ[x2p, x2]

            for vp in grids(G, 1)
                G1_v = view(G, vp, :, :) 
                G2_v = slice_extrp(G, w - vp)
                T_T  = calc_T(w, vp, v, η_D, λ_D, η_M, λ_M, F0, ch_T)

                @tullio λ_w_v[x1p, x1, x2p, x2] += -0.5 * T * T_T[x3, x1, x4, x2] * G1_v[x3, x1p] * G2_v[x4, x2p]
            end 
        end 
    end 

    return nothing 
end

# Hedin vertex in density channel
function calc_λ!(
    λ   :: λ_t,
    G   :: G_t, 
    η_S :: P_t,
    λ_S :: λ_t,
    η_T :: P_t,
    λ_T :: λ_t,
    η_D :: P_t,
    λ_D :: λ_t,
    η_M :: P_t,
    λ_M :: λ_t,
    F0  :: Array{ComplexF64, 4},
        :: Type{ch_D}
    )   :: Nothing

    T = temperature(G)
    δ = SMatrix{2, 2, ComplexF64}(1, 0, 0, 1)
    set!(λ, 0.0)

    Threads.@threads for w in grids(λ, 1)
        for v in grids(λ, 2)
            λ_w_v = view(λ, (w, v), :, :, :, :)

            # initialize λ
            @tullio λ_w_v[x1p, x1, x2p, x2] = δ[x2p, x1] * δ[x1p, x2]

            for vp in grids(G, 1)
                G1_v = slice_extrp(G, w + vp)
                G2_v = view(G, vp, :, :)
                T_D  = calc_T(w, v, vp, η_S, λ_S, η_T, λ_T, η_D, λ_D, η_M, λ_M, F0, ch_D)

                # calculate tensor contraction
                @tullio λ_w_v[x1p, x1, x2p, x2] += T * T_D[x1p, x1, x3, x4] * G1_v[x3, x2p] * G2_v[x2, x4]
            end 
        end 
    end 

    return nothing 
end

# Hedin vertex in magnetic channel
function calc_λ!(
    λ   :: λ_t,
    G   :: G_t, 
    η_S :: P_t,
    λ_S :: λ_t,
    η_T :: P_t,
    λ_T :: λ_t,
    η_D :: P_t,
    λ_D :: λ_t,
    η_M :: P_t,
    λ_M :: λ_t,
    F0  :: Array{ComplexF64, 4},
        :: Type{ch_M}
    )   :: Nothing

    T = temperature(G)
    δ = SMatrix{2, 2, ComplexF64}(1, 0, 0, 1)
    set!(λ, 0.0)
    
    Threads.@threads for w in grids(λ, 1)
        for v in grids(λ, 2)
            λ_w_v = view(λ, (w, v), :, :, :, :)

            # initialize λ
            @tullio λ_w_v[x1p, x1, x2p, x2] = δ[x2p, x1] * δ[x1p, x2]

            for vp in grids(G, 1)
                G1_v = slice_extrp(G, w + vp)
                G2_v = view(G, vp, :, :)
                T_M  = calc_T(w, v, vp, η_S, λ_S, η_T, λ_T, η_D, λ_D, η_M, λ_M, F0, ch_M)

                # calculate tensor contraction
                @tullio λ_w_v[x1p, x1, x2p, x2] += T * T_M[x1p, x1, x3, x4] * G1_v[x3, x2p] * G2_v[x2, x4]
            end 
        end 
    end 

    return nothing 
end