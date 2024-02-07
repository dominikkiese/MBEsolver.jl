mutable struct Solver
    # model parameters
    const T :: Float64 
    const U :: Float64 
    const J :: Float64 
    const Δ :: Float64 
    const μ :: Float64

    # parameters for periodic Pulay mixing
    m     :: Int64 
    p     :: Int64
    α     :: Float64 
    atol  :: Float64
    rtol  :: Float64
    iters :: Int64

    # propagators
    const G0 :: G_t
    const G  :: G_t 
    const Σ  :: G_t

    # bare vertices 
    const F0_S :: Array{ComplexF64, 4}
    const F0_T :: Array{ComplexF64, 4}
    const F0_D :: Array{ComplexF64, 4}
    const F0_M :: Array{ComplexF64, 4}

    # polarizations
    const P_S :: P_t
    const P_T :: P_t
    const P_D :: P_t
    const P_M :: P_t

    # screened interactions
    const η_S :: P_t
    const η_T :: P_t
    const η_D :: P_t
    const η_M :: P_t

    # Hedin vertices and their buffers for inplace calculations
    const λ_S       :: λ_t
    const λ_T       :: λ_t
    const λ_D       :: λ_t
    const λ_M       :: λ_t
    const λ_S_dummy :: λ_t
    const λ_T_dummy :: λ_t
    const λ_D_dummy :: λ_t
    const λ_M_dummy :: λ_t

    function Solver(
        T       :: Float64,
        U       :: Float64,
        J       :: Float64, 
        Δ       :: Float64,
        μ       :: Float64,
        num_G   :: Int64,
        num_Σ   :: Int64, 
        num_P   :: Int64, 
        num_λ_w :: Int64,  
        num_λ_v :: Int64, 
        ;
        m       :: Int64   = 5,
        p       :: Int64   = 3,
        α       :: Float64 = 0.5,
        atol    :: Float64 = 1e-5,
        rtol    :: Float64 = 1e-3,   
        iters   :: Int64   = 100
        )       :: Solver

        # initialization of G0 and G 
        grid_G = MatsubaraGrid(T, num_G, Fermion)
        G0     = MatsubaraFunction(grid_G, 2, 2)
        δ      = SMatrix{2, 2, ComplexF64}(1, 0, 0, 1)

        for v in grid_G
            G0_ij               = view(G0, v, :, :)
            @tullio G0_ij[i, j] = δ[i, j] / (im * value(v) - μ + im * Δ * sign(value(v)))
        end

        G = MatsubaraFunction(grid_G, 2, 2)
        set!(G, G0)

        # initialization of F0 
        F0_S = zeros(ComplexF64, 2, 2, 2, 2)
        F0_T = zeros(ComplexF64, 2, 2, 2, 2)
        F0_D = zeros(ComplexF64, 2, 2, 2, 2)
        F0_M = zeros(ComplexF64, 2, 2, 2, 2)

        @tullio F0_S[x1p, x1, x2p, x2] = 2.0 * U * δ[x1p, x2p] * δ[x1p, x1] * δ[x2p, x2] -
            0.75 * J * (1.0 - δ[x1p, x2p]) * (δ[x1p, x1] * δ[x2p, x2] + δ[x2p, x1] * δ[x1p, x2])

        @tullio F0_T[x1p, x1, x2p, x2] = 0.25 * J * (1.0 - δ[x1p, x2p]) * (δ[x1p, x1] * δ[x2p, x2] - δ[x2p, x1] * δ[x1p, x2])

        @tullio F0_D[x1p, x1, x2p, x2] = U * δ[x1p, x2p] * δ[x1p, x1] * δ[x2p, x2] -
            0.75 * J * (1.0 - δ[x1p, x2p]) * δ[x2p, x1] * δ[x1p, x2]

        @tullio F0_M[x1p, x1, x2p, x2] = -U * δ[x1p, x2p] * δ[x1p, x1] * δ[x2p, x2] +
            0.5 * J * (1.0 - δ[x1p, x2p]) * (δ[x1p, x1] * δ[x2p, x2] + 0.5 * δ[x2p, x1] * δ[x1p, x2])

        # initialization of P, η
        grid_P = MatsubaraGrid(T, num_P, Boson)
        P_S    = MatsubaraFunction(grid_P, 2, 2, 2, 2)
        P_T    = MatsubaraFunction(grid_P, 2, 2, 2, 2)
        P_D    = MatsubaraFunction(grid_P, 2, 2, 2, 2)
        P_M    = MatsubaraFunction(grid_P, 2, 2, 2, 2)
        η_S    = MatsubaraFunction(grid_P, 2, 2, 2, 2)
        η_T    = MatsubaraFunction(grid_P, 2, 2, 2, 2)
        η_D    = MatsubaraFunction(grid_P, 2, 2, 2, 2)
        η_M    = MatsubaraFunction(grid_P, 2, 2, 2, 2)

        set!(P_S, 0.0)
        set!(P_T, 0.0)
        set!(P_D, 0.0)
        set!(P_M, 0.0)

        for w in grid_P 
            η_S.data[index(w), :, :, :, :] .= F0_S
            η_T.data[index(w), :, :, :, :] .= F0_T
            η_D.data[index(w), :, :, :, :] .= F0_D
            η_M.data[index(w), :, :, :, :] .= F0_M
        end

        # initialization of λ 
        grid_λ_w  = MatsubaraGrid(T, num_λ_w, Boson)
        grid_λ_v  = MatsubaraGrid(T, num_λ_v, Fermion)
        λ_S       = MatsubaraFunction((grid_λ_w, grid_λ_v), 2, 2, 2, 2)
        λ_T       = MatsubaraFunction((grid_λ_w, grid_λ_v), 2, 2, 2, 2)
        λ_D       = MatsubaraFunction((grid_λ_w, grid_λ_v), 2, 2, 2, 2)
        λ_M       = MatsubaraFunction((grid_λ_w, grid_λ_v), 2, 2, 2, 2)
        λ_S_dummy = MatsubaraFunction((grid_λ_w, grid_λ_v), 2, 2, 2, 2)
        λ_T_dummy = MatsubaraFunction((grid_λ_w, grid_λ_v), 2, 2, 2, 2)
        λ_D_dummy = MatsubaraFunction((grid_λ_w, grid_λ_v), 2, 2, 2, 2)
        λ_M_dummy = MatsubaraFunction((grid_λ_w, grid_λ_v), 2, 2, 2, 2)

        for w in grid_λ_w, v in grid_λ_v 
            λ_S_w_v = view(λ_S, (w, v), :, :, :, :)
            λ_T_w_v = view(λ_T, (w, v), :, :, :, :)
            λ_D_w_v = view(λ_D, (w, v), :, :, :, :)
            λ_M_w_v = view(λ_M, (w, v), :, :, :, :)

            @tullio λ_S_w_v[x1p, x1, x2p, x2] = δ[x1p, x1] * δ[x2p, x2]
            @tullio λ_T_w_v[x1p, x1, x2p, x2] = δ[x1p, x1] * δ[x2p, x2]
            @tullio λ_D_w_v[x1p, x1, x2p, x2] = δ[x2p, x1] * δ[x1p, x2]
            @tullio λ_M_w_v[x1p, x1, x2p, x2] = δ[x2p, x1] * δ[x1p, x2]
        end

        # initialization of Σ
        grid_Σ = MatsubaraGrid(T, num_Σ, Fermion)
        Σ      = MatsubaraFunction(grid_Σ, 2, 2)
        set!(Σ, 0.0)

        # build the solver 
        return new(T, U, J, Δ, μ,
                m, p, α, atol, rtol, iters, 
                G0, G, Σ,
                F0_S, F0_T, F0_D, F0_M,
                P_S, P_T, P_D, P_M, 
                η_S, η_T, η_D, η_M, 
                λ_S, λ_T, λ_D, λ_M, 
                λ_S_dummy, λ_T_dummy, λ_D_dummy, λ_M_dummy)
    end
end

function Base.:length(
    S :: Solver
    ) :: Int64 

    return length(S.Σ) + 4 * length(S.P_S) + 4 * length(S.λ_S)
end

function MatsubaraFunctions.:flatten!(
    S :: Solver, 
    x :: Vector{ComplexF64}
    ) :: Nothing 

    @assert length(S) == length(x) "Length of flattened solver and input vector do not match"

    offset = 0
    flatten!(S.Σ, view(x, offset + 1 : offset + length(S.Σ)))
    offset += length(S.Σ)

    flatten!(S.P_S, view(x, offset + 1 : offset + length(S.P_S)))
    offset += length(S.P_S)

    flatten!(S.P_T, view(x, offset + 1 : offset + length(S.P_T)))
    offset += length(S.P_T)

    flatten!(S.P_D, view(x, offset + 1 : offset + length(S.P_D)))
    offset += length(S.P_D)

    flatten!(S.P_M, view(x, offset + 1 : offset + length(S.P_M)))
    offset += length(S.P_M)

    flatten!(S.λ_S, view(x, offset + 1 : offset + length(S.λ_S)))
    offset += length(S.λ_S)

    flatten!(S.λ_T, view(x, offset + 1 : offset + length(S.λ_T)))
    offset += length(S.λ_T)

    flatten!(S.λ_D, view(x, offset + 1 : offset + length(S.λ_D)))
    offset += length(S.λ_D)

    flatten!(S.λ_M, view(x, offset + 1 : offset + length(S.λ_M)))
    offset += length(S.λ_M)

    @assert length(S) == offset "Length of flattened solver and final offset do not match"
    return nothing 
end

function MatsubaraFunctions.:flatten(
    S :: Solver
    ) :: Vector{ComplexF64}

    x = Vector{ComplexF64}(undef, length(S))
    flatten!(S, x)

    return x
end 

function MatsubaraFunctions.:unflatten!(
    S :: Solver, 
    x :: Vector{ComplexF64}
    ) :: Nothing 

    @assert length(S) == length(x) "Length of flattened solver and input vector do not match"

    offset = 0
    unflatten!(S.Σ, view(x, offset + 1 : offset + length(S.Σ)))
    offset += length(S.Σ)

    unflatten!(S.P_S, view(x, offset + 1 : offset + length(S.P_S)))
    offset += length(S.P_S)

    unflatten!(S.P_T, view(x, offset + 1 : offset + length(S.P_T)))
    offset += length(S.P_T)

    unflatten!(S.P_D, view(x, offset + 1 : offset + length(S.P_D)))
    offset += length(S.P_D)

    unflatten!(S.P_M, view(x, offset + 1 : offset + length(S.P_M)))
    offset += length(S.P_M)

    unflatten!(S.λ_S, view(x, offset + 1 : offset + length(S.λ_S)))
    offset += length(S.λ_S)

    unflatten!(S.λ_T, view(x, offset + 1 : offset + length(S.λ_T)))
    offset += length(S.λ_T)

    unflatten!(S.λ_D, view(x, offset + 1 : offset + length(S.λ_D)))
    offset += length(S.λ_D)

    unflatten!(S.λ_M, view(x, offset + 1 : offset + length(S.λ_M)))
    offset += length(S.λ_M)

    @assert length(S) == offset "Length of flattened solver and final offset do not match"
    return nothing 
end

# perform a GW iteration
function GW!(
    S :: Solver
    ) :: Nothing 

    calc_P_pp!(S.P_S, S.G, S.λ_S)
    calc_P_pp!(S.P_T, S.G, S.λ_T)
    calc_P_ph!(S.P_D, S.G, S.λ_D)
    calc_P_ph!(S.P_M, S.G, S.λ_M)

    calc_η_pp!(S.η_S, S.P_S, S.F0_S)
    calc_η_pp!(S.η_T, S.P_T, S.F0_T)
    calc_η_ph!(S.η_D, S.P_D, S.F0_D)
    calc_η_ph!(S.η_M, S.P_M, S.F0_M)

    calc_Σ!(S.Σ, S.G, S.η_D, S.λ_D, S.η_M, S.λ_M, S.F0_D, S.F0_M)

    return nothing 
end

# implementation of fixed-point equation
function fixed_point!(
    F :: Vector{ComplexF64},
    x :: Vector{ComplexF64},
    S :: Solver
    ) :: Nothing 

    # update solver
    unflatten!(S, x)

    # update G
    calc_G!(S.G, S.G0, S.Σ)

    # calculate P
    calc_P_pp!(S.P_S, S.G, S.λ_S)
    calc_P_pp!(S.P_T, S.G, S.λ_T)
    calc_P_ph!(S.P_D, S.G, S.λ_D)
    calc_P_ph!(S.P_M, S.G, S.λ_M)

    # calculate Σ
    calc_Σ!(S.Σ, S.G, S.η_D, S.λ_D, S.η_M, S.λ_M, S.F0_D, S.F0_M)

    # calculate λ
    calc_λ!(S.λ_S_dummy, S.G, S.η_D, S.λ_D, S.η_M, S.λ_M, S.F0_S, ch_S)
    calc_λ!(S.λ_T_dummy, S.G, S.η_D, S.λ_D, S.η_M, S.λ_M, S.F0_T, ch_T)
    calc_λ!(S.λ_D_dummy, S.G, S.η_S, S.λ_S, S.η_T, S.λ_T, S.η_D, S.λ_D, S.η_M, S.λ_M, S.F0_D, ch_D)
    calc_λ!(S.λ_M_dummy, S.G, S.η_S, S.λ_S, S.η_T, S.λ_T, S.η_D, S.λ_D, S.η_M, S.λ_M, S.F0_M, ch_M)

    # update η
    calc_η_pp!(S.η_S, S.P_S, S.F0_S)
    calc_η_pp!(S.η_T, S.P_T, S.F0_T)
    calc_η_ph!(S.η_D, S.P_D, S.F0_D)
    calc_η_ph!(S.η_M, S.P_M, S.F0_M)

    # update λ   
    set!(S.λ_S, S.λ_S_dummy)
    set!(S.λ_T, S.λ_T_dummy)
    set!(S.λ_D, S.λ_D_dummy)
    set!(S.λ_M, S.λ_M_dummy)

    # compute residue
    flatten!(S, F)
    F .-= x 

    mpi_barrier()
    return nothing 
end

# run the solver and find the fixed point 
function solve!(
    S :: Solver
    ) :: Nothing 

    ti = time()
    PP = PeriodicPulay(flatten(S); m = S.m)
    mpi_println("")

    MatsubaraFunctions.solve!((F, x) -> fixed_point!(F, x, S), PP,
        p       = S.p,
        iters   = S.iters,
        α       = S.α,
        atol    = S.atol,
        rtol    = S.rtol,
        verbose = true)
    
    dt = time() - ti
    unflatten!(S, PP.x)

    mpi_println("")
    mpi_println("Done. Time elapsed $(dt)s.")

    return nothing 
end

# save the solver to file
function save_solver!(
    file :: HDF5.File, 
    S    :: Solver
    )    :: Nothing

    attributes(file)["T"]     = S.T 
    attributes(file)["U"]     = S.U 
    attributes(file)["J"]     = S.J
    attributes(file)["Δ"]     = S.Δ
    attributes(file)["μ"]     = S.μ
    attributes(file)["m"]     = S.m
    attributes(file)["p"]     = S.p
    attributes(file)["α"]     = S.α
    attributes(file)["atol"]  = S.atol 
    attributes(file)["rtol"]  = S.rtol 
    attributes(file)["iters"] = S.iters

    save_matsubara_function!(file, "G0", S.G0)
    save_matsubara_function!(file, "G", S.G)
    save_matsubara_function!(file, "Σ", S.Σ)

    file["F0_S"] = S.F0_S
    file["F0_T"] = S.F0_T
    file["F0_D"] = S.F0_D
    file["F0_M"] = S.F0_M

    save_matsubara_function!(file, "P_S", S.P_S)
    save_matsubara_function!(file, "P_T", S.P_T)
    save_matsubara_function!(file, "P_D", S.P_D)
    save_matsubara_function!(file, "P_M", S.P_M)

    save_matsubara_function!(file, "η_S", S.η_S)
    save_matsubara_function!(file, "η_T", S.η_T)
    save_matsubara_function!(file, "η_D", S.η_D)
    save_matsubara_function!(file, "η_M", S.η_M)

    save_matsubara_function!(file, "λ_S", S.λ_S)
    save_matsubara_function!(file, "λ_T", S.λ_T)
    save_matsubara_function!(file, "λ_D", S.λ_D)
    save_matsubara_function!(file, "λ_M", S.λ_M)

    return nothing 
end