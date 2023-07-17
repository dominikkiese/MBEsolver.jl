mutable struct Solver
    # numerical parameters
    T       :: Float64 
    U       :: Float64 
    V       :: Float64
    D       :: Float64
    num_Σ   :: Int64
    num_P   :: Int64
    mem     :: Int64 
    α       :: Float64 
    tol     :: Float64
    maxiter :: Int64

    # propagators
    G0 :: MatsubaraFunction{1, 1, 2, Float64}
    G  :: MatsubaraFunction{1, 1, 2, Float64}
    Σ  :: MatsubaraFunction{1, 1, 2, Float64}

    # polarizations
    P_S :: MatsubaraFunction{1, 1, 2, Float64}
    P_D :: MatsubaraFunction{1, 1, 2, Float64}
    P_M :: MatsubaraFunction{1, 1, 2, Float64}

    # screened interactions
    η_S :: MatsubaraFunction{1, 1, 2, Float64}
    η_D :: MatsubaraFunction{1, 1, 2, Float64}
    η_M :: MatsubaraFunction{1, 1, 2, Float64}

    # Hedin vertices and their buffers for inplace calculations
    λ_S       :: MatsubaraFunction{2, 1, 3, Float64}
    λ_D       :: MatsubaraFunction{2, 1, 3, Float64}
    λ_M       :: MatsubaraFunction{2, 1, 3, Float64}
    λ_S_dummy :: MatsubaraFunction{2, 1, 3, Float64}
    λ_D_dummy :: MatsubaraFunction{2, 1, 3, Float64}
    λ_M_dummy :: MatsubaraFunction{2, 1, 3, Float64}

    # buffers for irreducible vertices
    T_S :: MatsubaraFunction{3, 1, 4, Float64}
    T_T :: MatsubaraFunction{3, 1, 4, Float64}
    T_D :: MatsubaraFunction{3, 1, 4, Float64}
    T_M :: MatsubaraFunction{3, 1, 4, Float64}

    # multiboson vertices and their buffers for inplace calculations
    M_S       :: MatsubaraFunction{3, 1, 4, Float64}
    M_T       :: MatsubaraFunction{3, 1, 4, Float64}
    M_D       :: MatsubaraFunction{3, 1, 4, Float64}
    M_M       :: MatsubaraFunction{3, 1, 4, Float64}
    M_S_dummy :: MatsubaraFunction{3, 1, 4, Float64}
    M_T_dummy :: MatsubaraFunction{3, 1, 4, Float64}
    M_D_dummy :: MatsubaraFunction{3, 1, 4, Float64}
    M_M_dummy :: MatsubaraFunction{3, 1, 4, Float64}

    # vertex symmetry groups
    SG_λ_p :: MatsubaraSymmetryGroup 
    SG_λ_d :: MatsubaraSymmetryGroup
    SG_M_S :: MatsubaraSymmetryGroup 
    SG_M_T :: MatsubaraSymmetryGroup 
    SG_M_d :: MatsubaraSymmetryGroup

    # convenience constructor for the Solver (default constructor implicitly deleted)
    function Solver(
        T       :: Float64,
        U       :: Float64,
        V       :: Float64,
        D       :: Float64,
        num_G   :: Int64,
        num_Σ   :: Int64, 
        num_P   :: Int64, 
        num_λ_w :: Int64,  
        num_λ_v :: Int64, 
        num_M_w :: Int64,
        num_M_v :: Int64,
        ;
        mem     :: Int64   = 8,
        α       :: Float64 = 0.85,
        tol     :: Float64 = 1e-4,   
        maxiter :: Int64   = 100
        )       :: Solver

        # sanity checks 
        @assert num_Σ <= num_λ_v "num_Σ <= num_λ_v required"
        @assert num_λ_w > num_λ_v "num_λ_w > num_λ_v required"
        @assert num_λ_w > num_M_w "num_λ_w > num_M_w required"
        @assert num_λ_v > num_M_v "num_λ_v > num_M_v required"
        @assert num_M_w > num_M_v "num_M_w > num_M_v required"

        # initialization of G0 and G 
        grid_G = MatsubaraGrid(T, num_G, Fermion)
        G0     = MatsubaraFunction(grid_G, 1, Float64)
        G      = MatsubaraFunction(grid_G, 1, Float64)

        for v in grid_G
            G0[v] = 1.0 / (value(v) + V * V / D * atan(D / value(v)))
        end 

        set!(G, G0)

        # initialization of P, η
        grid_P = MatsubaraGrid(T, num_P, Boson)
        P_S    = MatsubaraFunction(grid_P, 1, Float64)
        P_D    = MatsubaraFunction(grid_P, 1, Float64)
        P_M    = MatsubaraFunction(grid_P, 1, Float64)
        η_S    = MatsubaraFunction(grid_P, 1, Float64)
        η_D    = MatsubaraFunction(grid_P, 1, Float64)
        η_M    = MatsubaraFunction(grid_P, 1, Float64)

        set!(P_S, 0.0)
        set!(P_D, 0.0)
        set!(P_M, 0.0)

        set!(η_S, +2.0 * U)
        set!(η_D, +1.0 * U)
        set!(η_M, -1.0 * U)

        # initialization of λ 
        grid_λ_w  = MatsubaraGrid(T, num_λ_w, Boson)
        grid_λ_v  = MatsubaraGrid(T, num_λ_v, Fermion)
        λ_S       = MatsubaraFunction((grid_λ_w, grid_λ_v), 1, Float64)
        λ_D       = MatsubaraFunction((grid_λ_w, grid_λ_v), 1, Float64)
        λ_M       = MatsubaraFunction((grid_λ_w, grid_λ_v), 1, Float64)
        λ_S_dummy = MatsubaraFunction((grid_λ_w, grid_λ_v), 1, Float64)
        λ_D_dummy = MatsubaraFunction((grid_λ_w, grid_λ_v), 1, Float64)
        λ_M_dummy = MatsubaraFunction((grid_λ_w, grid_λ_v), 1, Float64)

        set!(λ_S, 1.0)  
        set!(λ_D, 1.0) 
        set!(λ_M, 1.0)

        # initialization of T 
        grid_T = MatsubaraGrid(T, num_λ_w + num_λ_v + num_P, Fermion)
        T_S    = MatsubaraFunction((grid_λ_w, grid_T, grid_λ_v), 1, Float64)
        T_T    = MatsubaraFunction((grid_λ_w, grid_T, grid_λ_v), 1, Float64)
        T_D    = MatsubaraFunction((grid_λ_w, grid_λ_v, grid_T), 1, Float64)
        T_M    = MatsubaraFunction((grid_λ_w, grid_λ_v, grid_T), 1, Float64)

        set!(T_S, 0.0)
        set!(T_T, 0.0)
        set!(T_D, 0.0)
        set!(T_M, 0.0)

        # initialization of M
        grid_M_w  = MatsubaraGrid(T, num_M_w, Boson)
        grid_M_v  = MatsubaraGrid(T, num_M_v, Fermion)
        M_S       = MatsubaraFunction((grid_M_w, grid_M_v, grid_M_v), 1, Float64)
        M_T       = MatsubaraFunction((grid_M_w, grid_M_v, grid_M_v), 1, Float64)
        M_D       = MatsubaraFunction((grid_M_w, grid_M_v, grid_M_v), 1, Float64)
        M_M       = MatsubaraFunction((grid_M_w, grid_M_v, grid_M_v), 1, Float64)
        M_S_dummy = MatsubaraFunction((grid_M_w, grid_M_v, grid_M_v), 1, Float64)
        M_T_dummy = MatsubaraFunction((grid_M_w, grid_M_v, grid_M_v), 1, Float64)
        M_D_dummy = MatsubaraFunction((grid_M_w, grid_M_v, grid_M_v), 1, Float64)
        M_M_dummy = MatsubaraFunction((grid_M_w, grid_M_v, grid_M_v), 1, Float64)

        set!(M_S, 0.0) 
        set!(M_T, 0.0) 
        set!(M_D, 0.0) 
        set!(M_M, 0.0)

        # initialization of Σ (using 2nd order PBT)
        grid_Σ = MatsubaraGrid(T, num_Σ, Fermion)
        Σ      = MatsubaraFunction(grid_Σ, 1, Float64)

        P_D0 = calc_P(λ_D, G0, num_P, ch_D); η_D0 = calc_η(P_D0, η_D, +U)
        P_M0 = calc_P(λ_M, G0, num_P, ch_M); η_M0 = calc_η(P_M0, η_M, -U)
        set!(Σ, calc_Σ(G0, η_D0, λ_D, η_M0, λ_M, U, num_Σ))

        # dummy initialization of symmetry groups 
        SG_λ_p = MatsubaraSymmetryGroup(λ_S)
        SG_λ_d = deepcopy(SG_λ_p)
        SG_M_S = MatsubaraSymmetryGroup(M_S)
        SG_M_T = deepcopy(SG_M_S)
        SG_M_d = deepcopy(SG_M_S)

        # build the solver 
        return new(T, U, V, D, num_Σ, num_P, mem, α, tol, maxiter, 
                G0, G, Σ, 
                P_S, P_D, P_M, 
                η_S, η_D, η_M, 
                λ_S, λ_D, λ_M, λ_S_dummy, λ_D_dummy, λ_M_dummy, 
                T_S, T_T, T_D, T_M,
                M_S, M_T, M_D, M_M, M_S_dummy, M_T_dummy, M_D_dummy, M_M_dummy,
                SG_λ_p, SG_λ_d, SG_M_S, SG_M_T, SG_M_d)
    end
end

# symmetry group initialization from list of symmetries 
function init_sym_grp!(
    S       :: Solver,
    sym_λ_p :: Vector{MatsubaraSymmetry{2, 1}},
    sym_λ_d :: Vector{MatsubaraSymmetry{2, 1}},
    sym_M_S :: Vector{MatsubaraSymmetry{3, 1}},
    sym_M_T :: Vector{MatsubaraSymmetry{3, 1}},
    sym_M_d :: Vector{MatsubaraSymmetry{3, 1}}
    )       :: Nothing 

    S.SG_λ_p = MatsubaraSymmetryGroup(sym_λ_p, S.λ_S)
    S.SG_λ_d = MatsubaraSymmetryGroup(sym_λ_d, S.λ_D)
    S.SG_M_S = MatsubaraSymmetryGroup(sym_M_S, S.M_S)
    S.SG_M_T = MatsubaraSymmetryGroup(sym_M_T, S.M_T)
    S.SG_M_d = MatsubaraSymmetryGroup(sym_M_d, S.M_D)

    return nothing 
end

# symmetry group initialization from precomputed symmetry groups 
function init_sym_grp!(
    S      :: Solver,
    SG_λ_p :: MatsubaraSymmetryGroup,
    SG_λ_d :: MatsubaraSymmetryGroup,
    SG_M_S :: MatsubaraSymmetryGroup,
    SG_M_T :: MatsubaraSymmetryGroup,
    SG_M_d :: MatsubaraSymmetryGroup
    )      :: Nothing 

    @assert length(S.λ_S) == sum(length.(SG_λ_p.classes)) "MatsubaraSymmetryGroup incompatible with MatsubaraFunction"
    @assert length(S.λ_D) == sum(length.(SG_λ_d.classes)) "MatsubaraSymmetryGroup incompatible with MatsubaraFunction"
    @assert length(S.M_S) == sum(length.(SG_M_S.classes)) "MatsubaraSymmetryGroup incompatible with MatsubaraFunction"
    @assert length(S.M_T) == sum(length.(SG_M_T.classes)) "MatsubaraSymmetryGroup incompatible with MatsubaraFunction"
    @assert length(S.M_D) == sum(length.(SG_M_d.classes)) "MatsubaraSymmetryGroup incompatible with MatsubaraFunction"

    S.SG_λ_p = SG_λ_p
    S.SG_λ_d = SG_λ_d
    S.SG_M_S = SG_M_S
    S.SG_M_T = SG_M_T
    S.SG_M_d = SG_M_d

    return nothing 
end

# interface with NLsolve
function Base.:length(
    S :: Solver
    ) :: Int64 

    return length(S.Σ) + 3 * length(S.η_S) + 3 * length(S.λ_S) + 4 * length(S.M_S)
end

function MatsubaraFunctions.:flatten!(
    S :: Solver, 
    x :: Vector{Float64}
    ) :: Nothing 

    @assert length(S) == length(x) "Length of flattened solver and input vector do not match"

    offset = 0
    L_Σ    = length(S.Σ)
    L_η    = length(S.η_S)
    L_λ    = length(S.λ_S)
    L_M    = length(S.M_S)

    flatten!(S.Σ, view(x, offset + 1 : offset + L_Σ)); offset += L_Σ 
    flatten!(S.η_S, view(x, offset + 1 : offset + L_η)); offset += L_η
    flatten!(S.η_D, view(x, offset + 1 : offset + L_η)); offset += L_η
    flatten!(S.η_M, view(x, offset + 1 : offset + L_η)); offset += L_η
    flatten!(S.λ_S, view(x, offset + 1 : offset + L_λ)); offset += L_λ
    flatten!(S.λ_D, view(x, offset + 1 : offset + L_λ)); offset += L_λ
    flatten!(S.λ_M, view(x, offset + 1 : offset + L_λ)); offset += L_λ
    flatten!(S.M_S, view(x, offset + 1 : offset + L_M)); offset += L_M
    flatten!(S.M_T, view(x, offset + 1 : offset + L_M)); offset += L_M
    flatten!(S.M_D, view(x, offset + 1 : offset + L_M)); offset += L_M
    flatten!(S.M_M, view(x, offset + 1 : offset + L_M)); offset += L_M

    @assert length(S) == offset "Length of flattened solver and final offset do not match"
    return nothing 
end

function MatsubaraFunctions.:flatten(
    S :: Solver
    ) :: Vector{Float64}

    x = Vector{Float64}(undef, length(S))
    flatten!(S, x)

    return x
end 

function MatsubaraFunctions.:unflatten!(
    S :: Solver, 
    x :: Vector{Float64}
    ) :: Nothing 

    @assert length(S) == length(x) "Length of flattened solver and input vector do not match"

    offset = 0
    L_Σ    = length(S.Σ)
    L_η    = length(S.η_S)
    L_λ    = length(S.λ_S)
    L_M    = length(S.M_S)

    unflatten!(S.Σ, view(x, offset + 1 : offset + L_Σ)); offset += L_Σ 
    unflatten!(S.η_S, view(x, offset + 1 : offset + L_η)); offset += L_η
    unflatten!(S.η_D, view(x, offset + 1 : offset + L_η)); offset += L_η
    unflatten!(S.η_M, view(x, offset + 1 : offset + L_η)); offset += L_η
    unflatten!(S.λ_S, view(x, offset + 1 : offset + L_λ)); offset += L_λ
    unflatten!(S.λ_D, view(x, offset + 1 : offset + L_λ)); offset += L_λ
    unflatten!(S.λ_M, view(x, offset + 1 : offset + L_λ)); offset += L_λ
    unflatten!(S.M_S, view(x, offset + 1 : offset + L_M)); offset += L_M
    unflatten!(S.M_T, view(x, offset + 1 : offset + L_M)); offset += L_M
    unflatten!(S.M_D, view(x, offset + 1 : offset + L_M)); offset += L_M
    unflatten!(S.M_M, view(x, offset + 1 : offset + L_M)); offset += L_M

    @assert length(S) == offset "Length of flattened solver and final offset do not match"
    return nothing 
end

# implementation of fixed-point equation
function fixed_point!(
    F :: Vector{Float64},
    x :: Vector{Float64},
    S :: Solver
    ) :: Nothing 

    # update solver
    unflatten!(S, x)

    # update G
    set!(S.G, calc_G(S.G0, S.Σ))

    # calculate T
    calc_T_pp!(S.T_S, S.T_T, S.η_S, S.λ_S, S.η_D, S.λ_D, S.η_M, S.λ_M, S.M_S, S.M_T, S.M_D, S.M_M, S.U)
    calc_T_ph!(S.T_D, S.T_M, S.η_S, S.λ_S, S.η_D, S.λ_D, S.η_M, S.λ_M, S.M_S, S.M_T, S.M_D, S.M_M, S.U)

    # calculate λ
    calc_λ!(S.λ_S_dummy, S.G, S.T_S, S.SG_λ_p, ch_S)
    calc_λ!(S.λ_D_dummy, S.G, S.T_D, S.SG_λ_d, ch_D)
    calc_λ!(S.λ_M_dummy, S.G, S.T_M, S.SG_λ_d, ch_M)

    # calculate M
    calc_M!(S.M_S_dummy, S.G, S.T_S, S.M_S, S.SG_M_S, ch_S)
    calc_M!(S.M_T_dummy, S.G, S.T_T, S.M_T, S.SG_M_T, ch_T)
    calc_M!(S.M_D_dummy, S.G, S.T_D, S.M_D, S.SG_M_d, ch_D)
    calc_M!(S.M_M_dummy, S.G, S.T_M, S.M_M, S.SG_M_d, ch_M)

    # update P
    set!(S.P_S, calc_P(S.λ_S, S.G, S.num_P, ch_S))
    set!(S.P_D, calc_P(S.λ_D, S.G, S.num_P, ch_D))
    set!(S.P_M, calc_P(S.λ_M, S.G, S.num_P, ch_M))
    
    # calculate η
    set!(S.η_S, calc_η(S.P_S, S.η_S, +2.0 * S.U))
    set!(S.η_D, calc_η(S.P_D, S.η_D, +1.0 * S.U))
    set!(S.η_M, calc_η(S.P_M, S.η_M, -1.0 * S.U))

    # update λ   
    set!(S.λ_S, S.λ_S_dummy)
    set!(S.λ_D, S.λ_D_dummy)
    set!(S.λ_M, S.λ_M_dummy)

    # update M
    set!(S.M_S, S.M_S_dummy)
    set!(S.M_T, S.M_T_dummy)
    set!(S.M_D, S.M_D_dummy)
    set!(S.M_M, S.M_M_dummy)

    # calculate Σ
    set!(S.Σ, calc_Σ(S.G, S.η_D, S.λ_D, S.η_M, S.λ_M, S.U, S.num_Σ))

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

    mpi_println("")
    mpi_println("Running MBE solver ...")
    mpi_println("")

    ti     = time()
    result = nlsolve((F, x) -> fixed_point!(F, x, S), flatten(S),
            method     = :anderson, 
            beta       = S.α, 
            m          = S.mem, 
            ftol       = S.tol, 
            iterations = S.maxiter,
            show_trace = mpi_ismain())
    
    dt = time() - ti
    unflatten!(S, result.zero)

    mpi_println("")
    mpi_println("Done. Calculation took $(dt) seconds.")

    return nothing 
end

# save the solver to file
function save_solver!(
    file :: HDF5.File, 
    S    :: Solver
    )    :: Nothing

    attributes(file)["T"]       = S.T 
    attributes(file)["U"]       = S.U 
    attributes(file)["V"]       = S.V
    attributes(file)["D"]       = S.D
    attributes(file)["num_Σ"]   = S.num_Σ
    attributes(file)["num_P"]   = S.num_P
    attributes(file)["mem"]     = S.mem 
    attributes(file)["α"]       = S.α
    attributes(file)["tol"]     = S.tol 
    attributes(file)["maxiter"] = S.maxiter

    save_matsubara_function!(file, "G0", S.G0)
    save_matsubara_function!(file, "G", S.G)
    save_matsubara_function!(file, "Σ", S.Σ)

    save_matsubara_function!(file, "P_S", S.P_S)
    save_matsubara_function!(file, "P_D", S.P_D)
    save_matsubara_function!(file, "P_M", S.P_M)

    save_matsubara_function!(file, "η_S", S.η_S)
    save_matsubara_function!(file, "η_D", S.η_D)
    save_matsubara_function!(file, "η_M", S.η_M)

    save_matsubara_function!(file, "λ_S", S.λ_S)
    save_matsubara_function!(file, "λ_D", S.λ_D)
    save_matsubara_function!(file, "λ_M", S.λ_M)

    save_matsubara_function!(file, "T_S", S.T_S)
    save_matsubara_function!(file, "T_T", S.T_T)
    save_matsubara_function!(file, "T_D", S.T_D)
    save_matsubara_function!(file, "T_M", S.T_M)

    save_matsubara_function!(file, "M_S", S.M_S)
    save_matsubara_function!(file, "M_T", S.M_T)
    save_matsubara_function!(file, "M_D", S.M_D)
    save_matsubara_function!(file, "M_M", S.M_M)

    save_matsubara_symmetry_group!(file, "SG_λ_p", S.SG_λ_p)
    save_matsubara_symmetry_group!(file, "SG_λ_d", S.SG_λ_d)
    save_matsubara_symmetry_group!(file, "SG_M_S", S.SG_M_S)
    save_matsubara_symmetry_group!(file, "SG_M_T", S.SG_M_T)
    save_matsubara_symmetry_group!(file, "SG_M_d", S.SG_M_d)

    return nothing 
end