module MBEsolver

    using PrecompileTools

    @recompile_invalidations begin
        using MPI
        using MatsubaraFunctions
        using Polyester
        using HDF5
    end

    include("types.jl")
    include("dyson.jl")
    include("polarization.jl")
    include("Uirreducible.jl")
    include("hedin.jl")
    include("multiboson.jl")
    include("sde.jl")
    include("solver.jl")

    @compile_workload begin
        MPI.Init()
        
        # simulation parameters
        T       = 4.0
        U       = 5.75
        V       = 2.0
        D       = 10.0
        num_G   = 64
        num_Σ   = 12  
        num_P   = 32  
        num_λ_w = 16  
        num_λ_v = 12 
        num_M_w = 6
        num_M_v = 4 

        # build the solver 
        S = Solver(T, U, V, D, num_G, num_Σ, num_P, num_λ_w, num_λ_v, num_M_w, num_M_v)

        # calculate the symmetry groups 
        sym_λ_p = [MS2(s1_λ_p), MS2(s2_λ_p)]
        sym_λ_d = [MS2(s1_λ_d), MS2(s2_λ_d)]
        sym_M_S = [MS3(s1_M_S), MS3(s2_M_S), MS3(s3_M_p), MS3(s4_M_p)]
        sym_M_T = [MS3(s1_M_T), MS3(s2_M_T), MS3(s3_M_p), MS3(s4_M_p)]
        sym_M_d = [MS3(s1_M_d), MS3(s2_M_d), MS3(s3_M_d)]
        init_sym_grp!(S, sym_λ_p, sym_λ_d, sym_M_S, sym_M_T, sym_M_d)

        # execute fixed-point kernel 
        fixed_point!(flatten(S), flatten(S), S)
    end
 
    export 
        # hedin
        s1_λ_p,
        s2_λ_p,
        s1_λ_d,
        s2_λ_d,

        # multiboson
        s1_M_S,
        s1_M_T,
        s2_M_S,
        s2_M_T,
        s3_M_p,
        s4_M_p,
        s1_M_d,
        s2_M_d,
        s3_M_d,

        # solver
        init_sym_grp!,
        solve!,
        save_solver!
end
