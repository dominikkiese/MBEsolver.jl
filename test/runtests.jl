using MBEsolver
using MatsubaraFunctions
using HDF5
using MPI
using Test
MPI.Init()

@testset "RefNoSym" begin
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
    m       = 5
    p       = 3
    α       = 0.5
    atol    = 1e-8
    rtol    = 1e-8
    iters   = 25

    # build the solver 
    S = MBEsolver.Solver(T, U, V, D, num_G, num_Σ, num_P, num_λ_w, num_λ_v, num_M_w, num_M_v; m, p, α, atol, rtol, iters)

    # run the solver 
    MBEsolver.solve!(S)
    
    # compare to reference data
    ref_file = h5open(joinpath(dirname(@__FILE__), "refnosym.h5"), "r")

    @test S.G0  == load_matsubara_function(ref_file, "G0")
    @test S.G   == load_matsubara_function(ref_file, "G")
    @test S.Σ   == load_matsubara_function(ref_file, "Σ")
    @test S.P_S == load_matsubara_function(ref_file, "P_S")
    @test S.P_D == load_matsubara_function(ref_file, "P_D")
    @test S.P_M == load_matsubara_function(ref_file, "P_M")
    @test S.η_S == load_matsubara_function(ref_file, "η_S")
    @test S.η_D == load_matsubara_function(ref_file, "η_D")
    @test S.η_M == load_matsubara_function(ref_file, "η_M")
    @test S.λ_S == load_matsubara_function(ref_file, "λ_S")
    @test S.λ_D == load_matsubara_function(ref_file, "λ_D")
    @test S.λ_M == load_matsubara_function(ref_file, "λ_M")
    @test S.M_S == load_matsubara_function(ref_file, "M_S")
    @test S.M_T == load_matsubara_function(ref_file, "M_T")
    @test S.M_D == load_matsubara_function(ref_file, "M_D")
    @test S.M_M == load_matsubara_function(ref_file, "M_M")
    
    mpi_println("")
    close(ref_file)
end

@testset "RefSym" begin
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
    m       = 5
    p       = 3
    α       = 0.5
    atol    = 1e-8
    rtol    = 1e-8
    iters   = 25

    # build the solver 
    S = MBEsolver.Solver(T, U, V, D, num_G, num_Σ, num_P, num_λ_w, num_λ_v, num_M_w, num_M_v; m, p, α, atol, rtol, iters)

    # calculate the symmetry groups 
    sym_λ_p = [MatsubaraSymmetry{2, 1}(s1_λ_p), MatsubaraSymmetry{2, 1}(s2_λ_p)]
    sym_λ_d = [MatsubaraSymmetry{2, 1}(s1_λ_d), MatsubaraSymmetry{2, 1}(s2_λ_d)]
    sym_M_S = [MatsubaraSymmetry{3, 1}(s1_M_S), MatsubaraSymmetry{3, 1}(s2_M_S), MatsubaraSymmetry{3, 1}(s3_M_p), MatsubaraSymmetry{3, 1}(s4_M_p)]
    sym_M_T = [MatsubaraSymmetry{3, 1}(s1_M_T), MatsubaraSymmetry{3, 1}(s2_M_T), MatsubaraSymmetry{3, 1}(s3_M_p), MatsubaraSymmetry{3, 1}(s4_M_p)]
    sym_M_d = [MatsubaraSymmetry{3, 1}(s1_M_d), MatsubaraSymmetry{3, 1}(s2_M_d), MatsubaraSymmetry{3, 1}(s3_M_d)]
    init_sym_grp!(S, sym_λ_p, sym_λ_d, sym_M_S, sym_M_T, sym_M_d)

    # run the solver 
    MBEsolver.solve!(S)

    # compare to reference data
    ref_file = h5open(joinpath(dirname(@__FILE__), "refsym.h5"), "r")

    @test S.G0  == load_matsubara_function(ref_file, "G0")
    @test S.G   == load_matsubara_function(ref_file, "G")
    @test S.Σ   == load_matsubara_function(ref_file, "Σ")
    @test S.P_S == load_matsubara_function(ref_file, "P_S")
    @test S.P_D == load_matsubara_function(ref_file, "P_D")
    @test S.P_M == load_matsubara_function(ref_file, "P_M")
    @test S.η_S == load_matsubara_function(ref_file, "η_S")
    @test S.η_D == load_matsubara_function(ref_file, "η_D")
    @test S.η_M == load_matsubara_function(ref_file, "η_M")
    @test S.λ_S == load_matsubara_function(ref_file, "λ_S")
    @test S.λ_D == load_matsubara_function(ref_file, "λ_D")
    @test S.λ_M == load_matsubara_function(ref_file, "λ_M")
    @test S.M_S == load_matsubara_function(ref_file, "M_S")
    @test S.M_T == load_matsubara_function(ref_file, "M_T")
    @test S.M_D == load_matsubara_function(ref_file, "M_D")
    @test S.M_M == load_matsubara_function(ref_file, "M_M")

    mpi_println("")
    close(ref_file)
end