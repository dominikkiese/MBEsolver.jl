using MBEsolver
using MatsubaraFunctions
using HDF5
using MPI
using Test
MPI.Init()

@testset "Ref" begin
    # simulation parameters
    T       = 4.0
    U       = 5.75
    V       = 2.0
    D       = 10.0
    num_G   = 64 
    num_Σ   = 32  
    num_P   = 32  
    num_λ_w = 16  
    num_λ_v = 12 
    num_M_w = 12
    num_M_v = 8 
    mem     = 8
    α       = 0.5
    tol     = 1e-4 
    maxiter = 25

    # build the solver 
    S = Solver(T, U, V, D, num_G, num_Σ, num_P, num_λ_w, num_λ_v, num_M_w, num_M_v; mem, α, tol, maxiter)

    # calculate the symmetry groups 
    sym_λ_p = MatsubaraSymmetryGroup([MatsubaraSymmetry{2, 1}(s1_λ_p), MatsubaraSymmetry{2, 1}(s2_λ_p)], S.λ_S)
    sym_λ_d = MatsubaraSymmetryGroup([MatsubaraSymmetry{2, 1}(s1_λ_d), MatsubaraSymmetry{2, 1}(s2_λ_d)], S.λ_D)
    sym_M_p = MatsubaraSymmetryGroup([MatsubaraSymmetry{3, 1}(s1_M_p), MatsubaraSymmetry{3, 1}(s2_M_p), MatsubaraSymmetry{3, 1}(s3_M_p), MatsubaraSymmetry{3, 1}(s4_M_p)], S.M_S)
    sym_M_d = MatsubaraSymmetryGroup([MatsubaraSymmetry{3, 1}(s1_M_d), MatsubaraSymmetry{3, 1}(s2_M_d), MatsubaraSymmetry{3, 1}(s3_M_d)], S.M_D)
    init_sym_grp!(S, sym_λ_p, sym_λ_d, sym_M_p, sym_M_d)

    # run the solver 
    solve!(S)

    # compare to reference data
    ref_file = h5open("ref.h5", "r")

    S.G0  == load_matsubara_function(ref_file, "G0")
    S.G   == load_matsubara_function(ref_file, "G")
    S.Σ   == load_matsubara_function(ref_file, "Σ")
    S.P_S == load_matsubara_function(ref_file, "P_S")
    S.P_D == load_matsubara_function(ref_file, "P_D")
    S.P_M == load_matsubara_function(ref_file, "P_M")
    S.η_S == load_matsubara_function(ref_file, "η_S")
    S.η_D == load_matsubara_function(ref_file, "η_D")
    S.η_M == load_matsubara_function(ref_file, "η_M")
    S.λ_S == load_matsubara_function(ref_file, "λ_S")
    S.λ_D == load_matsubara_function(ref_file, "λ_D")
    S.λ_M == load_matsubara_function(ref_file, "λ_M")
    S.M_S == load_matsubara_function(ref_file, "M_S")
    S.M_T == load_matsubara_function(ref_file, "M_T")
    S.M_D == load_matsubara_function(ref_file, "M_D")
    S.M_M == load_matsubara_function(ref_file, "M_M")
    
    mpi_println("")
    close(ref_file)
end