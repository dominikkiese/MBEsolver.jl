module MBEsolver

    using PrecompileTools

    @recompile_invalidations begin
        using LinearAlgebra
        using StaticArrays
        using MPI
        using Polyester
        using Tullio
        using MatsubaraFunctions
        using HDF5
    end

    include("types.jl")
    include("dyson.jl")
    include("polarization.jl")
    include("Uirreducible.jl")
    include("hedin.jl")
    include("sde.jl")
    include("solver.jl")
end
