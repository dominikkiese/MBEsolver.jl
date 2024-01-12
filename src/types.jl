# channel dispatch
abstract type Channel end 
struct ch_S <: Channel end 
struct ch_T <: Channel end
struct ch_D <: Channel end
struct ch_M <: Channel end

# MatsubaraFunction aliases
const G_t = MatsubaraFunction{1, 2, 3, ComplexF64}
const P_t = MatsubaraFunction{1, 4, 5, ComplexF64}
const λ_t = MatsubaraFunction{2, 4, 6, ComplexF64}

# helper function for extrapolation (generalize and put into MatsubaraFunctions?)
function slice_extrp(f :: G_t, v :: AbstractMatsubaraFrequency)
    return SMatrix{2, 2, ComplexF64}(f(v, 1, 1), f(v, 2, 1), f(v, 1, 2), f(v, 2, 2))
end 