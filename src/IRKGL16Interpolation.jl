"""
    IRKGLInterpolation

Dense output of an [`IRKGL16`](@ref) solution: on each step `[t[i], t[i+1]]` the
solution is evaluated with the collocation polynomial of that step (the same
polynomial used for `saveat`), `u[i] + sum_j kappa_j(θ) * L[i][j, :]`, where
`θ = |t - t[i]| / h[i]`. At the saved time points the saved values are returned.
Only the solution itself (`Val{0}`) can be interpolated.
"""
struct IRKGLInterpolation{tType, uType, LType} <: SciMLBase.AbstractDiffEqInterpolation
    t::Vector{tType}
    u::Vector{uType}
    h::Vector{tType}
    L::Vector{LType}
    X2::Vector{tType}
    Y2::Matrix{tType}
end

SciMLBase.interp_summary(::IRKGLInterpolation) = "Gauss-Legendre collocation polynomial"

function _store_step_L!(Ls, L, indices, s, use_simd)
    M = Matrix{eltype(eltype(Ls))}(undef, s, length(indices))
    for (n, k) in enumerate(indices)
        if use_simd
            Lk = L[k]
            for j in 1:s
                M[j, n] = Lk[j]
            end
        else
            for j in 1:s
                M[j, n] = L[j][k]
            end
        end
    end
    push!(Ls, M)
    return nothing
end

function _irkgl_interp_point(id::IRKGLInterpolation, tval, idxs)
    t = id.t
    tdir = sign(t[end] - t[1])
    if tdir * tval > tdir * t[end] || tdir * tval < tdir * t[1]
        error("Solution interpolation cannot extrapolate outside of the time span [$(t[1]), $(t[end])].")
    end
    i = searchsortedfirst(t, tval; rev = tdir < 0)
    if t[i] == tval
        u = id.u[i]
    else
        step = i - 1
        u0 = id.u[step]
        Lstep = id.L[step]
        θ = convert(eltype(id.X2), abs((tval - t[step]) / id.h[step]))
        kappa = PolInterp(id.X2, id.Y2, [θ])
        u = copy(u0)
        for (n, k) in enumerate(eachindex(u0))
            acc = kappa[1] * Lstep[1, n]
            for j in 2:size(Lstep, 1)
                acc = muladd(kappa[j], Lstep[j, n], acc)
            end
            u[k] = u0[k] + acc
        end
    end
    return idxs === nothing ? copy(u) : u[idxs]
end

function _check_deriv(deriv)
    return deriv === Val{0} ||
        throw(ArgumentError("IRKGL16 dense output only supports interpolating the solution (Val{0}), got $deriv."))
end

_irkgl_interp(id, tval::Number, idxs) = _irkgl_interp_point(id, tval, idxs)
function _irkgl_interp(id, tvals, idxs)
    return DiffEqArray([_irkgl_interp_point(id, tv, idxs) for tv in tvals], tvals)
end

function _irkgl_interp!(val, id, tval::Number, idxs)
    val .= _irkgl_interp_point(id, tval, idxs)
    return val
end
function _irkgl_interp!(vals, id, tvals, idxs)
    for (j, tv) in enumerate(tvals)
        vals[j] = _irkgl_interp_point(id, tv, idxs)
    end
    return vals
end

function (id::IRKGLInterpolation)(tvals, idxs, deriv, p, continuity::Symbol)
    _check_deriv(deriv)
    return _irkgl_interp(id, tvals, idxs)
end

function (id::IRKGLInterpolation)(val, tvals, idxs, deriv, p, continuity::Symbol)
    _check_deriv(deriv)
    return _irkgl_interp!(val, id, tvals, idxs)
end
