importall MathProgBase.SolverInterface

export PipsNlpSolver
immutable PipsNlpSolver <: AbstractMathProgSolver
    options
end
PipsNlpSolver(;kwargs...) = PipsNlpSolver(kwargs)

type PipsNlpMathProgModel <: AbstractMathProgModel
    inner::Any
    state::Symbol # Uninitialized, LoadLinear, LoadNonlinear
    numvar::Int
    numconstr::Int
    warmstart::Vector{Float64}
    options
end
function PipsNlpMathProgModel(;options...)
    return PipsNlpMathProgModel(nothing,:Uninitialized,0,0,Float64[],options)
end
model(s::PipsNlpSolver) = PipsNlpMathProgModel(;s.options...)
export model

NonlinearModel(s::PipsNlpSolver) = PipsNlpMathProgModel(;s.options...)
LinearQuadraticModel(s::PipsNlpSolver) = NonlinearToLPQPBridge(NonlinearModel(s))

###############################################################################
# Begin interface implementation

function array_copy(src,dest)
    @show src
    @show dest
    assert(length(src)==length(dest))
    for i in 1:length(src)
        dest[i] = src[i]-1
    end    
end

# generic nonlinear interface
function loadproblem!(m::PipsNlpMathProgModel, numVar::Integer, numConstr::Integer, x_l, x_u, g_lb, g_ub, sense::Symbol, d::AbstractNLPEvaluator)

    initialize(d, [:Grad, :Jac, :Hess])
    Ijac, Jjac = jac_structure(d)
    Ihess, Jhess = hesslag_structure(d)
    @assert length(Ijac) == length(Jjac)
    @assert length(Ihess) == length(Jhess)
    @assert sense == :Min || sense == :Max

    # Objective callback
    if sense == :Min
        eval_f_cb(x) = eval_f(d,x)
    else
        eval_f_cb(x) = -eval_f(d,x)
    end

    # Objective gradient callback
    if sense == :Min
        eval_grad_f_cb(x, grad_f) = eval_grad_f(d, grad_f, x)
    else
        eval_grad_f_cb(x, grad_f) = (eval_grad_f(d, grad_f, x); scale!(grad_f,-1))
    end

    # Constraint value callback
    eval_g_cb(x, g) = eval_g(d, g, x)

    # Jacobian callback
    function eval_jac_g_cb(x, mode, irows, kcols, values)
        if mode == :Structure
            @show Ijac
            @show Jjac
            mat = sparse(Ijac, Jjac, ones(Float64,length(Ijac)))
            @show mat
			array_copy(mat.colptr, kcols)
            array_copy(mat.rowval, irows)
        else
			vals=zeros(Float64,length(Ijac))
            eval_jac_g(d, vals, x)
            mat = sparse(Ijac,Jjac,vals)
            array_copy(mat.nzval,values)
        end
    end

    # Hessian callback
    function eval_h_cb(x, mode, irows, kcols, obj_factor, lambda, values)
        if mode == :Structure
            @show Ihess
            @show Jhess
			mat = sparse(Ihess, Jhess, ones(Float64,length(Ihess)))
            @show mat
			array_copy(mat.colptr, kcols)
            array_copy(mat.rowval, irows)
        else
            if sense == :Max
                obj_factor *= -1
            end

			vals=zeros(Float64,length(Ihess))
            eval_hesslag(d, vals, x, obj_factor, lambda)
            mat = sparse(Ihess,Jhess,vals)
            @show vals,Ihess,Jhess
            @show mat
			array_copy(mat.nzval,values)
        end
    end

    st_jac_mat = sparse(Ijac,Jjac,ones(Float64,length(Ijac)))
    st_hess_mat = sparse(Ihess,Jhess,ones(Float64,length(Ihess)))
    m.inner = createProblem(numVar, numConstr,
            x_l, x_u, g_lb, g_ub, 
            length(st_jac_mat.nzval), length(st_hess_mat.nzval),
            eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb, eval_h_cb)
    
    m.inner.sense = sense
    
    m.state = :LoadNonlinear
end

getsense(m::PipsNlpMathProgModel) = m.inner.sense
numvar(m::PipsNlpMathProgModel) = m.numvar
numconstr(m::PipsNlpMathProgModel) = m.numconstr

function optimize!(m::PipsNlpMathProgModel)
    @assert m.state == :LoadNonlinear
    copy!(m.inner.x, m.warmstart) # set warmstart
    solveProblem(m.inner)
end

function status(m::PipsNlpMathProgModel)
	return :Optimal
end

getobjval(m::PipsNlpMathProgModel) = m.inner.obj_val * (m.inner.sense == :Max ? -1 : +1)
getsolution(m::PipsNlpMathProgModel) = m.inner.x


getrawsolver(m::PipsNlpMathProgModel) = m.inner
setwarmstart!(m::PipsNlpMathProgModel, x) = (m.warmstart = x)


