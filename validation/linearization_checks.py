


def testTLMLinearity(tlm, primalArg, tangentArg, scale, norm, tol):
    u = primalArg()
    du = tangentArg()
    
    _, dv = tlm(u, du)
    _, dv2 = tlm(u, scale(2.0, du))
    
    absolute_error = abs(norm(scale(2.0, dv)) - norm(dv2))

    return absolute_error < tol, absolute_error


def testTLMApprox(m, tlm, primalArg, tangentArg, scale, add, norm, tol):
    u = primalArg()
    du = tangentArg()

    def axpyOp(a,x,y):
        return add(scale(a,x),y)
    
    wa = m(u)
    _, dv = tlm(u, du)

    c = 1.0

    model_pert_norms = []
    absolute_errors = []
    relative_errors = []
    for _ in range(15):
        wb = m(axpyOp(c, du, u))
        dw = axpyOp(-1.0, wa, wb)
        
        model_pert_norm = norm(dw)
        model_pert_norms.append(model_pert_norm)
        absolute_error = norm(axpyOp(-c, dv, dw))
        absolute_errors.append(absolute_error)
        relative_error = absolute_error / model_pert_norm
        relative_errors.append(relative_error)

        c /= 10.0

    min_relative_error = min(relative_errors)

    return min_relative_error < tol, min_relative_error


def testADMApprox(tlm, adm, primalArg, tangentArg, cotangentArg, dot, tol):
    from copy import deepcopy

    u = primalArg()
    u_cpy = deepcopy(u)
    du = tangentArg()
    Dv = cotangentArg()

    _, dv = tlm(u, du)
    
    _, Du = adm(u_cpy, Dv)
  
    absolute_error = abs(dot(dv, Dv) - dot(du, Du))
    return absolute_error < tol, absolute_error

