

def testTLMLinearity(tlm, randomInput, axpyOp, norm, tol):
    u0 = randomInput()
    du0 = randomInput()
    
    v, dv = tlm(u0, du0)
    v2, dv2 = tlm(u0, axpyOp(2.0, du0, None))
    
    absolute_error = norm(axpyOp(-2.0, dv, dv2))
    
    return absolute_error < tol, absolute_error


def testTLMApprox(m, tlm, randomInput, axpyOp, norm, tol):
    u0 = randomInput()
    du0 = randomInput()
    
    wa = m(u0)
    v, dv = tlm(u0, du0)

    scale = 1.0

    absolute_errors = []
    relavite_errors = []
    for _ in range(15):
        wb = m(axpyOp(scale, du0, u0))
        
        absolute_error = norm(axpyOp(scale, dv, axpyOp(-1.0, wb, wa)))
        absolute_errors.append(absolute_error)
        relative_error = absolute_error / norm(axpyOp(-1.0, wa, wb))
        relavite_errors.append(relative_error)
        scale /= 10.0

    min_relative_error = min(relavite_errors)

    return min_relative_error < tol, min_relative_error


def testADMApprox(tlm, adm, randomInput, randomOutput, dot, tol):
    u0 = randomInput()
    du0 = randomInput()
    Dv = randomOutput()

    v, dv = tlm(u0, du0)
    
    v, Du0 = adm(u0, Dv)
    
    absolute_error = abs(dot(dv, Dv) - dot(du0, Du0))
    return absolute_error < tol, absolute_error

