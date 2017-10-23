# this global constant controls the order of Taylor expansions but setting it to, say, 4
# is not enough to obtain fourth order MEA because there are other assumptions about order == 3 in the code
ORD = 3

def truncate_to_order(expr, order, dt, dx):
    ret = 0
    for e in expr.expand().iterator():
        ord_e = e.degree(dt) + sum([e.degree(di) for di in dx])
        if ord_e < order:
            ret = ret + e
    return ret

def subs_and_truncate(expr, sub_expr, order, dt, dx):
    expr = expr.substitute(sub_expr)
    expr = truncate_to_order(expr, order, dt, dx)
    return expr

def ret_helper(expr, dt, dx):
    return truncate_to_order(expr.taylor((dt, 0), *([(di, 0) for di in dx] + [ORD])), ORD, dt, dx).simplify()

def define_vars(ndims, const_v = False, const_g = False):
    t, dt = var('t dt')
    x = vector(var('x y z')[0:ndims])
    dx = vector(var('dx dy dz')[0:ndims])

    zeros = vector([0, 0, 0][0:ndims], dx.coordinate_ring())
    e = [copy(zeros) for d in xrange(ndims)]
    for d in xrange(ndims):
        e[d][d] = dx[d]

    psi_f = function('psi')
    psi = lambda t, x : psi_f(t, *x)

    if const_v:
        u_c, v_c, w_c = var('u_0 v_0 w_0')
        v  = [
              lambda t, x : u_c,
              lambda t, x : v_c,
              lambda t, x : w_c
             ][0:ndims]
    else:
        u_f, v_f, w_f = function('u v w')
        v  = [
              lambda t, x : u_f(t, *x),
              lambda t, x : v_f(t, *x),
              lambda t, x : w_f(t, *x)
             ][0:ndims]

    if const_g:
        g_c = var('G_0')
        g = lambda t, x : g_c
    else:
        g_f = function('G')
        g = lambda t, x : g_f(t, *x)

    return t, dt, x, dx, e, psi, g, v

def mea(t, dt, x, dx, psi, g, v, flux_f, ndims):
    flx = [0 for d in range(ndims)]
    err_v = [0 for d in range(ndims)]
    rhs = 0
    # constructing the Taylor expanded numerical flux functions
    for d in range(ndims):
        flx[d] = flux_f(d, t, x)

        err_v[d] = -dx[d] / dt * (flx[d] + dx[d] ^ 2 / 24 * diff(flx[d], x[d], x[d]))
        err_v[d] = err_v[d] - dt / 2 * diff(err_v[d], t) + dt ^ 2 / 12 * diff(err_v[d], t, t)
        err_v[d] = truncate_to_order(err_v[d], ORD, dt, dx)

        rhs += diff(err_v[d], x[d])
       
        # make it actual error rather than error + flux
        err_v[d] += psi(t, x) * v[d](t, x) 

    dt_psi = (diff(psi(t, x), t) == (rhs / g(t, x) - diff(g(t, x), t) / g(t, x) * psi(t, x)))

    # constructing time derivatives that will be substituted
    dts = [dt_psi, diff(dt_psi, t)]
    dts.extend([diff(dt_psi, xc) for xc in x])

    for d in range(ndims):
        err = err_v[d]
        # changing time derivatives into spatial derivatives
        while True:
            exit = True
            for ds in dts:
                if err.has(ds.lhs()):
                    err = subs_and_truncate(err, ds, ORD, dt, dx)
                    exit = False
            if exit:
                err_v[d] = err.simplify()
                break
    return err_v
