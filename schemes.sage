from mea import ret_helper

def upwind_flux(d, p, v, sign_v, dt, dx, e):
    def ret(t, x):
        if sign_v[d] > 0:
            f = dt / dx[d] * v[d](t, x) * p(t, x - 1 / 2 * e[d])
        else:
            f = dt / dx[d] * v[d](t, x) * p(t, x + 1 / 2 * e[d])
        return ret_helper(f, dt, dx)
    return ret

def upwind(p, g, v, sign_v, dt, dx, e, ndims):
    def ret(t, x):
        new_p = p(t, x)
        for d in range(ndims):
            f1 = upwind_flux(d, p, v, sign_v, dt, dx, e)(t, x + 1 / 2 * e[d])
            f2 = upwind_flux(d, p, v, sign_v, dt, dx, e)(t, x - 1 / 2 * e[d])
            new_p = new_p - (f1 - f2) / g(x)
        return ret_helper(new_p, dt, dx)
    return ret

def mid_time(time_ext, u, dt):
    def ret(t, x):
        if time_ext:
            return 1 / 2 * (3 * u(t, x) - u(t - dt, x))
        else:
            return u(t + dt / 2, x)
    return ret

def mid_space(d, space_int, v, e):
    def ret(t, x):
        if space_int:
            return 1 / 2 * (v[d](t, x - 1 / 2 * e[d]) + v[d](t, x + 1 / 2 * e[d]))
        else:
            return v[d](t, x)
    return ret

def midpoint_velocity(v, time_ext, space_int, dt, e, ndims):
    vmid_s = [mid_space(d, space_int, v, e) for d in range(ndims)]
    vmid_ts = [mid_time(time_ext, vc, dt) for vc in vmid_s]
    return vmid_ts

def mpdata_flux(d, p, g, v, sign_v, sign_av, niters, dt, dx, e, ndims):
    def ret(t, x):
        np = p
        nv, sign_nv = v, sign_v
        f = 0
        for it in range(niters):
            if it > 0:
                np = upwind(np, g, nv, sign_nv, dt, dx, e, ndims)
                nv, sign_nv = antidiffusive_v(np, g, nv, sign_nv, dt, dx, e, ndims), sign_av
            f += upwind_flux(d, np, nv, sign_nv, dt, dx, e)(t, x)
        return ret_helper(f, dt, dx)
    return ret

def antidiffusive_v(p, g, v, sign_v, dt, dx, e, ndims):
    return [antidiffusive_v_d(d, p, g, v, sign_v, dt, dx, e, ndims) for d in range(ndims)]

def antidiffusive_v_d(d, p, g, v, sign_v, dt, dx, e, ndims):
    def ret(t, x):
        gmid = (g(x + 1 / 2 * e[d]) + g(x - 1 / 2 * e[d])) / 2
        a1 = (
              (sign_v[d] * v[d](t, x) - dt / dx[d] * v[d](t, x) ** 2 / gmid) *  
              (p(t, x + 1 / 2 * e[d]) - p(t, x - 1 / 2 * e[d])) / 
              (p(t, x + 1 / 2 * e[d]) + p(t, x - 1 / 2 * e[d]))
             )
        a2 = 0
        a3 = (
              - 1 / 4 * v[d](t, x) / gmid * 
                 (
                  v[d](t, x + e[d]) - v[d](t, x - e[d])
                 ) / dx[d]
             )
        
        for dd in range(ndims):
            if dd == d:
                continue
            vmid = ( 1 / 4 * ( v[dd](t, x + 1 / 2 * e[d] + 1 / 2 * e[dd]) +
                               v[dd](t, x - 1 / 2 * e[d] + 1 / 2 * e[dd]) +
                               v[dd](t, x + 1 / 2 * e[d] - 1 / 2 * e[dd]) +
                               v[dd](t, x - 1 / 2 * e[d] - 1 / 2 * e[dd]) )
                   )

            a2 += (
                  - 1 / 2 * v[d](t, x) * vmid / gmid *
                  (p(t, x + 1 / 2 * e[d] + 1 * e[dd]) + p(t, x - 1 / 2 * e[d] + 1 * e[dd]) -
                   p(t, x + 1 / 2 * e[d] - 1 * e[dd]) - p(t, x - 1 / 2 * e[d] - 1 * e[dd]) 
                  ) / 
                  (p(t, x + 1 / 2 * e[d] + 1 * e[dd]) + p(t, x - 1 / 2 * e[d] + 1 * e[dd]) +
                   p(t, x + 1 / 2 * e[d] - 1 * e[dd]) + p(t, x - 1 / 2 * e[d] - 1 * e[dd]) 
                  )
                  ) / dx[dd]

            a3 += (
                  - 1 / 4 * v[d](t, x) / gmid * 
                    (
                     v[dd](t, x + 1 / 2 * e[d] + 1 / 2 * e[dd]) - v[dd](t, x + 1 / 2 * e[d] - 1 / 2 * e[dd]) +
                     v[dd](t, x - 1 / 2 * e[d] + 1 / 2 * e[dd]) - v[dd](t, x - 1 / 2 * e[d] - 1 / 2 * e[dd])
                    ) / dx[dd]
                  )
            
        a = a1 + dt * (a2 + a3)
        return ret_helper(a, dt, dx)
    return ret
