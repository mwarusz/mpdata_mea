from mea import mea, define_vars, truncate_to_order
from schemes import midpoint_velocity, mpdata_flux, antidiffusive_v
import pytest

def div(v, p, ndims):
    def ret(t, x):
        r = 0
        for d in range(ndims):
            r += diff(v[d](t, x) * p(t, x), x[d])
        return r
    return ret

# testing only selected combinations of parameters to keep the tests running in reasonable time
# notably, the 3D version is not tested by default
@pytest.mark.parametrize("ndims, time_ext, space_int, sign_v, sign_av, niters", [
    (1, False, False, 1, 1, 2),
    (1, True, True, -1, -1, 2),
    (1, True, False, 1, -1, 3),
    (2, False, True, -1, -1, 2)
])
def test_mpdata(ndims, time_ext, space_int, sign_v, sign_av, niters):
    sign_v = [sign_v for d in range(ndims)]
    sign_av = [sign_av for d in range(ndims)]

    t, dt, x, dx, e, psi, g, v = define_vars(ndims)
    
    # definition of midpoint (staggered) velocity both in time and in space, the
    # results depends on whether the spatial interpolation (space_int)
    # or temporal extrapolation (time_ext) is used
    vmid = midpoint_velocity(v, time_ext, space_int, dt, e, ndims)

    flux_f = lambda d, t, x : mpdata_flux(d, psi, g, vmid, sign_v, sign_av, niters, dt, dx, e, ndims)(t, x)

    err_v = mea(t, dt, x, dx, psi, g, v, flux_f, ndims)

    if space_int:
        alpha = 4
    else:
        alpha = 1

    if niters > 2:
        beta_m = 0
    else:
        beta_m = 1

    if time_ext:
        gamma = 10
    else:
        gamma = 1

    dt_v = [lambda t, x, vd=vd : diff(vd(t, x), t) for vd in v]
    dtt_v = [lambda t, x, vd=vd: diff(vd(t, x), t, t) for vd in v]
    div_v_psi_over_g  = lambda t, x : div(v, psi, ndims)(t, x) / g(x)
    # instead of writing the analytical expression for the antidiffusive velocity
    # reuse the numerical expression expanded and truncated to give the same result
    av = [ lambda t, x, avd=avd : truncate_to_order(avd(t, x), 2, dt, dx)
           for avd in antidiffusive_v(psi, g, v, sign_v, dt, dx, e, ndims) ]
    
    for d in range(ndims):
        exact = (
                - dx[d] ^ 2 / 24 * (
                                     4 * v[d](t, x) * diff(psi(t, x), x[d], x[d])
                                   + 2 * diff(v[d](t, x), x[d]) * diff(psi(t, x), x[d])
                                   + alpha * diff(v[d](t, x), x[d], x[d]) * psi(t, x)
                                   )
                + dt * dx[d] / 2 * sign_v[d] * v[d](t, x) * diff(div_v_psi_over_g(t, x), x[d])
                + beta_m * dx[d] / 2 * sign_av[d] * av[d](t, x) * diff(psi(t, x), x[d])
                + dt ^ 2 / 24 * (
                                + gamma * dtt_v[d](t, x) * psi(t, x)
                                - 2 * diff(v[d](t, x), t) / g(x) * div(v, psi, ndims)(t, x)
                                + 2 * v[d](t, x) / g(x) * div(dt_v, psi, ndims)(t, x)
                                )
                - dt ^ 2 / 3 * v[d](t, x) / g(x) * div(v, div_v_psi_over_g, ndims)(t, x)
                )
        difference = (err_v[d] - exact).expand().simplify()
        assert(difference == 0)
test_mpdata(1, True, False, 1, 1, 2)
