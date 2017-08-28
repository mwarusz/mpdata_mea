from mea import mea, define_vars
from schemes import upwind_flux
import pytest

# this test checks that doing the MEA of the first-order accurate upwind scheme,
# assuming constant velocity and G field, reproduces the known results that lead
# to standard MPDATA with "third-order" corrections
@pytest.mark.parametrize("sign_v", [1, -1])
def test_upwind(sign_v):
    ndims = 3
    sign_v = [sign_v for d in range(ndims)]

    t, dt, x, dx, e, psi, g, v = define_vars(ndims, const_v = True, const_g = True)
    flux_f = lambda d, t, x : upwind_flux(d, psi, v, sign_v, dt, dx, e)(t, x)
    err_v = mea(t, dt, x, dx, psi, g, v, flux_f, ndims)
    
    # redefintions to compress the notation and make it more consistent with the one in the papers
    dx, dy, dz = dx[0], dx[1], dx[2]
    U, V, W = dt / dx * v[0](t, x), dt / dy * v[1](t, x), dt / dz * v[2](t, x)
    abs_U, abs_V, abs_W = U * sign_v[0], V * sign_v[1], W * sign_v[2]
    G = g(x)
    psi = psi(t, x)
    x, y, z = x[0], x[1], x[2]
    
    # expressions for the standard (normalised) antidiffusive velocities of MPDATA
    # see equations (10) and (11) in "A Fully Multidimensional Positive Definite Advection Transport Algorithm
    # with Small Implicit Diffusion", Smolarkiewicz, JCP, 1984
    adv_2nd_x = (
                  dx * abs_U / 2 * (1 - abs_U / G) * diff(psi, x) / psi
                - dy * U * V / (2 * G) * diff(psi, y) / psi
                - dz * U * W / (2 * G) * diff(psi, z) / psi
                )
    
    adv_2nd_y = (
                  dy * abs_V / 2 * (1 - abs_V / G) * diff(psi, y) / psi
                - dz * V * W / (2 * G) * diff(psi, z) / psi
                - dx * V * U / (2 * G) * diff(psi, x) / psi
                )
    
    adv_2nd_z = (
                  dz * abs_W / 2 * (1 - abs_W / G) * diff(psi, z) / psi
                - dx * W * U / (2 * G) * diff(psi, x) / psi
                - dy * W * V / (2 * G) * diff(psi, y) / psi
                )

    # expressions for the "third-order" corrections to the antidiffusive velocities
    # see equation (36) in "MPDATA: A Finite-Difference Solver for Geophysical Flows",
    # Smolarkiewicz and Margolin, JCP, 1998

    adv_3rd_x = (
                  dx ^ 2 / 6 * (3 * U * abs_U / G - 2 * U ^ 3 / G ^ 2 - U) * diff(psi, x, x) / psi
                + dx * dy * V / (2 * G) * (abs_U - 2 * U ^ 2 / G) * diff(psi, x, y) / psi
                + dx * dz * W / (2 * G) * (abs_U - 2 * U ^ 2 / G) * diff(psi, x, z) / psi
                - 2 * dy * dz * U * V * W / (3 * G ^ 2) * diff(psi, y, z) / psi
                )
    
    adv_3rd_y = (
                  dy ^ 2 / 6 * (3 * V * abs_V / G - 2 * V ^ 3 / G ^ 2 - V) * diff(psi, y, y) / psi
                + dy * dz * W / (2 * G) * (abs_V - 2 * V ^ 2 / G) * diff(psi, y, z) / psi
                + dy * dx * U / (2 * G) * (abs_V - 2 * V ^ 2 / G) * diff(psi, y, x) / psi
                - 2 * dz * dx * U * V * W / (3 * G ^ 2) * diff(psi, z, x) / psi
                )
    
    adv_3rd_z = (
                  dz ^ 2 / 6 * (3 * W * abs_W / G - 2 * W ^ 3 / G ^ 2 - W) * diff(psi, z, z) / psi
                + dz * dx * U / (2 * G) * (abs_W - 2 * W ^ 2 / G) * diff(psi, z, x) / psi
                + dz * dy * V / (2 * G) * (abs_W - 2 * W ^ 2 / G) * diff(psi, z, y) / psi
                - 2 * dx * dy * U * V * W / (3 * G ^ 2) * diff(psi, x, y) / psi
                )
   
    # we can only compare total error (divergence of error vector) because automatic
    # mea leads to different decomposition of error that the standard MPDATA formulae
    mea_error = diff(err_v[0], x) + diff(err_v[1], y) + diff(err_v[2], z)

    # transforming normalised antidiffusive velocities into error vector components
    adv_err_x = dx / dt * (adv_2nd_x + adv_3rd_x) * psi
    adv_err_y = dy / dt * (adv_2nd_y + adv_3rd_y) * psi
    adv_err_z = dz / dt * (adv_2nd_z + adv_3rd_z) * psi

    # total error based on antidiffusive velocities
    adv_error = diff(adv_err_x, x) + diff(adv_err_y, y) + diff(adv_err_z, z)

    difference = (mea_error - adv_error).expand().simplify()
    assert(difference == 0)
