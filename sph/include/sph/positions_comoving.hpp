/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief 2nd order time-step integrator combining a separate hydro and gravity acceleration
 *
 * This is a dedicated (duplicated) variant of a part of positions.hpp for the comoving-coordinates propagator.
 * The only difference to the standard integrator is that the acceleration driving the position update
 * is formed from two separate sets: the hydrodynamic acceleration in d.ax/ay/az and the gravitational
 * acceleration in the caller-provided agx/agy/agz arrays. The combination point below is where the
 * comoving scale-factor weighting will eventually be applied.
 *
 * @author Aurelien Cavelan
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include "sph/positions.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

/*! @brief comoving position and canonical momentum update in comoving coordinates, integrating the canonical momentum
 *
 * @param dt          time delta from step n to n+1
 * @param dt_m1       time delta from step n-1 to n
 * @param Xn          comoving coordinates at step n
 * @param An          scale-factor weighted acceleration a^2*A_x driving the momentum at step n
 * @param dXn         X_n - X_n-1, the comoving position increment
 * @param aPrevHalf   scale factor at t_n - dt_m1/2, the midpoint of the previous step
 * @param aHalf       scale factor at t_n + dt/2, the midpoint of the current step
 * @param aNext       scale factor at t_n+1, used to convert the momentum back to a peculiar velocity
 * @param box         global coordinate bounding box
 * @return            tuple(X_n+1, V_n+1, dX_n+1) with X the comoving position and, V the peculiar velocity and 
 *                    dX the comoving position increment
 *
 * The integration variable is the canonical momentum per unit mass, P = p/m = a*v, with p = m*a*v the
 * canonical momentum and v the peculiar velocity.
 * 
 * To use the same variables as the standard position update, we should use the equation involving the
 * peculiar velocity v = a*dX/dt. However, this would introduce a drag term 2*(da/dt)/a * dX/dt = 2*H*v 
 * on the left hand side of the equation of motion,
 * 
 *     d^2X/dt^2 + 2*(da/dt)/a * dX/dt = A_x
 *
 * which is not present in the standard position update. Instead, we multiply the entire equation
 * by a^2 and this turns the left hand side into the total derivative d(a^2 * dX/dt)/dt, so that dP/dt = a^2*A_x
 * carries no drag term: the Hubble drag (2*(da/dt)/a * dX/dt = 2*H*v) is absorbed exactly into P rather 
 * than applied as an explicit O(H*dt) factor. The scale-factor weighting of A itself (which differs 
 * between the gravitational and hydrodynamic contribution) is the caller's responsibility, see 
 * updatePositionsComovingHost.
 *
 * On mass: the canonical momentum is properly p = m*a*v and is therefore driven by a force, m*a^2*A_x,
 * not by an acceleration. The particle mass cancels identically here, though: it multiplies both the
 * momentum recovered from dXn and the kick term, and divides out again in the drift, once per particle
 * and independently of the mass distribution. We therefore integrate the specific momentum P = p/m and
 * drive it with the accelerations that the code already carries (d.ax/ay/az and the gravity arrays), 
 * which avoids a redundant multiplication and division by m in every step.
 *
 * Concerning the integration of comiving position, the update is identical to the standard scheme, 
 * except that the momentum is converted to a comoving displacement through 1/a^2 because of the equation
 * 
 * dX/dt = P/a^2
 * 
 * The peculiar velocity is then recovered from the momentum at the end of the step through 1/aNext. 
 *
 * Concerning time-reversibility:
 * positionUpdateComoving(-dt, dt_m1, X_n+1, An, dXn, aPrevHalf, aHalf, aNext, box) back-propagates
 * X_n+1 to X_n. aHalf is the midpoint of the step and therefore takes the same value in either
 * direction of traversal, so the sign structure of dX_n+1 flips exactly as in the standard scheme.
 * The above condition, however, only holds if forward half time steps have the same values of the corresponding
 * backward half time steps. 
 */
template<class T>
HOST_DEVICE_FUN auto positionUpdateComoving(double dt, double dt_m1, cstone::Vec3<T> Xn, cstone::Vec3<T> An,
                                            cstone::Vec3<T> dXn, T aPrevHalf, T aHalf, T aNext,
                                            const cstone::Box<T>& box)
{
    auto Pnmhalf = dXn * (aPrevHalf * aPrevHalf / T(dt_m1));
    auto Pn      = Pnmhalf + T(0.5) * dt_m1 * An;
    auto Pnp1    = Pn + An * dt;
    auto dXnp1   = (Pn + T(0.5) * An * std::abs(dt)) * (dt / (aHalf * aHalf));
    auto Xnp1    = cstone::putInBox(Xn + dXnp1, box);
    //! @brief the dataset stores the peculiar velocity, not the canonical momentum
    auto Vnp1 = Pnp1 * (T(1) / aNext);

    return util::tuple<cstone::Vec3<T>, cstone::Vec3<T>, cstone::Vec3<T>>{Xnp1, Vnp1, dXnp1};
}

 /*! @brief CPU position update combining hydro (d.ax/ay/az) and gravity (agx/agy/agz) accelerations evaluated on 
 *          comoving coordinates
 *
 * @param agx         per-particle gravitational acceleration in the x-direction, or nullptr when gravity is disabled
 * @param agy         per-particle gravitational acceleration in the y-direction, or nullptr when gravity is disabled
 * @param agz         per-particle gravitational acceleration in the z-direction, or nullptr when gravity is disabled
 * @param aNow        scale factor at t_n, the time at which the accelerations were evaluated
 * @param aPrevHalf   scale factor at t_n - dt_m1/2, the midpoint of the previous step
 * @param aHalf       scale factor at t_n + dt/2, the midpoint of the current step
 * @param aNext       scale factor at t_n+1, used to convert the momentum back to a peculiar velocity
 *
 * The two acceleration sets enter the canonical momentum with different powers of the scale factor, which
 * is why they are kept separate up to this point. With the comoving equation of motion written as
 * d2X/dt2 + 2H*dX/dt = f_phys/a for a physical specific force f_phys, the momentum is driven by
 * dP/dt = a^2 * A_x = a * f_phys, and the two contributions weigh in as follows.
 *
 * Gravity: the tree solver sums G*m_j*(x_j - x_i)/|x_j - x_i|^3 over the *comoving* separations in d.x, so
 * agx/agy/agz hold g_c = a^2 * g_phys. Hence dP/dt|grav = a * g_phys = g_c / a, i.e. wGrav = 1/a.
 *
 * Hydro: the kernel sums produce the comoving density rho_c = a^3 * rho_phys, and d.u holds the comoving
 * specific internal energy u_c = a^(3*(gamma-1)) * u_phys, so the equation of state yields
 * p_stored = (gamma-1) * rho_c * u_c = a^(3*gamma) * p_phys. Tracking that through the volume elements
 * and the kernel gradient, d.ax comes out as a^(3*gamma-2) * f_phys, so reaching a * f_phys takes
 * wHydro = a^(3-3*gamma) = a^(-3*(gamma-1)).
 *
 * The comoving internal energy is used in preference to the physical one because it removes the adiabatic
 * cooling of the expansion from the energy equation entirely: du_c/dt = -(gamma-1) * u_c * div_x(dx/dt)
 * carries no -3*H*(gamma-1)*u source term, the same way the canonical momentum carries no Hubble drag.
 * NOTE: due to above assumptions, d.c is now a^(3*(gamma-1)/2) times the physical sound speed.
 */
template<class T, class Tg, class Dataset>
void updatePositionsComovingHost(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box,
                                 const Tg* agx, const Tg* agy, const Tg* agz, T aNow, T aPrevHalf, T aHalf, T aNext)
{
    bool anyFBC = box.boundaryX() == cstone::BoundaryType::fixed || box.boundaryY() == cstone::BoundaryType::fixed ||
                  box.boundaryZ() == cstone::BoundaryType::fixed;

    cstone::Vec3<T> adjustForFBC{T(1.), T(1.), T(1.)};

    //! @brief scale-factor weights of the two acceleration sets, see the note above
    T wHydro = std::pow(aNow, T(-3) * (d.gamma - T(1)));
    T wGrav  = T(1) / aNow;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        cstone::Vec3<T> X{d.x[i], d.y[i], d.z[i]};

        if (anyFBC) { adjustForFBC = fbcAdjustFactors(X, box, d.h[i]); }

        // combination point: dP/dt = wHydro * hydro + wGrav * gravity, see the note above on the weights
        T ax_tot = wHydro * d.ax[i] + wGrav * (agx != nullptr ? T(agx[i]) : T(0));
        T ay_tot = wHydro * d.ay[i] + wGrav * (agy != nullptr ? T(agy[i]) : T(0));
        T az_tot = wHydro * d.az[i] + wGrav * (agz != nullptr ? T(agz[i]) : T(0));

        // To keep particles belonging to the fixed boundaries from moving, these two quantities need to be adjusted
        cstone::Vec3<T> A{ax_tot * adjustForFBC[0], ay_tot * adjustForFBC[1], az_tot * adjustForFBC[2]};
        cstone::Vec3<T> X_m1{d.x_m1[i] * adjustForFBC[0], d.y_m1[i] * adjustForFBC[1], d.z_m1[i] * adjustForFBC[2]};
        cstone::Vec3<T> V;
        util::tie(X, V, X_m1) =
            positionUpdateComoving(d.minDt, d.minDt_m1, X, A, X_m1, aPrevHalf, aHalf, aNext, box);

        util::tie(d.x[i], d.y[i], d.z[i])          = util::tie(X[0], X[1], X[2]);
        util::tie(d.x_m1[i], d.y_m1[i], d.z_m1[i]) = util::tie(X_m1[0], X_m1[1], X_m1[2]);
        util::tie(d.vx[i], d.vy[i], d.vz[i])       = util::tie(V[0], V[1], V[2]);
    }
}

/*! @brief advance positions using a separate hydro and gravity acceleration set
 *
 * Mirrors sph::computePositions, but combines d.ax/ay/az with the gravity arrays agx/agy/agz.
 * TODO: check energy calculation in the comoving case. 
 */
template<class T, class Tg, class Dataset>
void computePositionsComoving(const GroupView& grp, Dataset& d, const cstone::Box<T>& box, float dt_forward,
                              util::array<float, Timestep::maxNumRungs> dt_m1, const Tg* agx, const Tg* agy,
                              const Tg* agz, T aNow, T aPrevHalf, T aHalf, T aNext,
                              const uint8_t* rung = nullptr)
{
    if constexpr (d.useGpu)
    {
        T     constCv = d.mui.empty() ? idealGasCv(d.muiConst, d.gamma) : -1.0;
        auto* d_mui   = d.mui.empty() ? nullptr : rawPtr(d.mui);

        computePositionsComovingGpu(grp, dt_forward, dt_m1, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.vx),
                                    rawPtr(d.vy), rawPtr(d.vz), rawPtr(d.x_m1), rawPtr(d.y_m1), rawPtr(d.z_m1),
                                    rawPtr(d.ax), rawPtr(d.ay), rawPtr(d.az), agx, agy, agz, rung, rawPtr(d.temp),
                                    rawPtr(d.u), rawPtr(d.du), rawPtr(d.du_m1), rawPtr(d.h), d_mui, d.gamma, constCv,
                                    aNow, aPrevHalf, aHalf, aNext, box);
    }
    else
    {
        updatePositionsComovingHost(grp.firstBody, grp.lastBody, d, box, agx, agy, agz, aNow, aPrevHalf, aHalf,
                                    aNext);
        
        // TODO (energy): check energy calculation in the comoving case.
        if (!d.temp.empty()) { updateTempHost(grp.firstBody, grp.lastBody, d, box); }
        else if (!d.u.empty()) { updateIntEnergyHost(grp.firstBody, grp.lastBody, d, box); }
    }
}

} // namespace sph
