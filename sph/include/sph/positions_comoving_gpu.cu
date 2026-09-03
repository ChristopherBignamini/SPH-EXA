/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
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
 * @brief 2nd order time-step integrator on the GPU, combining a separate hydro and gravity acceleration
 *
 * Dedicated (duplicated) variant of positions_gpu.cu for the comoving-coordinates propagator.
 *
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/util/array.hpp"

#include "eos.hpp"
#include "positions.hpp"
#include "positions_comoving.hpp"
#include "sph_gpu.hpp"
#include "timestep.h"

namespace sph
{

using cstone::GpuConfig;
using cstone::LocalIndex;

/*! @brief advance positions using a separate hydro and gravity acceleration set
 *
 * TODO (documentation): update the documentation to reflect the comoving integration scheme.
 * Check position_comoving.hpp for a discussion of the comoving integration scheme and assumptions about
 * the input/output variables definition. 
 */
template<class Tc, class Tv, class Ta, class Tg, class Tdu, class Tm1, class Tt, class Thydro>
__global__ void computePositionsComovingKernel(GroupView grp, float dt,
                                               util::array<float, Timestep::maxNumRungs> dt_m1, Tc* x, Tc* y, Tc* z,
                                               Tv* vx, Tv* vy, Tv* vz, Tm1* x_m1, Tm1* y_m1, Tm1* z_m1, Ta* ax, Ta* ay,
                                               Ta* az, const Tg* agx, const Tg* agy, const Tg* agz, const uint8_t* rung,
                                               Tt* temp, Tt* u, Tdu* du, Tm1* du_m1, Thydro* h, Thydro* mui, Tc gamma,
                                               Tc constCv, Tc aNow, Tc aPrevHalf, Tc aHalf, Tc aNext,
                                               const cstone::Box<Tc> box, bool anyFBC)
{
    LocalIndex laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    float dt_m1_rung = (rung != nullptr) ? dt_m1[rung[i]] : dt_m1[0];

    cstone::Vec3<Tc> X{x[i], y[i], z[i]};
    cstone::Vec3<Tc> adjustForFBC{Tc(1.), Tc(1.), Tc(1.)};
    if (anyFBC) { adjustForFBC = fbcAdjustFactors(X, box, h[i]); }

    // combination point: total acceleration = hydro + gravity with comoving scale-factor weights
    // scale-factor weights of the two acceleration sets, see sph::updatePositionsComovingHost
    Tc wHydro = Tc(1);
    Tc wGrav  = Tc(1) / aNow;

    Ta ax_tot = wHydro * ax[i] + wGrav * (agx != nullptr ? Ta(agx[i]) : Ta(0));
    Ta ay_tot = wHydro * ay[i] + wGrav * (agy != nullptr ? Ta(agy[i]) : Ta(0));
    Ta az_tot = wHydro * az[i] + wGrav * (agz != nullptr ? Ta(agz[i]) : Ta(0));

    // To keep particles belonging to the fixed boundaries from moving, these two quantities need to be adjusted
    cstone::Vec3<Tc> A{ax_tot * adjustForFBC[0], ay_tot * adjustForFBC[1], az_tot * adjustForFBC[2]};
    cstone::Vec3<Tc> X_m1{x_m1[i] * adjustForFBC[0], y_m1[i] * adjustForFBC[1], z_m1[i] * adjustForFBC[2]};
    cstone::Vec3<Tc> V;
    util::tie(X, V, X_m1) = positionUpdateComoving(dt, dt_m1_rung, X, A, X_m1, aPrevHalf, aHalf, aNext, box);

    util::tie(x[i], y[i], z[i])          = util::tie(X[0], X[1], X[2]);
    util::tie(x_m1[i], y_m1[i], z_m1[i]) = util::tie(X_m1[0], X_m1[1], X_m1[2]);
    util::tie(vx[i], vy[i], vz[i])       = util::tie(V[0], V[1], V[2]);

    // TODO (energy): check energy calculation in the comoving case.
    Tc minDistanceFactor = min(adjustForFBC);
    if (temp != nullptr)
    {
        Thydro cv    = (constCv < 0) ? idealGasCv(mui[i], gamma) : constCv;
        auto   u_old = temp[i] * cv;
        temp[i]  = energyUpdate(u_old, dt * minDistanceFactor, dt_m1_rung * minDistanceFactor, du[i], du_m1[i]) / cv;
        du_m1[i] = du[i];
    }
    else if (u != nullptr)
    {
        u[i]     = energyUpdate(u[i], dt * minDistanceFactor, dt_m1_rung * minDistanceFactor, du[i], du_m1[i]);
        du_m1[i] = du[i];
    }
}

template<class Tc, class Tv, class Ta, class Tg, class Tdu, class Tm1, class Tt, class Thydro>
void computePositionsComovingGpu(const GroupView& grp, float dt, util::array<float, Timestep::maxNumRungs> dt_m1, Tc* x,
                                 Tc* y, Tc* z, Tv* vx, Tv* vy, Tv* vz, Tm1* x_m1, Tm1* y_m1, Tm1* z_m1, Ta* ax, Ta* ay,
                                 Ta* az, const Tg* agx, const Tg* agy, const Tg* agz, const uint8_t* rung, Tt* temp,
                                 Tt* u, Tdu* du, Tm1* du_m1, Thydro* h, Thydro* mui, Tc gamma, Tc constCv,
                                 Tc aNow, Tc aPrevHalf, Tc aHalf, Tc aNext,
                                 const cstone::Box<Tc>& box)
{
    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;

    bool anyFBC = box.boundaryX() == cstone::BoundaryType::fixed || box.boundaryY() == cstone::BoundaryType::fixed ||
                  box.boundaryZ() == cstone::BoundaryType::fixed;

    if (numBlocks == 0) { return; }
    computePositionsComovingKernel<<<numBlocks, numThreads>>>(grp, dt, dt_m1, x, y, z, vx, vy, vz, x_m1, y_m1, z_m1, ax,
                                                              ay, az, agx, agy, agz, rung, temp, u, du, du_m1, h, mui,
                                                              gamma, constCv, aNow, aPrevHalf, aHalf, aNext,
                                                              box,
                                                              anyFBC);
}

#define POS_COMOVING_GPU(Tc, Tv, Ta, Tg, Tdu, Tm1, Tt, Thydro)                                                         \
    template void computePositionsComovingGpu(                                                                         \
        const GroupView& grp, float dt, util::array<float, Timestep::maxNumRungs> dt_m1, Tc* x, Tc* y, Tc* z, Tv* vx,  \
        Tv* vy, Tv* vz, Tm1* x_m1, Tm1* y_m1, Tm1* z_m1, Ta* ax, Ta* ay, Ta* az, const Tg* agx, const Tg* agy,         \
        const Tg* agz, const uint8_t* rung, Tt* temp, Tt* u, Tdu* du, Tm1* du_m1, Thydro* h, Thydro* mui, Tc gamma,    \
        Tc constCv, Tc aNow, Tc aPrevHalf, Tc aHalf, Tc aNext, const cstone::Box<Tc>& box)

//               Tc      Tv     Ta     Tg     Tdu     Tm1     Tt      Thydro
POS_COMOVING_GPU(double, double, double, double, double, double, double, double);
POS_COMOVING_GPU(float, float, float, float, float, float, float, float);
POS_COMOVING_GPU(double, double, double, double, float, float, double, double);
POS_COMOVING_GPU(double, float, float, float, double, float, double, float);

} // namespace sph
