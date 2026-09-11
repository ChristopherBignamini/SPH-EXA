/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */


/*! @file
 * @brief Min-reduction to determine the global time step in comoving coordinates
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include "sph/ts_global.hpp"

namespace sph
{

/*! @brief Compute the global time step in comoving coordinates
 *
 * @param aNow Scale factor at the time the accelerations in d.ax/ay/az were evaluated
 *
 * Mirrors sph::computeTimestep, from which it differs only in correcting the acceleration criterion by a factor
 * aNow. The reduction, the MPI exchange and the rotation of d.ttot / d.minDt / d.minDt_m1 are identical.
 *
 * sph::accelerationTimestep returns etaAcc * (h^2 / |A|^2)^(1/4) and both of its inputs are comoving: h is a
 * comoving length, and d.ax/ay/az hold dP/dt = a * f_phys rather than an acceleration
 * (see sph::combineAccelerationsComoving). The ratio therefore carries
 *
 *     h_com^2 / |dP/dt|^2 = (h_phys^2 * a^-2) / (|f_phys|^2 * a^2) = a^-4 * h_phys^2 / |f_phys|^2,
 *
 * and the fourth root turns that into a single factor a^-1 which needs to be corrected in order to compare the
 * acceleration criterion to the other time-step criteria.
 *
 * NOTE: a dedicated function is used in preference to passing the corrected value into sph::computeTimestep through
 * its variadic extraTimesteps, because that would leave the uncorrected minDtAcc in the same min: it is larger
 * than the corrected one for a < 1 and would be ignored, but smaller for a > 1, where it would silently take
 * over and shorten the step for no reason.
 */
// TODO: lot of duplication with sph::computeTimestep, could be refactored to avoid that
template<class Dataset, class... Ts>
void computeTimestepComoving(size_t first, size_t last, Dataset& d, double aNow, Ts... extraTimesteps)
{
    using T = typename Dataset::RealType;

    T minDtAcc = (d.g != 0.0) ? T(aNow) * accelerationTimestep(first, last, d) : INFINITY;

    T minDtLoc = std::min({minDtAcc, d.minDtCourant, d.minDtRho, d.maxDtIncrease * d.minDt, extraTimesteps...});

    util::array<T, 3> varsIn{minDtLoc, 0, -T(d.size() - last + first)}, varsOut;
    if constexpr (d.useGpu) { varsIn[1] = -int(d.stackUsedGravity); }
    MPI_Allreduce(varsIn.data(), varsOut.data(), varsIn.size(), MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);
    T minDtGlobal = varsOut[0];
    if constexpr (d.useGpu) { d.stackUsedGravity = int(-varsOut[1]); }
    d.maxHalos = int(-varsOut[2]);

    d.ttot += minDtGlobal;

    d.minDt_m1 = d.minDt;
    d.minDt    = minDtGlobal;
}

} // namespace sph
