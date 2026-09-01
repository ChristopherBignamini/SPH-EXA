/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Adaptor to ChaNGa native COOLING_COSMO back-end for radiative cooling
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "changa_cosmo_cooler.hpp"
#include "code_units.hpp"

// ChaNGa's cooling.h already has its own extern "C" guard
#include "cooling.h"

namespace cooling
{

template<typename T>
struct ChangaCosmoCooler<T>::Impl
{
    friend struct ChangaCosmoCooler<T>;

public:

    // Needed to call CoolDerivsFinalize and CoolFinalize during the destruction of ChangaCosmoCooler::impl_ptr
    ~Impl()
    {
        for (auto* d : derivs_)
        {
            if (d != nullptr) { CoolDerivsFinalize(d); }
        }
        derivs_.clear();
        if (cl_ != nullptr) { CoolFinalize(cl_); }
    }

private:
    Impl() { setDefaultParameters(); }

    CodeUnitScales scales_;

    COOLPARAM param_;
    COOL*     cl_ = nullptr;

    //! @brief Cooling solver scratch space for each thread, allocated on demand
    std::vector<clDerivsData*> derivs_;

    /*! @brief cooling_cosmo is a PRIMORDIAL network: Y_H + 4*Y_He == 1 exactly, with no metal
     * component in the mass budget, no metal contribution to the electron budget, and no
     * metal-line cooling at any temperature. Metals belong to COOLING_METAL backends.
     * ZMetal is not entirely ignored though: it scales one empirical low-temperature fit
     * gated by bLowTCool. So both stay at zero and neither is exposed as a parameter.
     */
    inline constexpr static double ZMetal = 0.0;

    /*! @brief the adiabatic index, exactly 5/3 for this back-end
     *
     * Matches ChaNGa's own CoolCodePressureOnDensitySoundSpeed macro in cooling_cosmo.h, 
     * which hardcodes 5./3.
     */
    inline constexpr static double adiabaticIndex = 5. / 3.;

    /*! @brief Set cooling parameter to default values taken from ChaNGa's cooling_cosmo.c 
     * CoolInitParameters() function. These values are not exposed to the user.
     */
    void setDefaultParameters()
    {
        param_.bIonNonEqm = 0; // accepted by ChaNGa but unused: cosmo always solves equilibrium
        /* UV background OFF, and not exposed as a parameter. clRatesRedshift asserts
         * UV != NULL whenever bUV is set, and we never call clInitUV -- so bUV=1 would generate
         * fatal errors. Enabling it means implementing the CoolTableReadInfo/CoolTableRead path 
         * and shipping a UV table.
         */
        param_.bUV              = 0;
        param_.bUVTableUsesTime = 0;
        param_.bDoIonOutput     = 0;
        param_.bLowTCool        = 0; // no metals in this back-end, see comment on ZMetal above
        param_.bSelfShield      = 0; // inert without a UV background, see bUV
        param_.dMassFracHelium  = 0.25;
        param_.dCoolingTmin     = 10.0;
        param_.dCoolingTmax     = 1e9;
        param_.nCoolingTable    = 15001;
    }

    /*! @brief Initialization of the Changa cooling backend
     *
     * In ChaNGa the init steps are performed in different places, due to the interplay
     * with Charm++. Here we wrap them in a single function, which is called once at the beginning of the simulation.
     *
     * @param comoving_coordinates Whether the simulation is in comoving coordinates. Not supported yet.
     * @param time_unit_opt Optional time unit to use for the cooling backend. If not provided, the default time unit is used.
     */
    void init(const bool comoving_coordinates, const std::optional<T> time_unit_opt)
    {
        if (comoving_coordinates)
        {
            throw std::runtime_error("ChangaCosmoCooler: comoving coordinates are not supported yet. The redshift is "
                                     "fixed at init time; see TODO(comoving) in cooling_backend.hpp");
        }

        const CodeUnits units = makeCodeUnits(scales_, time_unit_opt);

        cl_ = CoolInit();
        if (cl_ == nullptr) { throw std::runtime_error("ChangaCosmoCooler: CoolInit failed"); }

        // At z = 0 the comoving and proper density units coincide.
        clInitConstants(cl_, units.density, units.density, units.specificEnergy, units.time, scales_.l_code_in_kpc, param_);
        CoolInitRatesTable(cl_, param_);

        // TODO (comoving): revisit this when comoving coordinates are supported; the redshift is fixed at init time but
        // this function should be called every time the redshift changes. For the time being, this is not
        // present in the contract. 
        CoolSetTime(cl_, 0.0, 0.0);

        prepareDerivsPool();
    }

    //! @brief Prepare the cooling solver scratch space for each thread
    void prepareDerivsPool()
    {
        const size_t nThreads = size_t(omp_get_max_threads());
        while (derivs_.size() < nThreads)
        {
            derivs_.push_back(CoolDerivsInit(cl_));
        }
    }

    /*! @brief Cool particles data in the range [first, last) using the ChaNGa cooling_cosmo back-end
     * 
     * This function corresponds to the TreePiece::updateuDot() function in Changa, 
     * but is adapted to the SPH-EXA contract.
     *
     * Step-by-step correspondence with updateuDot (Sph.cpp:860), COOLING_COSMO path:
     *
     *   ChaNGa                                      here
     *   -----------------------------------------   ---------------------------------------------
     *   loop over local gas particles active rung   omp for over [first,last), no rung filter
     *   CoolCodeTimeToSeconds(duDelta[rung])        dtSeconds, outside of the loop over particles
     *   CoolCodePressureOnDensitySoundSpeed()       computePressures/computeAdiabaticIndices,
     *                                               called separately, by eos_cooling
     *   PoverRhoFloorJeans() + PdV rescaling        absent. A pressure floor is an EOS/hydro
     *                                               concern; ChaNGa applies it here because 
     *                                               updateuDot fuses EOS, PdV and cooling
     *   ExternalHeating = PdV + AV + diff + SN      set to 0, see comment on the next line
     *   uDot = ExternalHeating when cooling is 
     *   off (SNe)                                   absent, no cooling-shutoff mechanism
     *   CoolIntegrateEnergyCode(.., p->fMetals())   same call, but ZMetal fixed at 0 and 
     *                                               ExternalHeating = 0.0
     *   CkAssert(E > 0), isfinite(uDot)             not checked
     *   uDot = (E - u)/dt         (assignment)      du += (E - u)/dt (accumulation)
     * 
     * @param dt Time step
     * @param rho Pointer to density values
     * @param u Pointer to internal energy values
     * @param chemistry Pointer to chemical abundance values
     * @param du Pointer to the rate of change of internal energy
     * @param first First particle index to cool
     * @param last Last particle index to cool
     */
    template<typename Trho, typename Tu>
    void cool_particles(const T dt, const Trho* rho, const Tu* u, const FieldPtrs& chemistry, Tu* du,
                        const size_t first, const size_t last)
    {
        prepareDerivsPool();

        T* yHI   = util::get<"Y_HI", CoolingFields>(chemistry);
        T* yHeI  = util::get<"Y_HeI", CoolingFields>(chemistry);
        T* yHeII = util::get<"Y_HeII", CoolingFields>(chemistry);

        // CoolIntegrateEnergyCode takes everything in code units EXCEPT the time step
        const double dtSeconds = CoolCodeTimeToSeconds(cl_, dt);

#pragma omp parallel
        {
            clDerivsData* derivs = derivs_[size_t(omp_get_thread_num())];

#pragma omp for schedule(static)
            for (size_t i = first; i < last; ++i)
            {
                COOLPARTICLE cp;
                cp.Y_HI         = yHI[i];
                cp.Y_HeI        = yHeI[i];
                cp.Y_HeII       = yHeII[i];
                cp.dLymanWerner = 0.0; // declared by cooling_cosmo.h but unused by it

                double E      = u[i];
                double pos[3] = {0.0, 0.0, 0.0}; // cosmo ignores the position argument

                /* TODO (operator-splitting): in current implementation, cooling is integrated
                 * alone at fixed u and the hydro rate is added afterwards (see du below).
                 * This matches how SPH-EXA drives GRACKLE, but NOT how ChaNGa drives this
                 * same solver: updateuDot passes
                 *     ExternalHeating = duDotPdV + uDotAV + uDotDiff + fESNrate
                 * into CoolIntegrateEnergyCode, so heating and cooling evolve TOGETHER inside
                 * the integrator and it can then assign uDot = (E - u)/dt outright.
                 *
                 * The two agree only while the hydro rate is small over a step. They diverge
                 * for gas cooling hard while being compressed, exactly where the coupling
                 * matters most: with H fed in, heating keeps the gas hotter and so changes
                 * the radiative cooling rate within the step, and near thermal balance the stiff
                 * solver simply sits at the equilibrium. Split, we instead integrate along a
                 * pure-cooling trajectory and add the hydro rate back linearly, performing the 
                 * radiative cooling at temperatures the gas never actually has.
                 *
                 * Current splitting is allows a fair comparison between GRACKLE and ChaNGa cooling 
                 * in SPH-EXA but it is a limit to how closely we can ever reproduce ChaNGa's 
                 * behavior.
                 *
                 * A possible fix would be a contract change: make du in/out (on entry the hydro 
                 * rate, on exit the total), pass it here as ExternalHeating and assign below 
                 * instead of accumulating -- E would already contain the hydro contribution, 
                 * so `du[i] +=` would double count it. Grackle would do the same. These changes
                 * touches CoolingBackend, eos_cooling, both adapters and the propagator -- and
                 * must be done for both back-ends at once. 
                 */
                CoolIntegrateEnergyCode(cl_, derivs, &cp, &E, 0.0, rho[i], ZMetal, pos, dtSeconds);

                // du update according t
                du[i] += (E - u[i]) / dt;

                yHI[i]   = cp.Y_HI;
                yHeI[i]  = cp.Y_HeI;
                yHeII[i] = cp.Y_HeII;
            }
        }
    }

    /*! @brief compute the pressure for each particle in the range [first, last)
     *
     * This function corresponds to the CoolCodePressureOnDensitySoundSpeed() function in Changa 
     * header cooling_cosmo.h
     *
     * @param rho Pointer to density values
     * @param u Pointer to internal energy values
     * @param chemistry Pointer to chemical abundance values (unused)
     * @param p Pointer to pressure values
     * @param first First particle index to compute pressure for
     * @param last Last particle index to compute pressure for]
    */
    template<typename Trho, typename Tu, typename Tp>
    void computePressures(const Trho* rho, const Tu* u, const FieldPtrs&, Tp* p, const size_t first, const size_t last)
    {
#pragma omp parallel for schedule(static)
        for (size_t i = first; i < last; ++i)
        {
            p[i] = (adiabaticIndex - 1.) * rho[i] * u[i];
        }
    }

    /*! @brief compute the adiabatic index for each particle in the range [first, last)
     *
     * This function corresponds to the CoolCodePressureOnDensitySoundSpeed() function in Changa 
     * header cooling_cosmo.h. For a monatomic ideal gas, the adiabatic index is constant and 
     * equal to 5/3.
     *
     * @param rho Pointer to density values (unused)
     * @param u Pointer to internal energy values (unused)
     * @param chemistry Pointer to chemical abundance values (unused)
     * @param gamma Pointer to adiabatic index values
     * @param first First particle index to compute adiabatic index for
     * @param last Last particle index to compute adiabatic index for
    */
    template<typename Trho, typename Tu, typename Tgamma>
    void computeAdiabaticIndices(const Trho*, const Tu*, const FieldPtrs&, Tgamma* gamma, const size_t first,
                                 const size_t last)
    {
        std::fill(gamma + first, gamma + last, Tgamma(adiabaticIndex));
    }

    /*! @brief smallest |u / edot| over [first, last), the cooling time criterion
     *
     * There is a mismatch between ChaNGa and SPH-EXA+Grackle here: Changa doesn't have
     * an explicit radiative cooling timestep which could be compared to the other timesteps 
     * (Courant, acceleration, etc...) so there is no equivalent to this function in ChaNGa. 
     * In principle we could skip this step and let the propagator pick the timestep based 
     * on the other criteria, but that would be unsafe and could lead to physically wrong 
     * results. For example: given that pressure, sound speed and the SPH forces are computed 
     * once per step from u in SPH-EXA time loop, if cooling removes most of u during that step 
     * those forces describe gas that no longer exists. The gas would over-collapse before the 
     * hydro noticed. So, the Changa cooling backend would be accurate over a long timestep but
     * the coupling with the hydro part could be wrong. 
     * 
     * Given that no minimum cooling time is computed in ChaNGa, there is no equivalent to this 
     * function in ChaNGa. However, the CoolEdotInstantCode function computes the value of du/dt 
     * due to radiative cooling only and it can be therefore used to compute a timestep limit
     * based on internal energy variation, particularly as |u / edot|. This quantity measures, 
     * for a given particle, how long it would take for the internal energy to be completely 
     * removed by radiative cooling at current cooling rate and it is therefore a measure of
     * the cooling timescale. 
     * 
     * @param rho Pointer to density values
     * @param u Pointer to internal energy values
     * @param chemistry Pointer to chemical abundance values
     * @param first First particle index to compute minimum cooling time for
     * @param last Last particle index to compute minimum cooling time for
     * @return The minimum cooling time across all particles in the range
     */
    template<typename Trho, typename Tu>
    double min_cooling_time(const Trho* rho, const Tu* u, const FieldPtrs& chemistry, const size_t first,
                            const size_t last)
    {
        T* yHI   = util::get<"Y_HI", CoolingFields>(chemistry);
        T* yHeI  = util::get<"Y_HeI", CoolingFields>(chemistry);
        T* yHeII = util::get<"Y_HeII", CoolingFields>(chemistry);

        // In case of thermal balance the cooling time is in principle infinite
        double minCoolingTimeScale = std::numeric_limits<double>::max();

#pragma omp parallel for schedule(static) reduction(min : minCoolingTimeScale)
        for (size_t i = first; i < last; ++i)
        {
            COOLPARTICLE cp;
            cp.Y_HI         = yHI[i];
            cp.Y_HeI        = yHeI[i];
            cp.Y_HeII       = yHeII[i];
            cp.dLymanWerner = 0.0;

            double pos[3] = {0.0, 0.0, 0.0}; // Changa ignores the position argument in CoolEdotInstantCode
            const double edot = CoolEdotInstantCode(cl_, &cp, u[i], rho[i], ZMetal, pos);

            if (edot != 0.0) { minCoolingTimeScale = std::min(minCoolingTimeScale, std::abs(u[i] / edot)); }
        }

        return minCoolingTimeScale;
    }

    static std::vector<const char*> getParameterNames()
    {
        return {"m_code_in_ms", "l_code_in_kpc", "dMassFracHelium", "dCoolingTmin",
                "dCoolingTmax", "nCoolingTable"};
    }

    std::vector<typename ChangaCosmoCooler<T>::FieldVariant> getFields()
    {
        return {&scales_.m_code_in_ms, &scales_.l_code_in_kpc, &param_.dMassFracHelium,
                &param_.dCoolingTmin,   &param_.dCoolingTmax,   &param_.nCoolingTable};
    }
};

} // namespace cooling
