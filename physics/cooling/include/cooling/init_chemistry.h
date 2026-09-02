//
// Created by Noah Kubli on 29.11.22.
//

#pragma once

#include <iostream>
#include "cstone/fields/field_get.hpp"

namespace cooling
{
//! @brief Initialize Grackle chemistry arrays with default data
template<typename ChemistryData>
void initChemistryData(ChemistryData& d, size_t n)
{
    std::cout << "resizing: " << n << std::endl;
    using T = typename ChemistryData::RealType;

    d.resize(n);

    auto fillVec = [](std::vector<T>& vec, T value) { std::fill(vec.begin(), vec.end(), value); };

#ifdef SPH_EXA_HAVE_CHANGA_COOLING
    {
        /* ChaNGa COOLING_COSMO: abundances per baryon, not mass fractions. Neutral primordial
         * gas at the default dMassFracHelium = 0.25, i.e. Y_H = 1 - 0.25 = 0.75 and
         * Y_He = 0.25/4 = 0.0625, matching CoolDefaultParticleData in cooling_cosmo.c.
         *
         * These deliberately do not sum to 1, unlike GRACKLE's mass fractions in the branch
         * below. A helium nucleus carries 4 baryons, so there is less than one nucleus per
         * baryon; the /4 in clInitConstants in the mass-fraction-to-per-baryon conversion. The
         * normalisation that holds is on mass:  Y_H*1 + Y_He*4 = 0.75 + 0.25 = 1.
         *
         * Unlike GRACKLE fractions below, these are NOT an initial condition in the usual
         * sense. GRACKLE evolves a non-equilibrium network, so its seed shapes the whole
         * trajectory. cooling_cosmo instead re-solves ionization equilibrium from (T, rho) on
         * every call: clIntegrateEnergy computes its own abundances at entry (clAbunds) and
         * assigns them back to the caller on exit, so the incoming values are never read. They
         * are outputs, not state.
         *
         * One quantity does see them, exactly once. On the very first step, before
         * cool_particles has run, coolingTimestep reads them to build the COOLPARTICLE for
         * CoolEdotInstantCode -- which, unlike the integrator, does NOT re-solve equilibrium.
         * eos_cooling is unaffected either way, since p and gamma ignore the chemistry.
         *
         * Given the value below, edot is exactly 0, min_cooling_time's `edot != 0` guard skips
         * every particle, and the first step carries no cooling constraint. That is physically
         * correct (neutral gas with no radiation field does not cool) and harmless here:
         * dt is still bounded by the Courant and acceleration criteria, cool_particles
         * integrates correctly at any dt and floors the result at EMin > 0, and from step two
         * onwards the stored values are real equilibrium abundances.
         *
         * ChaNGa avoids the degeneracy because CoolInitEnergyAndParticleData seeds these same
         * neutral values and then immediately calls clAbunds to equilibrate at the initial
         * (T, rho), storing the result. That cannot be done here: rho does not exist yet at
         * initialization, it comes from the SPH density sum in computeForces.
         *
         * TODO (chemistry_init): the above issue needs some further investigation and a proper
         * solution.
         */
        fillVec(cstone::get<"Y_HI">(d), 0.75);
        fillVec(cstone::get<"Y_HeI">(d), 0.0625);
        fillVec(cstone::get<"Y_HeII">(d), 0.0);
    }
#else
    {
        // This is done so in the sample implementation from GRACKLE – don't know if really needed
        constexpr T tiny_number = 1.e-20;

        constexpr T metal_fraction     = 0.0;
        const T     non_metal_fraction = 1. - metal_fraction;

        fillVec(cstone::get<"HI_fraction">(d), non_metal_fraction * 0.76);
        fillVec(cstone::get<"HeI_fraction">(d), non_metal_fraction * 0.24);
        fillVec(cstone::get<"DI_fraction">(d), 2.0 * 3.4e-5);
        fillVec(cstone::get<"HII_fraction">(d), tiny_number);
        fillVec(cstone::get<"HeII_fraction">(d), tiny_number);
        fillVec(cstone::get<"HeIII_fraction">(d), tiny_number);
        fillVec(cstone::get<"e_fraction">(d), tiny_number);
        fillVec(cstone::get<"HM_fraction">(d), tiny_number);
        fillVec(cstone::get<"H2I_fraction">(d), tiny_number);
        fillVec(cstone::get<"H2II_fraction">(d), tiny_number);
        fillVec(cstone::get<"DII_fraction">(d), tiny_number);
        fillVec(cstone::get<"HDI_fraction">(d), tiny_number);
        fillVec(cstone::get<"e_fraction">(d), tiny_number);
        fillVec(cstone::get<"metal_fraction">(d), metal_fraction);
        fillVec(cstone::get<"volumetric_heating_rate">(d), 0.);
        fillVec(cstone::get<"specific_heating_rate">(d), 0.);
        fillVec(cstone::get<"RT_heating_rate">(d), 0.);
        fillVec(cstone::get<"RT_HI_ionization_rate">(d), 0.);
        fillVec(cstone::get<"RT_HeI_ionization_rate">(d), 0.);
        fillVec(cstone::get<"RT_HeII_ionization_rate">(d), 0.);
        fillVec(cstone::get<"RT_H2_dissociation_rate">(d), 0.);
        fillVec(cstone::get<"H2_self_shielding_length">(d), 0.);
    }
#endif

    std::cout << "resizing: " << d.fields[0].size() << std::endl;
}
} // namespace cooling
