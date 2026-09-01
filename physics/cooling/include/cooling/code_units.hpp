/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief The code unit system shared by all cooling back-ends
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 *
 * Every back-end receives densities, energies and time steps in SPH-EXA code units and has to
 * hand its library the corresponding cgs scalings. Those scalings must agree between
 * back-ends: if two coolers derived them differently, identical code-unit input would mean
 * different physical states and comparing the back-ends would be meaningless. Hence one
 * definition here rather than a copy per adapter.
 */

#pragma once

#include <cmath>
#include <optional>

namespace cooling
{

//! @brief solar mass in g
inline constexpr double solarMassInGrams = 1.989e33;
//! @brief kiloparsec in cm
inline constexpr double kiloparsecInCm = 3.086e21;
//! @brief gravitational constant in cgs
inline constexpr double gravConstantCgs = 6.674e-8;

/*! @brief the tunable scales defining SPH-EXA's code units
 *
 * Held by each back-end as a member because they are loaded per run from file attributes
 * ("cooling::m_code_in_ms", "cooling::l_code_in_kpc"), but defined once here so the defaults
 * cannot drift apart between back-ends: two coolers disagreeing on them would silently
 * interpret the same code-unit density as different physical states.
 *
 * TODO(code-units-ownership): these describe how SPH-EXA's code units map to cgs, which is a
 * property of the SIMULATION, not of the cooling module -- hydro, gravity and I/O all use the
 * same units. Storing them on the cooler and serializing them as "cooling::*" is an accident
 * of the GRACKLE implementation that came first. The principled fix is to move them into the
 * simulation settings and hand a ready CodeUnits to whichever cooler is constructed, so a
 * back-end consumes the unit system instead of defining it. Deferred because it renames file
 * attributes (breaking checkpoint compatibility) and touches the initializers.
 */
struct CodeUnitScales
{
    //! @brief code unit mass, in solar masses
    double m_code_in_ms = 1e16;
    //! @brief code unit length, in kpc
    double l_code_in_kpc = 46400.;
};

//! @brief cgs value of one code unit, for each quantity a cooling back-end needs
struct CodeUnits
{
    double density;        //!< g/cm^3
    double time;           //!< s
    double length;         //!< cm
    double velocity;       //!< cm/s
    double specificEnergy; //!< erg/g, i.e. cm^2/s^2
};

/*! @brief derive the unit system from the code mass and length scales
 *
 * @param scales    code unit mass and length
 * @param timeUnit  code unit time in seconds; defaults to 1/sqrt(G*rho), which makes the
 *                  gravitational constant unity in code units
 */
inline CodeUnits makeCodeUnits(const CodeUnitScales& scales, std::optional<double> timeUnit = std::nullopt)
{
    CodeUnits units{};
    units.length         = scales.l_code_in_kpc * kiloparsecInCm;
    units.density        = scales.m_code_in_ms * solarMassInGrams / std::pow(units.length, 3);
    units.time           = timeUnit.value_or(std::sqrt(1. / (units.density * gravConstantCgs)));
    units.velocity       = units.length / units.time;
    units.specificEnergy = units.velocity * units.velocity;

    return units;
}

} // namespace cooling
