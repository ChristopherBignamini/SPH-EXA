/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Compile-time contract that every radiative cooling back-end must satisfy
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <optional>

namespace cooling
{

/*! @brief Cooling back-end interface concept
 *
 * @tparam C     the back-end under test
 * @tparam Trho  density type as stored by the hydro dataset
 * @tparam Tu    internal energy type; also the back-end's own real type
 * @tparam Tout  output type for pressure and adiabatic index
 *
 * The default type arguments are the combination the SPH-EXA hydro datasets actually use.
 * Not checked: every back-end must also provide
 * template<class Archive> void loadOrStoreAttributes(Archive*), taking the archives from
 * main/src/io -- requiring it here would make physics/cooling depend on the I/O layer.
 *
 * TODO (comoving): there is no time/redshift reference in the contract, and neither back-end
 * supports a changing redshift in current implementation.
 *
 *   - ChangaCosmoCooler calls ChaNGa's CoolSetTime(z) once in init() and rejects
 *     comoving=true.
 *   - GrackleCooler sets code_units.a_value = 1.0 in init() and never updates it. GRACKLE
 *     derives the redshift itself, z = 1/(a_value*a_units) - 1 and needs the a_value
 *     to be refreshed before each cool_particles.
 *
 * So the two would need different strategies: an explicit setTime(t, z) for ChaNGa, a writable
 * expansion factor for GRACKLE. A single contract member could serve both -- setTime(t, z)
 * with the GRACKLE implementation writing a_value = 1/(1+z).
 */
template<class C, class Trho = float, class Tu = double, class Tout = float>
concept CoolingBackend =
    // the field list drives ChemistryData, the domain sync and the file output
    requires {
        typename C::CoolingFields;
        typename C::FieldPtrs;
    } &&
    std::default_initializable<C> &&
    requires(C c, const typename C::FieldPtrs& chem, Trho* rho, Tu* u, Tu* du, Tout* out, std::size_t i) {
        //! @brief must be called once before any other member
        { c.init(bool{}, std::optional<Tu>{}) };

        //! @brief advance chemistry and energy over dt, accumulating the rate into du
        { c.cool_particles(Tu{}, rho, u, chem, du, i, i) } -> std::same_as<void>;

        //! @brief EOS quantities, used by eos_cooling
        { c.computePressures(rho, u, chem, out, i, i) } -> std::same_as<void>;
        { c.computeAdiabaticIndices(rho, u, chem, out, i, i) } -> std::same_as<void>;

        //! @brief smallest cooling time over [first, last), safety factor already applied
        { c.cooling_timestep(rho, u, chem, i, i) } -> std::convertible_to<double>;
    };

} // namespace cooling
