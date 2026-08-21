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
 */
template<class C, class Trho = float, class Tu = double, class Tout = float>
concept CoolingBackend =
    // the field list drives ChemistryData, the domain sync and the file output
    requires {
        typename C::CoolingFields;
        //! @brief the tuple of field pointers cool_particles & co. are called with
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
