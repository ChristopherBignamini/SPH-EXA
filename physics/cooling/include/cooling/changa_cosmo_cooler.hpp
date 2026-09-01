/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Interface to ChaNGa's native COOLING_COSMO back-end for radiative cooling
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "cstone/fields/field_get.hpp"
#include "cstone/util/type_list.hpp"
#include "cstone/util/value_list.hpp"

#include "cooling_backend.hpp"

namespace cooling
{

template<typename T>
struct ChangaCosmoCooler
{
public:
    /*! @brief Per-particle chemistry state
     *
     * Named Y_* rather than *_fraction on purpose: ChaNGa stores ABUNDANCES PER BARYON, not
     * mass fractions. Reusing GRACKLE's *_fraction names would make the two back-ends look
     * interchangeable in output files while meaning different physical quantities.
     */
    using CoolingFields = util::FieldList<"Y_HI", "Y_HeI", "Y_HeII">;

    inline static constexpr size_t numFields = util::FieldListSize<CoolingFields>{};

    using FieldPtrs = util::Reduce<std::tuple, util::Repeat<util::TypeList<std::add_pointer_t<T>>, numFields>>;

    ChangaCosmoCooler();

    ~ChangaCosmoCooler();

    /*! @brief Must be called before any other member and after parameters are set
     *
     * Also fixes the redshift for the whole run: ChaNGa needs CoolSetTime(z) to populate the
     * shared rate context, and there is no per-step hook for it in the CoolingBackend
     * contract. @p comoving_coordinates is therefore rejected -- see the TODO(comoving) note
     * in cooling_backend.hpp.
     */
    void init(bool comoving_coordinates = false, std::optional<T> time_unit = std::nullopt);

    //! @brief Integrate cooling over @p dt, writing the resulting energy rate into du
    template<typename Trho, typename Tu>
    void cool_particles(T dt, const Trho* rho, const Tu* u, const FieldPtrs& chemistry, Tu* du, size_t first,
                        size_t last);

    //! @brief Pressure of an ideal gas at gamma = 5/3
    template<typename Trho, typename Tu, typename Tp>
    void computePressures(const Trho* rho, const Tu* u, const FieldPtrs& chemistry, Tp* p, size_t first, size_t last);

    //! @brief Constant 5/3, for the reason given on computePressures
    template<typename Trho, typename Tu, typename Tgamma>
    void computeAdiabaticIndices(const Trho* rho, const Tu* u, const FieldPtrs& chemistry, Tgamma* gamma, size_t first,
                                 size_t last);

    // @brief Cooling timescale evaluation
    template<typename Trho, typename Tu>
    double cooling_timestep(const Trho* rho, const Tu* u, const FieldPtrs& chemistry, size_t first, size_t last);

    //! @brief Parameter for cooling time criterion
    T ct_crit{0.1};

    // TODO: this is duplicated in GrackleCooler
    template<class Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        auto parameterNames = getParameterNames();
        auto parameters     = getParameters();
        // the two are maintained by hand and zipped by index below
        assert(parameterNames.size() == parameters.size());
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                ar->stepAttribute("cooling::" + attribute, location, attrSize);
            }
            catch (std::out_of_range&)
            {
                if (ar->rank() == 0)
                {
                    std::cout << "Attribute cooling::" << attribute
                              << " not set in file or initializer, setting to default value " << *location << std::endl;
                }
            }
        };
        for (size_t i = 0; i < parameterNames.size(); i++)
        {
            std::visit([&](auto* location) { optionalIO(std::string(parameterNames[i]), location, 1); }, parameters[i]);
        }
        optionalIO("cooling::ct_crit", &ct_crit, 1);
    }

    struct Impl;

private:
    std::unique_ptr<Impl> impl_ptr;
    using FieldVariant = std::variant<float*, double*, int*>;
    static std::vector<const char*> getParameterNames();
    std::vector<FieldVariant>       getParameters();
};

static_assert(CoolingBackend<ChangaCosmoCooler<double>>, "the ChaNGa cosmo cooler must satisfy the cooling back-end contract");

} // namespace cooling
