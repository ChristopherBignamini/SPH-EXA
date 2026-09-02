/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Contains the object holding all simulation data
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <mpi.h>

#include "cooling/chemistry_data.hpp"
#ifdef SPH_EXA_HAVE_GRACKLE
#include "cooling/grackle_cooler.hpp"
#endif
#ifdef SPH_EXA_HAVE_CHANGA_COOLING
#include "cooling/changa_cosmo_cooler.hpp"
#endif
#include "sph/particles_data.hpp"

namespace sphexa
{

//! @brief the place to store hydro, chemistry, nuclear and other simulation data
template<cstone::execution::Policy Execution>
class SimulationData
{
public:
    using Exec     = Execution;
    using KeyType  = sph::SphTypes::KeyType;
    using RealType = sph::SphTypes::CoordinateType;

    using HydroData = ParticlesData<Exec>;

    /* TODO (no-cooler): with SPH_EXA_COOLING=none neither flag is set, so this falls to the
     * GRACKLE branch and a build with no cooling still carries GRACKLE's 21-field chemistry
     * dataset. Implement a NoCooler with an empty CoolingFields list.
     */
#ifdef SPH_EXA_HAVE_CHANGA_COOLING
    using ChemData  = cooling::ChemistryData<RealType, cooling::ChangaCosmoCooler<RealType>>;
#else
    using ChemData  = cooling::ChemistryData<RealType, cooling::GrackleCooler<RealType>>;
#endif

    //! @brief spacially distributed data for hydrodynamics and gravity
    HydroData hydro;

    //! @brief chemistry data for radiative cooling, e.g. for GRACKLE
    ChemData chem;

    //! @brief non-spacially distributed nuclear abundances
    // NuclearData nuclear;

    MPI_Comm comm;

    //! @brief record user selection of output fields
    void setOutputFields(std::vector<std::string> outFields)
    {
        hydro.setOutputFields(outFields);
        chem.setOutputFields(outFields);

        if (!outFields.empty())
        {
            std::string msg;
            for (auto& s : outFields)
            {
                msg += s + ", ";
            }
            throw std::runtime_error("The following fields for output were not found: " + msg);
        }
    }
};

} // namespace sphexa
