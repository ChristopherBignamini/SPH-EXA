/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Evrard collapse with Changa cooling initialization
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "evrard_init.hpp"
#include "cooling/changa_cosmo_cooler.hpp"
#include "cooling/init_chemistry.h"

namespace sphexa
{

InitSettings evrardChangaCoolingConstants()
{
    // TODO (changa_cooling_backend_selection): in current implementation, the only changa 
    // cooling backend is cooling cosmo which only consider primordial chemistry. Therefore, 
    // in the constant list below, some parameters used in the grackle backend are not present 
    // (primordial_chemistry, dust_chemistry, metal_cooling, etc..). 
    return {{"gravConstant", 1.},
            {"r", 1.},
            {"mTotal", 1.},
            {"gamma", 5. / 3.},
            {"u0", 0.05},
            {"minDt", 1e-4},
            {"minDt_m1", 1e-4},
            {"mui", 10},
            {"ng0", 100},
            {"ngmax", 150},
            {"cooling::m_code_in_ms", 1e16},
            {"cooling::l_code_in_kpc", 46400.},
            {"cooling::dMassFracHelium", 0.25},
            {"cooling::dCoolingTmin", 10.},
            {"cooling::dCoolingTmax", 1e9},
            {"cooling::nCoolingTable", 15001}};
}

template<class Dataset>
class EvrardGlassSphereChangaCooling : public RadialProfile<Dataset>
{
    using Base = RadialProfile<Dataset>;
    using Base::settings_;

public:
    EvrardGlassSphereChangaCooling(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : Base(std::move(initBlock), evrardChangaCoolingConstants(), std::move(settingsFile), reader)
    {
    }

    cstone::Box<typename Dataset::RealType> initImpl(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                     IFileReader* reader) const override
    {
        auto radialTransform = [](auto r) { return std::sqrt(r); };
        auto globalBox = Base::init(rank, numRanks, cbrtNumPart, simData, reader, settings_.at("r"), radialTransform);
        initEvrardFields(simData.hydro, settings_);
        cooling::initChemistryData(simData.chem, simData.hydro.x.size());
        return globalBox;
    }
};

} // namespace sphexa