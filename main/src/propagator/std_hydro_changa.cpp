/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Translation unit for the std hydro changa propagator initializer
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#ifdef SPH_EXA_HAVE_CHANGA_COOLING

#include "sph/types.hpp"
#include "propagator.h"
#include "std_hydro_changa.hpp"

namespace sphexa
{

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
PropLib<DomainType, ParticleDataType>::makeHydroChangaProp(std::ostream& output, size_t rank,
                                                            const InitSettings& settings)
{
    return std::make_unique<HydroChangaProp<DomainType, ParticleDataType>>(output, rank, settings);
}

#ifdef USE_CUDA
template struct PropLib<cstone::Domain<SphTypes::KeyType, SphTypes::CoordinateType, cstone::execution::Gpu>,
                        SimulationData<cstone::execution::Gpu>>;
#else
template struct PropLib<cstone::Domain<SphTypes::KeyType, SphTypes::CoordinateType, cstone::execution::Cpu>,
                        SimulationData<cstone::execution::Cpu>>;
#endif

} // namespace sphexa

#endif
