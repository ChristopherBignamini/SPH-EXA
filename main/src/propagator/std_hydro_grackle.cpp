/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief Translation unit for the std hydro grackle propagator initializer
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#ifdef SPH_EXA_HAVE_GRACKLE

#include "ipropagator_init.hpp"
#include "propagator.hpp"
#include "std_hydro_grackle.hpp"

namespace sphexa
{

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
PropLib<DomainType, ParticleDataType>::makeHydroGrackleProp(std::ostream& output, size_t rank, const InitSettings& settings)
{
    return std::make_unique<HydroGrackleProp<DomainType, ParticleDataType>>(output, rank, settings);
}
//#endif

#ifdef USE_CUDA
template struct PropLib<cstone::Domain<unsigned long, double, cstone::GpuTag>, SimulationData<cstone::GpuTag> >;
#else
template struct PropLib<cstone::Domain<unsigned long, double, cstone::CpuTag>, SimulationData<cstone::CpuTag> >;
#endif

} // namespace sphexa

#endif