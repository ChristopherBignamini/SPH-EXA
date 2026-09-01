/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Definitions for ChangaCosmoCooler, with explicit instantiations
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <optional>
#include <vector>

#include "changa_cosmo_cooler_impl.hpp"

namespace cooling
{

template<typename T>
ChangaCosmoCooler<T>::ChangaCosmoCooler()
    : impl_ptr(new Impl)
{
}

template<typename T>
ChangaCosmoCooler<T>::~ChangaCosmoCooler() = default;

template<typename T>
void ChangaCosmoCooler<T>::init(const bool comoving_coordinates, const std::optional<T> time_unit)
{
    impl_ptr->init(comoving_coordinates, time_unit);
}

template<typename T>
template<typename Trho, typename Tu>
void ChangaCosmoCooler<T>::cool_particles(const T dt, const Trho* rho, const Tu* u, const FieldPtrs& chemistry, Tu* du,
                                          const size_t first, const size_t last)
{
    impl_ptr->cool_particles(dt, rho, u, chemistry, du, first, last);
}

template void ChangaCosmoCooler<double>::cool_particles(double, const float*, const double*, const FieldPtrs&, double*,
                                                        const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu, typename Tp>
void ChangaCosmoCooler<T>::computePressures(const Trho* rho, const Tu* u, const FieldPtrs& chemistry, Tp* p,
                                            const size_t first, const size_t last)
{
    impl_ptr->computePressures(rho, u, chemistry, p, first, last);
}

template void ChangaCosmoCooler<double>::computePressures(const float*, const double*, const FieldPtrs&, float*,
                                                          const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu, typename Tgamma>
void ChangaCosmoCooler<T>::computeAdiabaticIndices(const Trho* rho, const Tu* u, const FieldPtrs& chemistry,
                                                   Tgamma* gamma, const size_t first, const size_t last)
{
    impl_ptr->computeAdiabaticIndices(rho, u, chemistry, gamma, first, last);
}

template void ChangaCosmoCooler<double>::computeAdiabaticIndices(const float*, const double*, const FieldPtrs&, float*,
                                                                 const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu>
double ChangaCosmoCooler<T>::cooling_timestep(const Trho* rho, const Tu* u, const FieldPtrs& chemistry,
                                              const size_t first, const size_t last)
{
    return ct_crit * impl_ptr->min_cooling_time(rho, u, chemistry, first, last);
}

template double ChangaCosmoCooler<double>::cooling_timestep(const float*, const double*, const FieldPtrs&,
                                                            const size_t, const size_t);

template<typename T>
std::vector<const char*> ChangaCosmoCooler<T>::getParameterNames()
{
    return ChangaCosmoCooler<T>::Impl::getParameterNames();
}

template<typename T>
std::vector<typename ChangaCosmoCooler<T>::FieldVariant> ChangaCosmoCooler<T>::getParameters()
{
    return impl_ptr->getFields();
}

template struct ChangaCosmoCooler<double>;

} // namespace cooling
