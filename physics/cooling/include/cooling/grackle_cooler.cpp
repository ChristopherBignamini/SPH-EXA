//
// Created by Noah Kubli on 24.11.22.
//

extern "C"
{
#include <grackle.h>
}

#include <optional>
#include <vector>

#include "grackle_cooler_impl.hpp"

namespace cooling
{

template<typename T>
GrackleCooler<T>::GrackleCooler()
    : impl_ptr(new Impl)
{
}

template<typename T>
GrackleCooler<T>::~GrackleCooler() = default;

template<typename T>
void GrackleCooler<T>::init(const bool comoving_coordinates, const std::optional<T> time_unit)
{
    impl_ptr->init(comoving_coordinates, time_unit);
}

template<typename T>
template<typename Trho, typename Tu>
void GrackleCooler<T>::cool_particles(const T dt, const Trho* rho, const Tu* u, const GrackleFieldPtrs& chemistry, Tu* du,
                               const size_t first, const size_t last)
{
    impl_ptr->cool_particles(dt, rho, u, chemistry, du, first, last);
}

template void GrackleCooler<double>::cool_particles(double, const float*, const double*, const GrackleFieldPtrs&, double*,
                                             const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu, typename Ttemp>
void GrackleCooler<T>::computeTemperature(const Trho* rho, const Tu* u, const GrackleFieldPtrs& chemistry, Ttemp* temp,
                                   const size_t first, const size_t last)
{
    return impl_ptr->computeTemperature(rho, u, chemistry, temp, first, last);
}

template void GrackleCooler<double>::computeTemperature(const float*, const double*, const GrackleFieldPtrs&, double*,
                                                 const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu, typename Tp>
void GrackleCooler<T>::computePressures(const Trho* rho, const Tu* u, const GrackleFieldPtrs& chemistry, Tp* p,
                                 const size_t first, const size_t last)
{
    impl_ptr->computePressures(rho, u, chemistry, p, first, last);
}

template void GrackleCooler<double>::computePressures(const float*, const double*, const GrackleFieldPtrs&, float*,
                                               const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu, typename Tgamma>
void GrackleCooler<T>::computeAdiabaticIndices(const Trho* rho, const Tu* u, const GrackleFieldPtrs& chemistry, Tgamma* gamma,
                                        const size_t first, const size_t last)
{
    return impl_ptr->computeAdiabaticIndices(rho, u, chemistry, gamma, first, last);
}

template void GrackleCooler<double>::computeAdiabaticIndices(const float*, const double*, const GrackleFieldPtrs&, float*,
                                                      const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu>
double GrackleCooler<T>::cooling_timestep(const Trho* rho, const Tu* u, const GrackleFieldPtrs& chemistry, const size_t first,
                                   const size_t last)
{
    return ct_crit * impl_ptr->min_cooling_time(rho, u, chemistry, first, last);
}

template double GrackleCooler<double>::cooling_timestep(const float*, const double*, const GrackleFieldPtrs&, const size_t,
                                                 const size_t);

template<typename T>
std::vector<const char*> GrackleCooler<T>::getParameterNames()
{
    return GrackleCooler<T>::Impl::getParameterNames();
}

template<typename T>
std::vector<typename GrackleCooler<T>::FieldVariant> GrackleCooler<T>::getParameters()
{
    return impl_ptr->getFields();
}

template struct GrackleCooler<double>;

} // namespace cooling
