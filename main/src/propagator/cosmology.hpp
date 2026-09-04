/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief scale factor evolution a(t) for the comoving-coordinates propagator
 *
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "init/settings.hpp"

namespace sphexa
{

/*! @brief Base class for the scale factor evolution a(t) calculation
 *
 * Implementations must evaluate a(t) at an arbitrary simulation time t. Evaluating the same time twice, 
 * from different steps, must return bit-identical values -- that identity is what preserves exact momentum 
 * conservation under free streaming, because the scale factor used to drift a step must be the same one 
 * used to recover the momentum on the next.
 * 
 * Times are simulation times, i.e. d.ttot, which is zero at the start of the run. Mapping them onto
 * cosmic time, i.e. accounting for the scale factor the run starts from, is up to the implementation.
 *
 */
class ScaleFactorEvolution
{
public:
    virtual ~ScaleFactorEvolution() = default;

    //! @brief scale factor at simulation time t
    virtual double a(double t) const = 0;

    //! @brief Hubble parameter H = (da/dt)/a at simulation time t
    virtual double hubble(double t) const = 0;

protected:
    //! @brief to avoid slicing through references
    ScaleFactorEvolution()                                       = default;
    ScaleFactorEvolution(const ScaleFactorEvolution&)            = default;
    ScaleFactorEvolution(ScaleFactorEvolution&&)                 = default;
    ScaleFactorEvolution& operator=(const ScaleFactorEvolution&) = default;
    ScaleFactorEvolution& operator=(ScaleFactorEvolution&&)      = default;
};

/*! @brief the no-cosmology case: a static universe with a == 1 and H == 0
 */
class StaticScaleFactorEvolution final : public ScaleFactorEvolution
{
public:
    //! @brief scale factor, one at all times
    double a(double) const override { return 1.0; }

    //! @brief Hubble parameter, zero at all times
    double hubble(double) const override { return 0.0; }
};

/*! @brief closed-form scale factor evolution for a flat matter + Lambda cosmology
 *
 * Solves the Friedmann equation (da/dt / a)^2 = H0^2 * (omegaM / a^3 + omegaL) for a flat universe,
 * omegaM + omegaL = 1, which admits the closed form
 *
 *     a(t) = (omegaM/omegaL)^(1/3) * sinh^(2/3)(t / tLambda),   tLambda = 2 / (3 * H0 * sqrt(omegaL))
 *
 * The cosmic time offset corresponding to the initial scale factor is computed once by inverting the
 * relation above.
 *
 * Requires 0 < omegaM < 1, i.e. a strictly positive omegaL: tLambda diverges as omegaL -> 0 and the
 * closed form degenerates. That limit is the Einstein-de Sitter universe and has its own model,
 * EinsteinDeSitterScaleFactorEvolution, which the omegaM == 1 error message points at.
 */
class FlatLcdmScaleFactorEvolution final : public ScaleFactorEvolution
{
public:
    /*! @param H0       Hubble constant in code units, must be positive
     *  @param omegaM   matter density parameter, must lie in (0, 1), omegaL = 1 - omegaM is implied by flatness
     *  @param aStart   scale factor at simulation time zero, must be positive
     */
    FlatLcdmScaleFactorEvolution(double H0, double omegaM, double aStart)
    {
        if (H0 <= 0.0) { throw std::runtime_error("FlatLcdmScaleFactorEvolution: H0 must be positive\n"); }
        if (aStart <= 0.0) { throw std::runtime_error("FlatLcdmScaleFactorEvolution: aStart must be positive\n"); }
        if (omegaM <= 0.0 || omegaM >= 1.0)
        {
            throw std::runtime_error("FlatLcdmScaleFactorEvolution: omegaM must lie in (0, 1), use "
                                     "EinsteinDeSitterScaleFactorEvolution for omegaM == 1\n");
        }

        double omegaL = 1.0 - omegaM;

        tLambda_ = 2.0 / (3.0 * H0 * std::sqrt(omegaL));
        aCoeff_  = std::cbrt(omegaM / omegaL);
        tStart_  = tLambda_ * std::asinh(std::sqrt(omegaL / omegaM) * std::pow(aStart, 1.5));
    }

    //! @brief scale factor at simulation time t
    double a(double t) const override { return aCoeff_ * std::pow(std::sinh((tStart_ + t) / tLambda_), 2.0 / 3.0); }

    //! @brief Hubble parameter H = (da/dt)/a at simulation time t
    double hubble(double t) const override { return 2.0 / (3.0 * tLambda_ * std::tanh((tStart_ + t) / tLambda_)); }

private:
    double tLambda_{0.0};
    double tStart_{0.0};
    double aCoeff_{1.0};
};

/*! @brief closed-form scale factor evolution for an Einstein-de Sitter universe, omegaM == 1, omegaL == 0
 *
 * The matter-only flat solution, i.e. the omegaL -> 0 limit of FlatLcdmScaleFactorEvolution, for which the
 * Friedmann equation (da/dt / a)^2 = H0^2 / a^3 integrates to
 *
 *     a(t) = ((3/2) * H0 * t)^(2/3),   H(t) = 2 / (3 * t)
 *
 * It is a separate model rather than a branch of the Lambda solution because tLambda = 2/(3*H0*sqrt(omegaL))
 * diverges at omegaL == 0, which turns the sinh form into inf * 0. No omegaM enters the expressions: flatness
 * with omegaL == 0 fixes omegaM == 1 exactly, so the sqrt(omegaM) factor of the general matter-only solution
 * is one.
 */
class EinsteinDeSitterScaleFactorEvolution final : public ScaleFactorEvolution
{
public:
    /*! @param H0       Hubble constant in code units, must be positive
     *  @param aStart   scale factor at simulation time zero, must be positive
     */
    EinsteinDeSitterScaleFactorEvolution(double H0, double aStart)
        : H0_(H0)
    {
        if (H0 <= 0.0) { throw std::runtime_error("EinsteinDeSitterScaleFactorEvolution: H0 must be positive\n"); }
        if (aStart <= 0.0)
        {
            throw std::runtime_error("EinsteinDeSitterScaleFactorEvolution: aStart must be positive\n");
        }

        // invert a = ((3/2) * H0 * t)^(2/3)
        tStart_ = std::pow(aStart, 1.5) / (1.5 * H0_);
    }

    //! @brief scale factor at simulation time t
    double a(double t) const override { return std::pow(1.5 * H0_ * (tStart_ + t), 2.0 / 3.0); }

    //! @brief Hubble parameter H = (da/dt)/a at simulation time t
    double hubble(double t) const override { return 2.0 / (3.0 * (tStart_ + t)); }

private:
    double H0_{0.0};
    double tStart_{0.0};
};

//! @brief keys under which the cosmological parameters are looked up in InitSettings
inline const std::string cosmoH0Key     = "cosmology::H0";
inline const std::string cosmoOmegaMKey = "cosmology::omegaM";
inline const std::string cosmoAStartKey = "cosmology::aStart";

/*! @brief build the scale factor evolution described by @p settings
 *
 * @param settings   the init settings, which may or may not carry the cosmology:: keys
 * @return           the matching model, never null
 *
 * None of the three keys present means no cosmology, i.e. a static universe. A partial set is rejected.
 *
 * The model itself is not a settable parameter, it is implied by omegaM: flatness makes omegaL = 1 - omegaM,
 * so omegaM == 1 is the Einstein-de Sitter universe and anything below it the Lambda one.
 * 
 * TODO (scale_factor_evolution): the implemented strategy for model selection fails when a new
 * model appears that omegaM does not distinguish, such as tabulated a(t) or a non-flat `cosmology, 
 * which is when a selector (or an input option such as --init or --prop) should be added.
 */
inline std::unique_ptr<ScaleFactorEvolution> makeScaleFactorEvolution(const InitSettings& settings)
{
    size_t numKeys = settings.count(cosmoH0Key) + settings.count(cosmoOmegaMKey) + settings.count(cosmoAStartKey);

    if (numKeys == 0) { return std::make_unique<StaticScaleFactorEvolution>(); }
    if (numKeys < 3)
    {
        std::string missing;
        for (const auto& key : {cosmoH0Key, cosmoOmegaMKey, cosmoAStartKey})
        {
            if (settings.count(key) == 0) { missing += " " + key; }
        }
        throw std::runtime_error("Incomplete cosmology in the init settings, missing:" + missing + "\n");
    }

    double H0     = settings.at(cosmoH0Key);
    double omegaM = settings.at(cosmoOmegaMKey);
    double aStart = settings.at(cosmoAStartKey);

    if (omegaM == 1.0) { return std::make_unique<EinsteinDeSitterScaleFactorEvolution>(H0, aStart); }
    if (omegaM > 1.0)
    {
        throw std::runtime_error("cosmology::omegaM > 1 is a closed universe, which is not covered by the flat "
                                 "closed-form solutions\n");
    }
    return std::make_unique<FlatLcdmScaleFactorEvolution>(H0, omegaM, aStart);
}

} // namespace sphexa
