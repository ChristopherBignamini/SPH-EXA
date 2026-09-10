/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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
 * @brief A Propagator class for modern SPH with generalized volume elements in comoving coordinates
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include <memory>

#include "cstone/fields/field_get.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"
#include "sph/positions_comoving.hpp"

#include "ipropagator.hpp"
#include "gravity_wrapper.hpp"
#include "cosmology.hpp"

namespace sphexa
{

using namespace sph;
using util::FieldList;

template<bool avClean, class DomainType, class DataType>
class HydroVeComovingProp : public Propagator<DomainType, DataType>
{
protected:
    using Base = Propagator<DomainType, DataType>;
    using Base::pmReader;
    using Base::timer;

    using T             = typename DataType::RealType;
    using KeyType       = typename DataType::KeyType;
    using Tmass         = typename DataType::HydroData::Tmass;
    using MultipoleType = ryoanji::CartesianQuadrupole<Tmass>;

    using Acc       = typename DataType::Exec;
    using MHolder_t = std::conditional_t<cstone::execution::HaveGpu<Acc>{},
                                         MultipoleHolderGpu<MultipoleType, DomainType, typename DataType::HydroData>,
                                         MultipoleHolderCpu<MultipoleType, DomainType, typename DataType::HydroData>>;
    template<class VType>
    using AccVector =
        std::conditional_t<cstone::execution::HaveGpu<Acc>{}, cstone::DeviceVector<VType>, std::vector<VType>>;

    MHolder_t      mHolder_;
    GroupData<Acc> groups_;

    //! @brief acceleration element type, must match the ax/ay/az particle fields
    using HydroType = typename DataType::HydroData::HydroType;

    //! @brief gravitational acceleration, kept separate from the hydro acceleration in ax/ay/az
    AccVector<HydroType> agx_, agy_, agz_;

    /*! @brief scale factors consumed by the comoving integrator, see sph::positionUpdateComoving
     *
     * aNow_ = a(t_n), aPrevHalf_ = a(t_n - dt_m1/2), aHalf_ = a(t_n + dt/2), aNext_ = a(t_n+1). aNow_ is the
     * one the acceleration weights are derived from, the others drive the drift and the velocity output.
     *
     * They are recomputed once per step in integrate() from cosmo_, so that all four derive from a single
     * a(t). Keeping them derived rather than independently settable is what guarantees the alignment
     * aPrevHalf(step n+1) == aHalf(step n), on which the exact momentum conservation of the integrator
     * rests.
     */
    T aNow_{1}, aPrevHalf_{1}, aHalf_{1}, aNext_{1};

    //! @brief scale factor evolution, never null, the static universe when the settings carry no cosmology
    std::unique_ptr<ScaleFactorEvolution> cosmo_;

    //! @brief true until the first call to updateScaleFactors, which has no previous step to carry over
    bool firstStep_{true};


    /*! @brief compute the scale factors of the current step
     *
     * TODO: check if the following condition is correct
     * Must be called after computeTimestep, which has already advanced d.ttot to the end of the step and
     * updated d.minDt / d.minDt_m1. The start of the current step is therefore d.ttot - d.minDt.
     *
     * aPrevHalf_ is carried over from the previous step's aHalf_ rather than recomputed. The two denote the
     * same instant, but reaching it as (ttot - dt) + dt/2 on one step and as ttot - dt_m1/2 on the next
     * rounds differently, and sph::positionUpdateComoving conserves the canonical momentum exactly under
     * the condition that the scale factor that divided the drift is bitwise the one that multiplies
     * the momentum recovery. Carrying the value over makes that identity hold by construction.
     */
    void updateScaleFactors(double ttot, double dt, double dt_m1)
    {
        double tn = ttot - dt;

        aPrevHalf_ = firstStep_ ? T(cosmo_->a(tn - 0.5 * dt_m1)) : aHalf_;
        aNow_      = cosmo_->a(tn);
        aHalf_     = cosmo_->a(tn + 0.5 * dt);
        aNext_     = cosmo_->a(tn + dt);
        firstStep_ = false;
    }

    /*! @brief the list of conserved particles fields with values preserved between iterations
     *
     * x, y, z, h and m are automatically considered conserved and must not be specified in this list
     *
     * Unlike HydroVeProp, the thermal variable is "u" rather than "temp". The comoving propagator evolves the
     * comoving specific internal energy u_hat = a^(3*(gamma-1)) * u_phys, which absorbs the adiabatic cooling
     * of the expansion (see sph::updatePositionsComovingHost). This choice also selects the code execution paths
     * by making computeEOS take idealGasEOS_u and sph::computePositionsComoving take updateIntEnergyHost,
     * both of which work directly on u_hat.
     *
     * NOTE: Dividing u by the heat capacity produces now a^(3*(gamma-1)) * T, which is not a physicaltemperature.
     */
    using ConservedFields = FieldList<"u", "vx", "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "alpha", "id">;

    //! @brief list of dependent fields, these may be used as scratch space during domain sync
    using DependentFields_ = FieldList<"ax", "ay", "az", "prho", "c", "du", "c11", "c12", "c13", "c22", "c23", "c33",
                                       "xm", "kx", "nc", "dtCourant">;

    //! @brief velocity gradient fields will only be allocated when avClean is true
    using GradVFields = FieldList<"dV11", "dV12", "dV13", "dV22", "dV23", "dV33">;

    //! @brief what will be allocated based AV cleaning choice
    using DependentFields =
        std::conditional_t<avClean, decltype(DependentFields_{} + GradVFields{}), decltype(DependentFields_{})>;

    const InitSettings& settings_;

public:
    HydroVeComovingProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
        , cosmo_(makeScaleFactorEvolution(settings))
        , settings_(settings)
    {
        if (avClean && rank == 0) { std::cout << "AV cleaning is activated" << std::endl; }

        if (rank == 0)
        {
            if (settings.count(cosmoH0Key) == 0) { std::cout << "No cosmology, the scale factor stays at one\n"; }
            else
            {
                std::cout << "Cosmology: H0 = " << settings.at(cosmoH0Key) << ", omegaM = "
                          << settings.at(cosmoOmegaMKey) << ", aStart = " << settings.at(cosmoAStartKey) << std::endl;
            }
        }
    }

    std::vector<std::string> conservedFields() const override
    {
        std::vector<std::string> ret{"x", "y", "z", "h", "m"};
        for_each_tuple([&ret](auto f) { ret.push_back(f.value); }, make_tuple(ConservedFields{}));
        return ret;
    }

    void activateFields(DataType& simData) override
    {
        auto& d = simData.hydro;
        //! @brief Fields accessed in domain sync (x,y,z,h,m,keys) are not part of extensible lists.
        d.setConserved("x", "y", "z", "h", "m");
        d.setDependent("keys");
        std::apply([&d](auto... f) { d.setConserved(f.value...); }, make_tuple(ConservedFields{}));
        std::apply([&d](auto... f) { d.setDependent(f.value...); }, make_tuple(DependentFields{}));
    }

    void sync(DomainType& domain, DataType& simData) override
    {
        auto& d = simData.hydro;
        if (d.g != 0.0)
        {
            domain.syncGrav(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d), get<"m">(d),
                            get<ConservedFields>(d), get<DependentFields>(d));
        }
        else
        {
            domain.sync(get<"keys">(d), get<"x">(d), get<"y">(d), get<"z">(d), get<"h">(d),
                        std::tuple_cat(std::tie(get<"m">(d)), get<ConservedFields>(d)), get<DependentFields>(d));
        }
        d.treeView = domain.octreeProperties();
    }

    /*! @brief Rescale the internal comoving energy rate d.du by 1/aNow, according to the comoving description of system evolution
     *
     * TODO (energy scaling): this is a temporary solution to correct the energy variation calculated by SPH-EXA in
     * physical coordinates in order to be consistent with the switch to comoving variables (position, density, etc...)
     * In a future implementation we could think of a more clean solution.
     *
     * NOTE: this is called from computeForces, where d.ttot is still the time the state belongs to: computeTimestep has
     * not yet run, so this is the same t_n that integrate() derives as d.ttot - d.minDt, and therefore the
     * same scale factor that updateScaleFactors will store in aNow_.
     *
     * NOTE: by including the expansion factor in the description, a Hubble dragging term given by -3*H*(gamma-1)*u
     * is introduced in the energy equation. That term is a cooling contribution purely due to the expansion itself and
     * can be "removed" by switchig to the comoving internal energy u_c = u* a^(3*(gamma-1)). This means that the internal
     * energy of the HydroData structure is assumed to be the comoving one.
     */
    void scaleInternalEnergyRate(DomainType& domain, typename DataType::HydroData& d, size_t first, size_t last)
    {
        double aNow = cosmo_->a(d.ttot);
        // exact for the static universe, and a no-op at the present epoch of a real one
        if (aNow == 1.0) { return; }

        auto* du = cstone::rawPtr(d.du);
        cstone::scale(domain.exec(), du + first, du + last, du + first, 1.0 / aNow);
    }

    /*! @brief Reject the equations of state whose scale-factor weighting this propagator does not implement
     *
     * updatePositionsComovingHost hard-codes wHydro = a^(-3*(gamma-1)), which is the weight that converts the
     * hydrodynamic acceleration into a canonical momentum rate only for the ideal gas driven by the comoving
     * internal energy u_c. Writing p_stored = a^n * p_phys, where p_ is stands for pressure, the volume elements
     * and the kernel gradient turn that into d.ax = a^(n-2) * f_phys, and reaching a * f_phys needs
     * wHydro = a^(3-n). The three equations ofstate do not share an n:
     *
     *   ideal gas with u_hat   p_stored = (gamma-1) * rho_c * u_c   = a^(3*gamma) * p_phys   -> w_hydro = a^(-3*(gamma-1))
     *   isothermal             p_stored = rho_c * c^2               = a^3 * p_phys           -> w_hydro = 1
     *   polytropic             p_stored = K * rho_c^gamma_poly      = a^(3*gamma_poly)       -> w_hydro = a^(-3*(gamma_poly-1))
     *
     * The isothermal case would therefore be wrong by a^(-3*(gamma-1)) and the polytropic one needs d.polytropic_index
     * rather than d.gamma. Both are also conceptually wrong for this propagator: neither closure involves the
     * internal energy, so the comoving energy variable that motivates this propagator carries no meaning for them.
     * In the ideal case p_stored corresponds to the comoving pressure.
     *
     * The check cannot live in the constructor, which never sees the dataset, nor in activateFields, which runs
     * before the initializer has populated d.eosChoice.
     *
     * TODO: this is a temporary solution to avoid the use of unsupported equations of state with the comoving propagator.
     * In a future implementation we could think of a more clean solution.
     * TODO: it should be possible to implement the isothermal and polytropic equations of state in a comoving description
     * with minimal code changes by using an effective gamma depending on the EoS choice and by changing the ConservedFields
     * list accordingly since the comoving internal energy is not needed for those two EoS.
     */
    void checkEosSupported(const typename DataType::HydroData& d) const
    {
        if (d.eosChoice == sph::EosType::idealGas) { return; }

        throw std::runtime_error("The comoving propagator only supports the ideal gas equation of state: the "
                                 "scale-factor weight of the hydrodynamic acceleration is derived for it alone\n");
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        timer.start();
        pmReader.start();
        checkEosSupported(simData.hydro);
        sync(domain, simData);
        timer.step("domain::sync");
        Base::logDomainStats(domain, simData);

        auto& d = simData.hydro;
        d.resize(domain.nParticlesWithHalos());
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        fillMassHalos(domain.exec(), get<"m">(d), first, last);

        computeGroups(first, last, d, domain.box(), groups_);
        updateSmoothingLengthIterative(groups_.view(), d, domain.box());
        findNeighborsSfc(groups_.view(), d, domain.box());
        timer.step("FindNeighbors");
        pmReader.step();

        computeXMass(groups_.view(), d, domain.box());
        timer.step("XMass");
        domain.exchangeHalos(std::tie(get<"xm">(d)), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        computeVe(groups_.view(), d, domain.box());
        timer.step("Generalized Volume Elements");
        domain.exchangeHalos(get<"vx", "vy", "vz", "kx">(d), get<"ax">(d), get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        release(d, "ay", "az");
        acquire(d, "divv", "gradh");
        computeIadDivvCurlvGradh(groups_.view(), d, domain.box());
        d.minDtRho = rhoTimestep(first, last, d);
        timer.step("IadVelocityDivCurlGradh");

        computeEOS(first, last, d);
        timer.step("EquationOfState");

        domain.exchangeHalos(get<"c11", "c12", "c13", "c22", "c23", "c33", "divv", "c">(d), get<"ax">(d),
                             get<"keys">(d));
        timer.step("mpi::synchronizeHalos");

        computeAVswitches(groups_.view(), d, domain.box());
        timer.step("AVswitches");

        if (avClean)
        {
            domain.exchangeHalos(get<"dV11", "dV12", "dV13", "dV22", "dV23", "dV33", "prho", "alpha">(d), get<"ax">(d),
                                 get<"keys">(d));
        }
        else { domain.exchangeHalos(get<"prho", "alpha">(d), get<"ax">(d), get<"keys">(d)); }
        timer.step("mpi::synchronizeHalos");

        release(d, "divv", "gradh");
        acquire(d, "ay", "az");
        computeMomentumEnergy<avClean>(groups_.view(), nullptr, d, domain.box());
        scaleInternalEnergyRate(domain, d, first, last);
        timer.step("MomentumAndEnergy");
        pmReader.step();

        if (d.g != 0.0)
        {
            //! gravitational acceleration is accumulated into a dedicated set, separate from the hydro ax/ay/az
            reallocate(domain.nParticlesWithHalos(), d.getAllocGrowthRate(), agx_, agy_, agz_);
            cstone::fill(domain.exec(), agx_.begin(), agx_.end(), HydroType(0));
            cstone::fill(domain.exec(), agy_.begin(), agy_.end(), HydroType(0));
            cstone::fill(domain.exec(), agz_.begin(), agz_.end(), HydroType(0));

            auto groups = mHolder_.computeSpatialGroups(d, domain);
            mHolder_.upsweep(d, domain);
            timer.step("Upsweep");
            pmReader.step();
            mHolder_.traverse(groups, d, domain, cstone::rawPtr(agx_), cstone::rawPtr(agy_), cstone::rawPtr(agz_));
            timer.step("Gravity");
            pmReader.step();

            auto stats = mHolder_.readStats();
            timer.logStatistics("sumP2P", stats[0] / timer.getLastStepTime());
            timer.logStatistics("sumM2P", stats[2] / timer.getLastStepTime());
        }
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeTimestep(first, last, d);
        updateScaleFactors(d.ttot, d.minDt, d.minDt_m1);
        timer.step("Timestep");
        //! gravity is stored in a separate acceleration set; pass it (or nullptr) to the combining integrator
        const HydroType* agx = d.g != 0.0 ? cstone::rawPtr(agx_) : nullptr;
        const HydroType* agy = d.g != 0.0 ? cstone::rawPtr(agy_) : nullptr;
        const HydroType* agz = d.g != 0.0 ? cstone::rawPtr(agz_) : nullptr;
        computePositionsComoving(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)}, agx, agy, agz,
                                 aNow_, aPrevHalf_, aHalf_, aNext_);
        bool haveUnconvergedParticles = updateSmoothingLength(groups_.view(), d);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
        }
        timer.step("UpdateQuantities");
    }

    void saveFields(IFileWriter* writer, size_t first, size_t last, DataType& simData,
                    const cstone::Box<T>& box) override
    {
        auto& d             = simData.hydro;
        auto  fieldPointers = d.data();
        auto  indicesDone   = d.outputFieldIndices;
        auto  namesDone     = d.outputFieldNames;

        auto output = [&]()
        {
            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    std::visit(
                        [writer, c = column, key = namesDone[i]](auto field)
                        {
                            auto&& tmp = cstone::toHost(*field);
                            writeField(writer, key, tmp.data(), c);
                        },
                        fieldPointers[fidx]);
                    indicesDone.erase(indicesDone.begin() + i);
                    namesDone.erase(namesDone.begin() + i);
                }
            }
        };

        // first output pass: write everything allocated at the end of computeForces()
        output();

        // second output pass: write temporary quantities produced by the EOS
        release(d, "c11", "c12", "c13");
        acquire(d, "rho", "p", "gradh");
        computeEOS(first, last, d);
        output();
        release(d, "rho", "p", "gradh");
        acquire(d, "c11", "c12", "c13");

        // third output pass: recover temporary curlv and divv quantities
        release(d, "prho", "c");
        acquire(d, "divv", "curlv");
        // partial recovery of cij in range [first:last] without halos, which are not needed for divv and curlv
        if (!indicesDone.empty()) { computeIadDivvCurlvGradh(groups_.view(), d, box); }
        output();
        release(d, "divv", "curlv");
        acquire(d, "prho", "c");

        /* The following data is now lost and no longer available in the integration step
         *  c11, c12, c12: halos invalidated
         *  prho, c: destroyed
         */

        if (!indicesDone.empty() && Base::rank_ == 0)
        {
            std::cout << "WARNING: the following fields are not in use and therefore not output: ";
            for (std::size_t fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
            {
                std::cout << d.fieldNames[fidx] << ",";
            }
            std::cout << d.fieldNames[indicesDone.back()] << std::endl;
        }
        timer.step("FileOutput");
    }
};

} // namespace sphexa
