/*
 * SPH-EXA
 *
 * Copyright (c) 2026 CSCS, ETH Zurich, University of Zurich, University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief A Propagator class for modern SPH with generalized volume elements in comoving coordinates
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include <filesystem>
#include <memory>

#include "cstone/fields/field_get.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"
#include "sph/positions_comoving.hpp"

#include "io/arg_parser.hpp"
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
     * The factors are defined as: aPrevHalf_ = a(t_n - dt_m1/2), aHalf_ = a(t_n + dt/2), aNext_ = a(t_n+1).
     * aNow_ = a(t_n) is set in computeForces.
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

    //! @brief set by load() when a snapshot provides a scale factor, cleared by the first consistency check
    bool restoredFromSnapshot_{false};

    /*! @brief Propagator status parameters
     *
     * scaleFactor is needed to make a snapshot physically meaningful: the dataset holds comoving positions
     * and peculiar velocities, which cannot be interpreted without the a they belong to.
     *
     * aHalf is integrator state needed to correctly restart a simulation: updateScaleFactors carries aPrevHalf
     * over from the previous step aHalf so the two are identical, which is what makes the canonical momentum
     * conserve exactly. Saving and then loading it lets a restarted run continue that chain instead of
     * restarting it from a differently rounded expression for the same instant.
     *
     * TODO: to be removed once the documentation is in place.
     * This is a temporary measure to document the expected behavior of the snapshot format.
     * A snapshot stores every field in whatever convention the code holds it in, plus the scale factor,
     * and leaves the conversion to the external users. Physical quantities can be obtained by multiplying
     * the stored ones by the appropriate power of the scale factor reported in the second column.
     *
     *   x, y, z            a                  comoving position R, physical r = a*R
     *   x_m1, y_m1, z_m1   a                  comoving position increment of the previous step
     *   vx, vy, vz         1                  peculiar velocity v = a*dR/dt; the physical one adds H*r
     *   h                  a                  smoothing length, comoving like the coordinates
     *   m                  1                  mass
     *   u                  a^(-3*(gamma-1))   comoving internal energy u_c
     *   rho                a^-3               comoving density rho_c = a^3 * rho_phys
     *   p, prho            a^(-3*gamma)       comoving pressure p_c = a^(3*gamma) * p_phys
     *   c                  1                  already physical, see scaleSoundSpeed
     *   ax, ay, az         a^-1               dP/dt = a * f_phys, see sph::combineAccelerationsComoving
     *   ugrav              a^-1               potential summed over comoving separations
     *   c11 ... c33        a^-2               inverse second moment of the IAD operator
     *   divv, curlv        a^-1               comoving gradients of the peculiar velocity, so this is the
     *                                         peculiar part alone: the physical divergence adds 3*H
     *   dV11 ... dV33      a^-1               velocity gradient components, as divv
     *   xm                 a^3                comoving volume element
     *   kx, gradh          1                  dimensionless
     *   alpha, nc, id      1                  dimensionless or counters
     *   dtCourant          a                  pending the time-step correction, see the Courant notes
     *   du, du_m1          n/a                rate of the comoving u, not a fixed power of a:
     *                                         du_c/dt = a^(3*(gamma-1)) * (du/dt + 3*H*(gamma-1)*u)
     *   temp, cv, mui      n/a                not allocated here, this propagator evolves u
     *   mue, tdpdTrho      n/a                not allocated here
     *   keys               n/a                SFC keys of the comoving coordinates, no physical counterpart
     */
    struct Params
    {
        //! @brief the scale factor at the current time
        double scaleFactor{1};

        //! @brief the scale factor at the midpoint of the current step
        double aHalf{1};

        template<class Archive>
        void loadOrStoreAttributes(Archive* ar)
        {
            ar->stepAttribute("cosmo::scaleFactor", &scaleFactor, 1);
            ar->stepAttribute("cosmo::aHalf", &aHalf, 1);
        }
    };
    Params params_;

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
     * This function uses aNow_, which computeForces sets from d.ttot before calling this. computeTimestep has not yet
     * run at that point, so it is the correct scale factor of the current state.
     *
     * By including the expansion factor in the description, a Hubble dragging term given by -3*H*(gamma-1)*u
     * is introduced in the energy equation. That term is a cooling contribution purely due to the expansion itself and
     * can be "removed" by switchig to the comoving internal energy u_c = u* a^(3*(gamma-1)). This means that the internal
     * energy of the HydroData structure is assumed to be the comoving one.
     */

    void scaleInternalEnergyRate(DomainType& domain, typename DataType::HydroData& d, size_t first, size_t last)
    {
        // exact for the static universe, and a no-op at the present epoch of a real one
        if (aNow_ == T(1)) { return; }

        auto* du = cstone::rawPtr(d.du);
        cstone::scale(domain.exec(), du + first, du + last, du + first, T(1) / aNow_);
    }

    /*! @brief Rescale the sound speed produced by computeEOS into the physical one
     *
     * The equation of state builds c from the stored internal energy, which is the comoving u_c, so it returns
     * sqrt(gamma*(gamma-1)*u_c) = a^(3*(gamma-1)/2) * c_phys. This has to be taken into account since the sound
     * speed is used to compute the signal velocity in MomentumAndEnergyInteraction and is involved in the calculation
     * of artificial_viscosity as well, where c is mixed with peculiar velocities times comoving separations with no
     * implicit cancelation of the scaling factor so we have to do that explicitly.
     *
     * It must be noted that this function must be called before the halo exchange of "c": scaling only [first, last)
     * and exchanging afterwards would leave halo particles with the comoving value while local ones carry the physical
     * one, which shows up only at rank boundaries and only for some decompositions.
     *
     * TODO: This leaves the dataset deliberately split, d.prho stays comoving, d.c becomes physical, so the two are no
     * longer related by c^2 = gamma*p/rho in stored variables. Nothing recomputes one from the other today, but
     * anything that starts to must account for it.
     */
    void scaleSoundSpeed(typename DataType::HydroData& d, size_t first, size_t last)
    {
        // exact for the static universe, and a no-op at the present epoch of a real one
        if (aNow_ == T(1)) { return; }

        auto* c = cstone::rawPtr(d.c);
        cstone::scale(d.exec, c + first, c + last, c + first, std::pow(aNow_, T(-1.5) * (d.gamma - T(1))));
    }

    /*! @brief Correct the Courant time step for the comoving smoothing length
     *
     * sph::tsKCourant returns Kcour * h / v, with v the signal velocity. Every other input is already a physical
     * velocity: c after scaleSoundSpeed, and w_ij because the comoving separation cancels between the dot
     * product and the distance it is divided by. The smoothing length is not, it is comoving like the
     * coordinates it is used with, so the returned step is short of the physical one by exactly one factor of a:
     *
     *     dt_phys = Kcour * h_phys / v = Kcour * a * h_com / v = a * tsKCourant(Kcour, h_com, v)
     *
     * Applied to the reduced scalar rather than to the per-particle d.dtCourant, because both backends already
     * reduce into d.minDtCourant (see sph::computeMomentumEnergy) and nothing else reads the per-particle field.
     * No guard on a == 1 is needed: multiplying a double by exactly 1.0 is the identity.
     *
     * TODO: the signal velocity is still underestimated because the peculiar approach speed w_ij is used instead
     * of the physical one, which adds a dragging term H*r_ij with r_ij the physical distance.
     */
    void scaleCourantTimestep(typename DataType::HydroData& d) { d.minDtCourant *= aNow_; }

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

    /*! @brief Combine the hydrodynamic and gravitational accelerations
     *
     * This function is responsible for combining the hydrodynamic and gravitational accelerations into the
     * ax/ay/az fields of the HydroData according to the comoving description of the system evolution, namely with
     * correct scale-factor dependencies.
     *
     * Gravity: the tree solver sums G*m_j*(x_j - x_i)/|x_j - x_i|^3 over the *comoving* separations in d.x, so
     * agx/agy/agz hold g_c = a^2 * g_phys. Hence dP/dt|grav = a * g_phys = g_c / a, i.e. wGrav = 1/a.
     *
     * Hydro: the kernel sums produce the comoving density rho_c = a^3 * rho_phys, and d.u holds the comoving
     * specific internal energy u_c = a^(3*(gamma-1)) * u_phys, so the equation of state yields
     * p_stored = (gamma-1) * rho_c * u_c = a^(3*gamma) * p_phys. Tracking that through the volume elements and the
     * kernel gradient, d.ax comes out as a^(3*gamma-2) * f_phys, so reaching a * f_phys takes
     * wHydro = a^(3-3*gamma) = a^(-3*(gamma-1)).
     *
     * After this call d.ax/ay/az hold dP/dt = a * f_phys, the canonical momentum rate. Both weights are
     * dimensionless, so this still has the dimensions of an acceleration, but it is not the comoving acceleration
     * d2X/dt2: the two differ by a^2 and the Hubble drag,
     *
     *     dP/dt = a^2 * (d2X/dt2 + 2*H*dX/dt).
     *
     */
    void combineAccelerations(typename DataType::HydroData& d, size_t first, size_t last)
    {
        T wHydro = std::pow(aNow_, T(-3) * (d.gamma - T(1)));
        T wGrav  = T(1) / aNow_;

        const HydroType* agx = d.g != 0.0 ? cstone::rawPtr(agx_) : nullptr;
        const HydroType* agy = d.g != 0.0 ? cstone::rawPtr(agy_) : nullptr;
        const HydroType* agz = d.g != 0.0 ? cstone::rawPtr(agz_) : nullptr;

        combineAccelerationsComoving(first, last, d, agx, agy, agz, wHydro, wGrav);
    }

    /*! @brief Check if the reconstructed a(t) agrees with the one the snapshot was written at
     */
    void checkRestartScaleFactor()
    {
        restoredFromSnapshot_ = false;

        double stored = params_.scaleFactor;
        if (std::abs(aNow_ - stored) <= 1e-9 * std::abs(stored)) { return; }

        throw std::runtime_error("Restart inconsistency: the snapshot was written at scale factor " +
                                 std::to_string(stored) + " but the cosmology reconstructs " +
                                 std::to_string(double(aNow_)) +
                                 " at the restored time. Check that the cosmology settings match the run being "
                                 "continued\n");
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

        // d.ttot is still the time the particles are at: computeTimestep, which advances it, runs in integrate().
        // This is therefore the same t_n that updateScaleFactors later derives as d.ttot - d.minDt, so aNow_ stays
        // bitwise consistent with aPrevHalf_/aHalf_/aNext_.
        aNow_ = cosmo_->a(d.ttot);
        if (restoredFromSnapshot_) { checkRestartScaleFactor(); }

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
        //! @brief must precede the halo exchange of "c" below, so that halos carry the physical sound speed too
        scaleSoundSpeed(d, first, last);
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
        scaleCourantTimestep(d);
        timer.step("MomentumAndEnergy");
        pmReader.step();

        if (d.g != 0.0)
        {
            //! gravitational acceleration is accumulated into a dedicated set, separate from the hydro ax/ay/az
            reallocate(domain.nParticlesWithHalos(), d.getAllocGrowthRate(), agx_, agy_, agz_);
            //TODO: it is possible that we can avoid the use of separate arrays for the gravitational acceleration 
            // and directly accumulate it into the hydro acceleration with the right weight.
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

        //! @brief from here on d.ax/ay/az hold dP/dt = a * f_phys, not the hydrodynamic acceleration alone
        combineAccelerations(d, first, last);
        timer.step("CombineAccelerations");
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        // d.ax/ay/az already contains the weighted hydro+grav accelerations at this point,
        // so calculation of the timestep is consistent with the acceleration used in the
        // integration step
        computeTimestep(first, last, d);
        updateScaleFactors(d.ttot, d.minDt, d.minDt_m1);
        timer.step("Timestep");
        computePositionsComoving(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)}, aPrevHalf_, aHalf_,
                                 aNext_);
        bool haveUnconvergedParticles = updateSmoothingLength(groups_.view(), d);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
        }
        timer.step("UpdateQuantities");
    }

    /*! @brief Update the scale factor belonging to the particle data being written and
     *         the restarting scale factor at the half-step.
     */
    void save(IFileWriter* writer) override
    {
        params_.scaleFactor = aNow_;
        params_.aHalf       = aHalf_;
        params_.loadOrStoreAttributes(writer);
    }

    /*! @brief Restore the scale factor for simulation restarting
     */
    void load(const std::string& initCond, IFileReader* reader) override
    {
        const std::string path = removeModifiers(initCond);
        if (std::filesystem::exists(path))
        {
            int snapshotIndex = numberAfterSign(initCond, ":");
            reader->setStep(path, snapshotIndex, FileMode::independent);
            if (reader->stepAttributeSize("cosmo::aHalf") > 0) {
                params_.loadOrStoreAttributes(reader);
                aHalf_                = params_.aHalf;
                firstStep_            = false;
                restoredFromSnapshot_ = true;
            }
            reader->closeStep();
        }
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
        //! @brief "c" is not in the default output set, but if it is requested it should match the run's value
        scaleSoundSpeed(d, first, last);
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
