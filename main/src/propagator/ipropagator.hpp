/*
 * MIT License
 *
 * SPH-EXA
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief A common interface for different kinds of propagators
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <variant>

#include "cstone/sfc/box.hpp"
#include "cstone/primitives/accel_switch.hpp"
#include "io/ifile_io.hpp"
#include "sph/particles_data.hpp"
#include "util/pm_reader.hpp"
#include "util/timer.hpp"

namespace sphexa
{

template<class DomainType, class ParticleDataType>
class Propagator
{
    using T = typename ParticleDataType::RealType;

public:

    using ParticleIndexVectorType = ParticlesData<typename ParticleDataType::AcceleratorType>::template FieldVector<uint64_t>;

    Propagator(std::ostream& output, int rank)
        : out(output)
        , timer(output)
        , pmReader(rank)
        , rank_(rank)
    {
    }

    //! @brief get a list of field strings marked as conserved at runtime
    virtual std::vector<std::string> conservedFields() const = 0;

    //! @brief Marks conserved and dependent fields inside the particle dataset as active, enabling memory allocation
    virtual void activateFields(ParticleDataType& d) = 0;

    //! @brief synchronize computational domain
    virtual void sync(DomainType& domain, ParticleDataType& d) = 0;

    //! @brief synchronize domain and compute forces
    virtual void computeForces(DomainType& domain, ParticleDataType& d) = 0;

    //! @brief integrate and/or drift particles in time
    virtual void integrate(DomainType& domain, ParticleDataType& d) = 0;

    //! @brief save particle data fields to file
    virtual void saveFields(IFileWriter*, size_t, size_t, ParticleDataType&, const cstone::Box<T>&){};

    //! @brief save selected particle data fields to file                                                   // TODO: const ParticleDataType::HydroData&
    void saveSelParticlesFields(IFileWriter* writer, size_t first, size_t last, const ParticleIndexVectorType& selectedParticlesPositions, ParticleDataType::HydroData& hydroSimData)
    {
        // TODO: outputSelParticlesAllocatedFields not needed, keeping it as separate function for refactoring reasons
        outputSelParticlesAllocatedFields(writer, first, last, selectedParticlesPositions, hydroSimData);
        timer.step("SelectedParticlesFileOutput");
    }

    //! @brief save extra customizable stuff
    virtual void saveExtra(IFileWriter*, ParticleDataType&){};

    //! @brief save internal state to file
    virtual void save(IFileWriter*){};

    //! @brief load internal state from file
    virtual void load(const std::string& path, IFileReader*){};

    //! @brief whether conserved quantities are time-synchronized (when completing a full time-step hierarchy)
    virtual bool isSynced() { return true; }

    //! @brief add pm counters if they exist
    void addCounters(const std::string& pmRoot, int numRanksPerNode) { pmReader.addCounters(pmRoot, numRanksPerNode); }

    //! @brief print timing info
    void writeMetrics(IFileWriter* writer, const std::string& outFile)
    {
        timer.writeTimings(writer, outFile);
        pmReader.writeTimings(writer, outFile);
    };

    virtual ~Propagator() = default;

    //! @brief Returns time elapsed since the start of last call to computeForces()
    float stepElapsed() const { return timer.sumOfSteps(); }

    void printIterationTimings(const DomainType& domain, const ParticleDataType& simData)
    {
        const auto& d   = simData.hydro;
        const auto& box = domain.box();

        auto nodeCount          = domain.globalTree().numLeafNodes();
        auto particleCount      = domain.nParticles();
        auto haloCount          = domain.nParticlesWithHalos() - domain.nParticles();
        auto totalNeighbors     = d.totalNeighbors;
        auto totalParticleCount = d.numParticlesGlobal;

        out << "### Check ### Global Tree Nodes: " << nodeCount << ", Particles: " << particleCount
            << ", Halos: " << haloCount << std::endl;
        out << "### Check ### Computational domain: " << box.xmin() << " " << box.xmax() << " " << box.ymin() << " "
            << box.ymax() << " " << box.zmin() << " " << box.zmax() << std::endl;
        out << "### Check ### Total Neighbors: " << totalNeighbors
            << ", Avg neighbor count per particle: " << totalNeighbors / totalParticleCount << std::endl;
        out << "### Check ### Total time: " << d.ttot - d.minDt << ", current time-step: " << d.minDt << std::endl;
        out << "### Check ### Total energy: " << d.etot << ", (internal: " << d.eint << ", kinetic: " << d.ecin;
        out << ", gravitational: " << d.egrav;
        out << ")" << std::endl;
        out << "### Check ### Focus Tree Nodes: " << domain.focusTree().octreeViewAcc().numLeafNodes << ", maxDepth "
            << domain.focusTree().depth();
        if constexpr (cstone::HaveGpu<typename ParticleDataType::AcceleratorType>{})
        {
            out << ", maxStackNc " << d.devData.stackUsedNc << ", maxStackGravity " << d.devData.stackUsedGravity;
        }
        out << "\n=== Total time for iteration(" << d.iteration << ") " << timer.sumOfSteps() << "s\n\n";
    }

    //! @brief get the index (position) vector, in local domain, of selected particles
    // TODO: last parameter should be const& but at some point we use the data() method which is non-const
    static auto getSelectedParticleIndexes(const ParticleIndexVectorType& selParticlesIds, ParticleDataType::HydroData& hydroSimData) {

        // TODO: should we run a domain sync before looking to the particles?
        // sync(domain, simData);
        ParticleIndexVectorType localSelectedParticlePositions;

        // Get the column index of the id field
        auto idColumn = hydroSimData.outputFieldIndices[std::find(hydroSimData.outputFieldNames.begin(), hydroSimData.outputFieldNames.end(), "id") - 
                                                        hydroSimData.outputFieldNames.begin()];

        // TODO: use copy_if
        const ParticleIndexVectorType* localParticleIds(std::get<ParticleIndexVectorType*>(hydroSimData.data()[idColumn])); // = std::array<FieldVariant, sizeof...(fields)>{&fields...}
        std::for_each(selParticlesIds.begin(), selParticlesIds.end(), [localParticleIds, &localSelectedParticlePositions](auto selParticleId){
            const auto selParticleIndex = std::find(localParticleIds->begin(), localParticleIds->end(), selParticleId) - localParticleIds->begin();
            if (selParticleIndex < localParticleIds->size()) {
                localSelectedParticlePositions.push_back(selParticleIndex);
            }
        });

        return localSelectedParticlePositions;
    }


protected:
    static void outputAllocatedFields(IFileWriter* writer, size_t first, size_t last, ParticleDataType& simData)
    {
        auto output = [](size_t first, size_t last, auto& d, IFileWriter* writer)
        {
            auto fieldPointers = d.data();
            auto indicesDone   = d.outputFieldIndices;
            auto namesDone     = d.outputFieldNames;

            for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
            {
                int fidx = indicesDone[i];
                if (d.isAllocated(fidx))
                {
                    int column = std::find(d.outputFieldIndices.begin(), d.outputFieldIndices.end(), fidx) -
                                 d.outputFieldIndices.begin();
                    transferToHost(d, first, last, {d.fieldNames[fidx]});
                    std::visit([writer, c = column, key = namesDone[i]](auto field)
                               { writer->writeField(key, field->data(), c); },
                               fieldPointers[fidx]);
                    indicesDone.erase(indicesDone.begin() + i);
                    namesDone.erase(namesDone.begin() + i);
                }
            }

            if (!indicesDone.empty() && writer->rank() == 0)
            {
                std::cout << "WARNING: the following fields are not in use and therefore not output: ";
                for (int fidx = 0; fidx < indicesDone.size() - 1; ++fidx)
                {
                    std::cout << d.fieldNames[fidx] << ",";
                }
                std::cout << d.fieldNames[indicesDone.back()] << std::endl;
            }
        };

        output(first, last, simData.hydro, writer);
        output(first, last, simData.chem, writer);
    }

    // TODO: last parameter should be const& but at some point we use the data() method which is non-const
    static void outputSelParticlesAllocatedFields(IFileWriter* writer, size_t first, size_t last, const ParticleIndexVectorType& selectedParticlesPositions, 
        ParticleDataType::HydroData& hydroSimData)
    {
        auto fieldPointers = hydroSimData.data();
        auto indicesDone   = hydroSimData.outputFieldIndices;
        auto namesDone     = hydroSimData.outputFieldNames;

        // // Locally untag the selected particles to print the right ID
        // // TODO: it can probably be done with templates
        // constexpr uint64_t msbMask = static_cast<uint64_t>(1) << (sizeof(uint64_t)*8 - 1);
        // std::vector<uint64_t> localSelectedParticlesIds = {};
        // std::for_each(selectedParticlesPositions.begin(), selectedParticlesPositions.end(), [&hydroSimData, &localSelectedParticlesIds](auto particlePosition){
        //     localSelectedParticlesIds.push_back(hydroSimData.id[particlePosition] & ~msbMask);
        // });

        // TODO: check existence of code duplication with outputAllocatedFields
        // TODO: here we assume that selected particles is the last field in the dataset
        for (int i = int(indicesDone.size()) - 1; i >= 0; --i)
        {
            int fidx = indicesDone[i];
            if (hydroSimData.isAllocated(fidx))
            {
                int column = std::find(hydroSimData.outputFieldIndices.begin(), hydroSimData.outputFieldIndices.end(), fidx) -
                                       hydroSimData.outputFieldIndices.begin();

                // TODO: the call frequency of this and of the outputAllocatedFields method will be different, 
                // we can save transfer time with a status flag but it will be the current method the one that 
                // will be called more frequently and which will be responsible for the data transfer.
                // TODO: should we just transfer a minimal subset of particles including the selected ones?
                transferToHost(hydroSimData, first, last, {hydroSimData.fieldNames[fidx]});


                // Copy current field data to a new vector only for the selected particles
                std::visit([writer, c = column, key = namesDone[i], &selectedParticlesPositions](auto field){
                    std::remove_pointer_t<decltype(field)> selectedParticleFieldValues;
                    // TODO: use copy_if
                    // std::copy_if(field->begin(), field->end(), std::back_inserter(selectedParticleFieldPointers),
                    //     [](){return true;});
                    std::for_each(selectedParticlesPositions.begin(), selectedParticlesPositions.end(),
                        [&selectedParticleFieldValues, &field](auto particlePosition){
                            selectedParticleFieldValues.push_back(field->at(particlePosition));
                        });
                    writer->writeField(key, selectedParticleFieldValues.data(), c); 
                },
                fieldPointers[fidx]);             

                indicesDone.erase(indicesDone.begin() + i);
                namesDone.erase(namesDone.begin() + i);
            }
        }    
    }

    std::ostream& out;
    Timer         timer;
    PmReader      pmReader;
    int           rank_;
};

} // namespace sphexa
