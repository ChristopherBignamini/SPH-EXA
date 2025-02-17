/*! @file
 * @brief  CPU/GPU Particle subset positions identification function implementations
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <chrono>

#include "sph/particles_data.hpp"
#include "sph/particles_data_gpu.cuh"

#include "particle_selection.hpp"

namespace sphexa
{

__global__ void lowerBoundLoop(uint64_t* dThrustScanResult, uint64_t* dThrustSubsetPos, const uint64_t inputSize, const uint64_t outputSize) {

    const uint64_t i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= outputSize) return;

    auto lower = thrust::lower_bound(thrust::seq, dThrustScanResult, dThrustScanResult + inputSize, dThrustSubsetPos[i]);
    auto lowerIndex = thrust::distance(dThrustScanResult, lower);

    if(lowerIndex < inputSize && *lower == dThrustSubsetPos[i]) {
        dThrustSubsetPos[i] = lowerIndex;
    }
}
    

// TODO: retrieve particle id type from ParticlesData
struct MaskFunctor
{
    __device__
    uint64_t operator()(uint64_t elId) const
    {
        return (elId & msbMask) != 0;
    }
};

struct SearchFunctor
{
    const uint64_t* m_devRawScanResult;
    const uint64_t m_scanResultSize;

//TODO: nested kernels do not work
// i =  thrust::upper_bound(thrust::device, dThrustScanResultPtr, dThrustScanResultPtr + 10, 
//      1, thrust::less<unsigned int>()) - dThrustScanResultPtr;

    __device__
    void operator()(uint64_t& i) const
    {
        for(auto j = i; j<m_scanResultSize; j++){ // TODO: check performance penalty due to break
            if(m_devRawScanResult[j] == i+1){
                i = j;
                break;
            }
        }
    }
};


void findParticlesInSphere_gpu(ParticlesData<cstone::GpuTag>& d, size_t firstIndex, size_t lastIndex, const ParticleSelectionSphere& selSphereData) {

}

void findParticlesInIdList_gpu(ParticlesData<cstone::GpuTag>& d, size_t firstIndex, size_t lastIndex, const std::vector<ParticleIdType>& selParticlesIds) {

}


// TODO: remove, debug only
//#include <inttypes.h>
// struct printf_functor
// {
//   __host__ __device__
//   void operator()(uint64_t x)
//   {
//     // note that using printf in a __device__ function requires
//     // code compiled for a GPU with compute capability 2.0 or
//     // higher (nvcc --arch=sm_20)
//     printf("%" PRIu64 "\n", x);
//   }
// };

void findSelectedParticlesIndexes_gpu(const ParticlesData<cstone::GpuTag>& d, std::vector<uint64_t>& localSelectedParticlesIndexes){

    auto devRawId = d.devData.id.data();
    const auto devIdSize = d.devData.id.size();

    // // TODO: remove, debug only
    // thrust::for_each(thrust::device, devRawId, devRawId+30, printf_functor());

    // Create device containers
    thrust::device_vector<uint64_t> devMask(devIdSize, 0);
    thrust::device_vector<uint64_t> devScanResult(devIdSize, 0);

    // Generate mask
    thrust::transform(devRawId, devRawId + devIdSize, devMask.begin(), MaskFunctor{});

    // Run scan
    thrust::inclusive_scan(devMask.begin(), devMask.end(), devScanResult.begin());

    // Create particle subset position container on GPU and initialize it sequentially
    thrust::device_vector<uint64_t> devSubsetPos(devScanResult.back());
    #if 1
    thrust::sequence(thrust::device, devSubsetPos.begin(), devSubsetPos.end());
    
    // Find the position of the particle in the subset
    // TODO: can I use a zip iterator here instead of raw pointer?
    auto* devRawScanResult = thrust::raw_pointer_cast(devScanResult.data());
    const auto scanResultSize = devScanResult.size();

    auto start = std::chrono::high_resolution_clock::now();
    SearchFunctor searchFunctor{devRawScanResult, scanResultSize};
    thrust::for_each(thrust::device, devSubsetPos.begin(), devSubsetPos.end(), searchFunctor);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << duration.count() << " microseconds" << std::endl;
    #else
    thrust::sequence(thrust::device, devSubsetPos.begin(), devSubsetPos.end(),1);
    
    // Find the position of the particle in the subset
    // TODO: can I use a zip iterator here instead of raw pointer?
    auto* devRawScanResult = thrust::raw_pointer_cast(devScanResult.data());
    const auto scanResultSize = devScanResult.size();

    auto start = std::chrono::high_resolution_clock::now();
    constexpr int numThreads = 256;
    dim3 blockSize(numThreads);
    dim3 gridSize(devSubsetPos.size()/numThreads + 1);
    lowerBoundLoop<<<gridSize, blockSize>>>(devRawScanResult, thrust::raw_pointer_cast(devSubsetPos.data()),
                                            devScanResult.size(), devSubsetPos.size());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << duration.count() << " microseconds" << std::endl;
    #endif



    // Copy result to host
    // TODO: find better solution
    thrust::host_vector<uint64_t> hostSubsetPos(devSubsetPos);
    localSelectedParticlesIndexes.assign(thrust::raw_pointer_cast(hostSubsetPos.data()), thrust::raw_pointer_cast(hostSubsetPos.data()) + hostSubsetPos.size());


    return;

}

}