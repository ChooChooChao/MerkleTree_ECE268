#include "MerkleTree_GPU_v2.cuh"

// K12: device-callable, no separate .cu needed
#include "../../KangarooTwelve_ECE268/GPU/k12_device.h"
#include "../../KangarooTwelve_ECE268/GPU/k12_device.cuh"

// Poseidon: shared device primitives (sbox, mix_layer, permutation, hash)
// poseidon_device.cuh includes poseidon.h for uint256 and field arithmetic
#include "../../ECE268_Spring2026_Poseidon/GPU/poseidon_device.cuh"

#include <cmath>
#include <cstring>
#include <cstdio>

// =============================================================================
// Poseidon constant memory — private to this translation unit.
// Call poseidon_load_constants_v2() once before using HashType::POSEIDON.
// =============================================================================
static __device__ __constant__ uint64_t V2_POSEIDON_RC[TOTAL_RC * 4];
static __device__ __constant__ uint64_t V2_POSEIDON_MDS[T * T * 4];

void poseidon_load_constants_v2(const uint64_t* h_rc, size_t rc_count,
                                 const uint64_t* h_mds, size_t mds_count)
{
    cudaError_t err;
    err = cudaMemcpyToSymbol(V2_POSEIDON_RC, h_rc, rc_count * sizeof(uint64_t));
    if (err != cudaSuccess)
        printf("V2 RC load failed: %s\n", cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(V2_POSEIDON_MDS, h_mds, mds_count * sizeof(uint64_t));
    if (err != cudaSuccess)
        printf("V2 MDS load failed: %s\n", cudaGetErrorString(err));
}

// =============================================================================
// Unified leaf kernel — one block per leaf, T threads per block.
//
// K12 branch:    only thread 0 calls k12_device_hash; threads 1-2 idle.
// Poseidon branch: all T threads cooperate via poseidon_hash_device.
//
// Launching with T threads for every hash type lets a single kernel serve
// both. The two idle threads for K12 are the only cost.
// =============================================================================
__global__ void leafKernel_unified(const uint8_t* d_leafData,
                                    const size_t*  d_leafOffsets,
                                    const size_t*  d_leafLengths,
                                    uint8_t*       d_nodes,
                                    size_t         leafLayerOffset,
                                    size_t         numLeaves,
                                    HashType       hashType,
                                    uint8_t*       d_scratch,
                                    size_t         scratch_per_leaf)
{
    size_t i = blockIdx.x;
    if (i >= numLeaves) return;

    const uint8_t* in  = d_leafData + d_leafOffsets[i];
    int            len = static_cast<int>(d_leafLengths[i]);
    uint8_t*       out = d_nodes + leafLayerOffset + i * HASH_SIZE_V2;

    if (hashType == HashType::K12) {
        // Sequential device function — thread 0 only
        if (threadIdx.x == 0) {
            uint8_t* scratch = (scratch_per_leaf > 0)
                               ? d_scratch + i * scratch_per_leaf : nullptr;
            k12_device_hash(in, static_cast<size_t>(len), out, HASH_SIZE_V2, scratch);
        }
    } else {
        // All T threads cooperate; poseidon_hash_device handles sync internally
        poseidon_hash_device(in, len, out, V2_POSEIDON_RC, V2_POSEIDON_MDS);
    }
}

// =============================================================================
// Unified internal-node kernel — one block per parent, T threads per block.
//
// Input is always 64 bytes (left_hash || right_hash), built in shared memory
// by thread 0 so both K12 (thread 0 read) and Poseidon (all threads read)
// can access it after __syncthreads().
//
// Internal nodes are always 64 bytes — well below K12's 8192-byte chunk
// threshold — so no scratch buffer is needed for K12.
// =============================================================================
__global__ void internalNodeKernel_unified(uint8_t* d_nodes,
                                            size_t         prevOffset,
                                            size_t         currOffset,
                                            size_t         numParents,
                                            HashType       hashType)
{
    size_t i = blockIdx.x;
    if (i >= numParents) return;

    const uint8_t* left  = d_nodes + prevOffset + (2 * i)     * HASH_SIZE_V2;
    const uint8_t* right = d_nodes + prevOffset + (2 * i + 1) * HASH_SIZE_V2;

    // Build combined input in shared memory so both hash branches can read it
    __shared__ uint8_t combined[HASH_SIZE_V2 * 2];
    if (threadIdx.x == 0) {
        for (int b = 0; b < (int)HASH_SIZE_V2; b++) {
            combined[b]               = left[b];
            combined[HASH_SIZE_V2 + b] = right[b];
        }
    }
    __syncthreads();

    uint8_t* parent = d_nodes + currOffset + i * HASH_SIZE_V2;

    if (hashType == HashType::K12) {
        if (threadIdx.x == 0)
            k12_device_hash(combined, HASH_SIZE_V2 * 2, parent, HASH_SIZE_V2, nullptr);
    } else {
        poseidon_hash_device(combined, (int)(HASH_SIZE_V2 * 2), parent,
                             V2_POSEIDON_RC, V2_POSEIDON_MDS);
    }
}

// =============================================================================
// buildMerkleTree_GPU_v2
// =============================================================================
MerkleTreeGPU_v2 buildMerkleTree_GPU_v2(
    const std::vector<std::vector<uint8_t>>& inData,
    HashType hashType)
{
    // Pad leaf count to even
    std::vector<std::vector<uint8_t>> leaves = inData;
    if (leaves.size() % 2 != 0)
        leaves.push_back(leaves.back());
    size_t n = leaves.size();

    // Layer sizes and byte offsets
    size_t numLayers =
        static_cast<size_t>(std::ceil(std::log2(static_cast<double>(n)))) + 1;

    std::vector<size_t> layerSizes;
    std::vector<size_t> offsets;
    size_t currentSize = n;
    size_t byteOffset  = 0;
    for (size_t i = 0; i < numLayers; i++) {
        layerSizes.push_back(currentSize);
        offsets.push_back(byteOffset);
        byteOffset  += currentSize * HASH_SIZE_V2;
        currentSize  = (currentSize + 1) / 2;
    }

    // Flat node array on GPU
    uint8_t* d_nodes = nullptr;
    cudaMalloc(&d_nodes, byteOffset);
    cudaMemset(d_nodes, 0, byteOffset);

    // Pack all leaf data and metadata
    std::vector<size_t> leafOffsets(n);
    std::vector<size_t> leafLengths(n);
    size_t maxLeafLen     = 0;
    size_t totalLeafBytes = 0;
    for (size_t i = 0; i < n; i++) {
        leafOffsets[i]  = totalLeafBytes;
        leafLengths[i]  = leaves[i].size();
        totalLeafBytes += leaves[i].size();
        if (leaves[i].size() > maxLeafLen) maxLeafLen = leaves[i].size();
    }

    std::vector<uint8_t> leafDataFlat(totalLeafBytes);
    for (size_t i = 0; i < n; i++)
        memcpy(leafDataFlat.data() + leafOffsets[i],
               leaves[i].data(), leaves[i].size());

    uint8_t* d_leafData    = nullptr;
    size_t*  d_leafOffsets = nullptr;
    size_t*  d_leafLengths = nullptr;
    cudaMalloc(&d_leafData,    totalLeafBytes);
    cudaMalloc(&d_leafOffsets, n * sizeof(size_t));
    cudaMalloc(&d_leafLengths, n * sizeof(size_t));
    cudaMemcpy(d_leafData,    leafDataFlat.data(), totalLeafBytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_leafOffsets, leafOffsets.data(),  n * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_leafLengths, leafLengths.data(),  n * sizeof(size_t),
               cudaMemcpyHostToDevice);

    // K12 scratch — only needed when leaves exceed one chunk (8192 bytes)
    size_t   scratch_per_leaf = k12_device_leaf_scratch_bytes(maxLeafLen, 0);
    uint8_t* d_scratch        = nullptr;
    if (hashType == HashType::K12 && scratch_per_leaf > 0)
        cudaMalloc(&d_scratch, n * scratch_per_leaf);

    // Launch leaf kernel — one block per leaf, T threads per block
    leafKernel_unified<<<n, T>>>(d_leafData, d_leafOffsets, d_leafLengths,
                                  d_nodes, offsets[0], n,
                                  hashType, d_scratch, scratch_per_leaf);
    cudaDeviceSynchronize();

    if (d_scratch) cudaFree(d_scratch);
    cudaFree(d_leafData);
    cudaFree(d_leafOffsets);
    cudaFree(d_leafLengths);

    // Build each internal layer bottom-up
    for (size_t layer = 1; layer < numLayers; layer++) {
        size_t prevSize   = layerSizes[layer - 1];
        size_t paddedSize = (prevSize % 2 == 0) ? prevSize : prevSize + 1;

        // Duplicate last node when the layer is odd-sized
        if (prevSize % 2 != 0) {
            size_t src = offsets[layer - 1] + (prevSize - 1) * HASH_SIZE_V2;
            size_t dst = offsets[layer - 1] +  prevSize      * HASH_SIZE_V2;
            cudaMemcpy(d_nodes + dst, d_nodes + src,
                       HASH_SIZE_V2, cudaMemcpyDeviceToDevice);
        }

        size_t numParents = paddedSize / 2;
        internalNodeKernel_unified<<<numParents, T>>>(
            d_nodes, offsets[layer - 1], offsets[layer], numParents, hashType);
        cudaDeviceSynchronize();
    }

    MerkleTreeGPU_v2 result;
    result.d_nodes    = d_nodes;
    result.offsets    = offsets;
    result.layerSizes = layerSizes;
    result.numLayers  = numLayers;
    result.totalBytes = byteOffset;
    return result;
}

// =============================================================================
// copyTreeToHost_v2
// =============================================================================
MerkleTree copyTreeToHost_v2(const MerkleTreeGPU_v2& gpuTree)
{
    MerkleTree cpu;
    cpu.numLayers  = gpuTree.numLayers;
    cpu.offsets    = gpuTree.offsets;
    cpu.layerSizes = gpuTree.layerSizes;
    cpu.nodes.resize(gpuTree.totalBytes);
    cudaMemcpy(cpu.nodes.data(), gpuTree.d_nodes,
               gpuTree.totalBytes, cudaMemcpyDeviceToHost);
    return cpu;
}

// =============================================================================
// freeMerkleTree_GPU_v2
// =============================================================================
void freeMerkleTree_GPU_v2(MerkleTreeGPU_v2& tree)
{
    if (tree.d_nodes) {
        cudaFree(tree.d_nodes);
        tree.d_nodes = nullptr;
    }
}
