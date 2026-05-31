#include "MerkleTree_GPU.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>

// =============================================================================
// Placeholder hash device function
// XOR + bit rotation — same logic as CPU placeholder
// Replace with k12_gpu device functions later
// =============================================================================
__device__ void placeholderHash_device(const uint8_t* data, size_t len,
                                        uint8_t* result)
{
    for (int i = 0; i < GPU_HASH_SIZE; i++) result[i] = 0;
    for (size_t i = 0; i < len; i++) {
        result[i % GPU_HASH_SIZE]       ^= data[i];
        result[(i + 1) % GPU_HASH_SIZE] ^= (data[i] << 1) & 0xFF;
    }
}

// =============================================================================
// Leaf hashing kernel
// Each thread hashes one raw data block into the leaf layer
// Thread i reads from d_leafData at d_leafOffsets[i] with length d_leafLengths[i]
// Writes GPU_HASH_SIZE bytes into d_nodes at leafLayerOffset + i * GPU_HASH_SIZE
// =============================================================================
__global__ void leafHashKernel(const uint8_t* d_leafData,
                                const size_t*  d_leafOffsets,
                                const size_t*  d_leafLengths,
                                uint8_t*       d_nodes,
                                size_t         leafLayerOffset,
                                size_t         numLeaves)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLeaves) return;

    const uint8_t* block  = d_leafData + d_leafOffsets[i];
    size_t         len    = d_leafLengths[i];
    uint8_t*       output = d_nodes + leafLayerOffset + i * GPU_HASH_SIZE;

    placeholderHash_device(block, len, output);
}

// =============================================================================
// Internal node hashing kernel
// Each thread hashes one (left || right) pair to produce one parent node
// Thread i reads nodes at prevOffset + (2*i) and prevOffset + (2*i+1)
// Writes result to currOffset + i * GPU_HASH_SIZE
// =============================================================================
__global__ void internalNodeKernel(const uint8_t* d_nodes,
                                    size_t         prevOffset,
                                    size_t         currOffset,
                                    size_t         numParents)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParents) return;

    // Read left and right child from previous layer
    const uint8_t* left  = d_nodes + prevOffset + (2 * i)     * GPU_HASH_SIZE;
    const uint8_t* right = d_nodes + prevOffset + (2 * i + 1) * GPU_HASH_SIZE;

    // Concatenate left || right into a local buffer
    uint8_t combined[GPU_HASH_SIZE * 2];
    for (int b = 0; b < GPU_HASH_SIZE; b++) combined[b]                = left[b];
    for (int b = 0; b < GPU_HASH_SIZE; b++) combined[GPU_HASH_SIZE + b] = right[b];

    // Hash combined into parent node
    uint8_t* parent = d_nodes + currOffset + i * GPU_HASH_SIZE;
    placeholderHash_device(combined, GPU_HASH_SIZE * 2, parent);
}

// =============================================================================
// buildMerkleTree_GPU
// Builds the full Merkle tree on the GPU using the flat 1D array layout.
// Returns a MerkleTreeGPU struct with all layer metadata and the GPU node array.
// =============================================================================
MerkleTreeGPU buildMerkleTree_GPU(const std::vector<std::vector<uint8_t>>& inData)
{
    // --- Pad leaf layer to even length ---
    std::vector<std::vector<uint8_t>> leaves = inData;
    if (leaves.size() % 2 != 0) {
        leaves.push_back(leaves.back());
    }
    size_t n = leaves.size();

    // --- Compute layer sizes and byte offsets (same as CPU version) ---
    size_t numLayers = (size_t)std::ceil(std::log2((double)n)) + 1;

    std::vector<size_t> layerSizes;
    std::vector<size_t> offsets;
    size_t currentSize = n;
    size_t byteOffset  = 0;

    for (size_t i = 0; i < numLayers; i++) {
        layerSizes.push_back(currentSize);
        offsets.push_back(byteOffset);
        byteOffset  += currentSize * GPU_HASH_SIZE;
        currentSize  = (currentSize + 1) / 2;
    }

    // --- Allocate flat node array on GPU ---
    uint8_t* d_nodes = nullptr;
    cudaMalloc(&d_nodes, byteOffset);
    cudaMemset(d_nodes, 0, byteOffset);

    // --- Pack all leaf data into a flat host array for upload ---
    std::vector<size_t> leafOffsets(n);
    std::vector<size_t> leafLengths(n);
    size_t totalLeafBytes = 0;
    for (size_t i = 0; i < n; i++) {
        leafOffsets[i]  = totalLeafBytes;
        leafLengths[i]  = leaves[i].size();
        totalLeafBytes += leaves[i].size();
    }

    std::vector<uint8_t> leafDataFlat(totalLeafBytes);
    for (size_t i = 0; i < n; i++) {
        memcpy(leafDataFlat.data() + leafOffsets[i],
               leaves[i].data(), leaves[i].size());
    }

    // --- Upload leaf data to GPU ---
    uint8_t* d_leafData    = nullptr;
    size_t*  d_leafOffsets = nullptr;
    size_t*  d_leafLengths = nullptr;

    cudaMalloc(&d_leafData,    totalLeafBytes);
    cudaMalloc(&d_leafOffsets, n * sizeof(size_t));
    cudaMalloc(&d_leafLengths, n * sizeof(size_t));

    cudaMemcpy(d_leafData,    leafDataFlat.data(),  totalLeafBytes,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_leafOffsets, leafOffsets.data(),   n * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_leafLengths, leafLengths.data(),   n * sizeof(size_t), cudaMemcpyHostToDevice);

    // --- Launch leaf hashing kernel ---
    int threads = 256;
    int blocks  = ((int)n + threads - 1) / threads;
    leafHashKernel<<<blocks, threads>>>(d_leafData, d_leafOffsets, d_leafLengths,
                                        d_nodes, offsets[0], n);
    cudaDeviceSynchronize();

    // Free leaf upload buffers
    cudaFree(d_leafData);
    cudaFree(d_leafOffsets);
    cudaFree(d_leafLengths);

    // --- Build each internal layer ---
    for (size_t layer = 1; layer < numLayers; layer++) {
        size_t prevSize    = layerSizes[layer - 1];
        size_t paddedSize  = (prevSize % 2 == 0) ? prevSize : prevSize + 1;

        // If odd, duplicate last node of previous layer on GPU
        if (prevSize % 2 != 0) {
            size_t lastNodeSrc = offsets[layer - 1] + (prevSize - 1) * GPU_HASH_SIZE;
            size_t lastNodeDst = offsets[layer - 1] + prevSize       * GPU_HASH_SIZE;
            cudaMemcpy(d_nodes + lastNodeDst, d_nodes + lastNodeSrc,
                       GPU_HASH_SIZE, cudaMemcpyDeviceToDevice);
        }

        size_t numParents = paddedSize / 2;
        int    blks       = ((int)numParents + threads - 1) / threads;

        internalNodeKernel<<<blks, threads>>>(d_nodes,
                                              offsets[layer - 1],
                                              offsets[layer],
                                              numParents);
        cudaDeviceSynchronize();
    }

    // --- Fill and return result struct ---
    MerkleTreeGPU result;
    result.d_nodes    = d_nodes;
    result.offsets    = offsets;
    result.layerSizes = layerSizes;
    result.numLayers  = numLayers;
    result.totalBytes = byteOffset;
    return result;
}

// =============================================================================
// copyTreeToHost
// Copies the GPU flat node array back to a CPU MerkleTree struct
// so that proofGen and proofValidation (CPU functions) can be reused as-is
// =============================================================================
MerkleTree copyTreeToHost(const MerkleTreeGPU& gpuTree)
{
    MerkleTree cpuTree;
    cpuTree.numLayers  = gpuTree.numLayers;
    cpuTree.offsets    = gpuTree.offsets;
    cpuTree.layerSizes = gpuTree.layerSizes;
    cpuTree.nodes.resize(gpuTree.totalBytes);

    cudaMemcpy(cpuTree.nodes.data(), gpuTree.d_nodes,
               gpuTree.totalBytes, cudaMemcpyDeviceToHost);

    return cpuTree;
}

// =============================================================================
// freeMerkleTree_GPU
// Frees the GPU memory allocated for the tree
// =============================================================================
void freeMerkleTree_GPU(MerkleTreeGPU& tree)
{
    if (tree.d_nodes) {
        cudaFree(tree.d_nodes);
        tree.d_nodes = nullptr;
    }
}
