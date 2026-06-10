#pragma once

#include "MerkleTree.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

// =============================================================================
// Constants
// =============================================================================
#define GPU_HASH_SIZE 32  // 32 byte hash output (must match CPU HASH_SIZE)

// =============================================================================
// GPU Merkle Tree Structure
// Mirrors the CPU MerkleTree but d_nodes lives in GPU global memory
// =============================================================================
struct MerkleTreeGPU {
    uint8_t*            d_nodes;     // flat node array in GPU global memory
    std::vector<size_t> offsets;     // byte offset of each layer (host-side)
    std::vector<size_t> layerSizes;  // number of nodes per layer (host-side)
    size_t              numLayers;
    size_t              totalBytes;  // total size of d_nodes allocation
};

// =============================================================================
// Device Kernels (defined in MerkleTree_GPU.cu)
// =============================================================================

// Hash all raw leaf data blocks in parallel → leaf layer of d_nodes
__global__ void leafHashKernel(const uint8_t* d_leafData,
                                const size_t*  d_leafOffsets,
                                const size_t*  d_leafLengths,
                                uint8_t*       d_nodes,
                                size_t         leafLayerOffset,
                                size_t         numLeaves);

// Hash all (left || right) pairs in parallel → next layer of d_nodes
__global__ void internalNodeKernel(const uint8_t* d_nodes,
                                    size_t         prevOffset,
                                    size_t         currOffset,
                                    size_t         numParents);

// =============================================================================
// Host Functions
// =============================================================================

// Build the full Merkle tree on the GPU
// Returns a MerkleTreeGPU with d_nodes allocated in GPU global memory
MerkleTreeGPU buildMerkleTree_GPU(const std::vector<std::vector<uint8_t>>& inData);

// Copy the GPU tree back to a CPU MerkleTree struct
// Allows reuse of CPU proofGen and proofValidation functions
MerkleTree copyTreeToHost(const MerkleTreeGPU& gpuTree);

// Free GPU memory allocated for the tree
void freeMerkleTree_GPU(MerkleTreeGPU& tree);
