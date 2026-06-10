#pragma once

#include "../CPU/MerkleTree.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// Hash size (32 bytes = 256 bits; matches K12 output and Poseidon uint256)
// ---------------------------------------------------------------------------
#define HASH_SIZE_V2 32u

// ---------------------------------------------------------------------------
// Hash function selector
// ---------------------------------------------------------------------------
enum class HashType { K12, POSEIDON };

// ---------------------------------------------------------------------------
// GPU Merkle tree — mirrors MerkleTreeGPU but works with both hash types.
// d_nodes is a flat array in GPU global memory; layer metadata lives on host.
// ---------------------------------------------------------------------------
struct MerkleTreeGPU_v2 {
    uint8_t*            d_nodes;
    std::vector<size_t> offsets;     // byte offset of layer i in d_nodes
    std::vector<size_t> layerSizes;  // number of nodes in layer i
    size_t              numLayers;
    size_t              totalBytes;
};

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

// Load Poseidon round-constants and MDS matrix into GPU constant memory.
// Must be called once before any buildMerkleTree_GPU_v2 with HashType::POSEIDON.
void poseidon_load_constants_v2(const uint64_t* h_rc, size_t rc_count,
                                 const uint64_t* h_mds, size_t mds_count);

// Build the full Merkle tree on the GPU using one CUDA block per hash.
// hashType selects K12 (1 thread/block) or Poseidon (T=3 threads/block).
MerkleTreeGPU_v2 buildMerkleTree_GPU_v2(
    const std::vector<std::vector<uint8_t>>& inData,
    HashType hashType);

// Copy GPU tree back to a CPU MerkleTree so the existing proofGen /
// proofValidation functions can be reused without modification.
MerkleTree copyTreeToHost_v2(const MerkleTreeGPU_v2& gpuTree);

// Free GPU memory allocated for the tree.
void freeMerkleTree_GPU_v2(MerkleTreeGPU_v2& tree);
