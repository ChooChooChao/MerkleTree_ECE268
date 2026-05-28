#pragma once

#include <vector>
#include <functional>
#include <string>
#include <cstdint>

// =============================================================================
// Constants
// =============================================================================
const size_t HASH_SIZE = 32;  // 32 byte hash output

// =============================================================================
// Type Aliases
// =============================================================================
using HashFunction = std::function<std::vector<uint8_t>(const std::vector<uint8_t>&)>;
using ProofEntry   = std::pair<std::vector<uint8_t>, std::string>;

// =============================================================================
// Merkle Tree Structure
// Stores all nodes in a flat 1D array (GPU-ready)
// Each layer's starting byte offset is stored in offsets[]
// Each node is exactly HASH_SIZE bytes
//
// To access node j in layer i:
//   nodes[ offsets[i] + j * HASH_SIZE ]  ...  [ offsets[i] + (j+1) * HASH_SIZE ]
// =============================================================================
struct MerkleTree {
    std::vector<uint8_t> nodes;      // flat array of all hash values
    std::vector<size_t>  offsets;    // byte offset of each layer in nodes[]
    std::vector<size_t>  layerSizes; // number of nodes in each layer
    size_t               numLayers;
};

// =============================================================================
// Function Declarations
// =============================================================================

// Placeholder hash using XOR + bit rotation
// Replace with KangarooTwelve or Poseidon later
std::vector<uint8_t> placeholderHash(const std::vector<uint8_t>& data);

// Read one node from the flat array
std::vector<uint8_t> getNode(const MerkleTree& tree, size_t layer, size_t idx);

// Write one node into the flat array
void setNode(MerkleTree& tree, size_t layer, size_t idx,
             const std::vector<uint8_t>& hash);

// Build a Merkle tree from raw data blocks
MerkleTree buildMerkleTree(const std::vector<std::vector<uint8_t>>& inData,
                           HashFunction hashFunction);

// Generate an inclusion proof for a given data block
std::vector<ProofEntry> proofGen(const std::vector<uint8_t>& inData,
                                 const MerkleTree& tree,
                                 HashFunction hashFunction);

// Verify an inclusion proof against the root
bool proofValidation(const std::vector<uint8_t>& inData,
                     const std::vector<ProofEntry>& proof,
                     HashFunction hashFunction);