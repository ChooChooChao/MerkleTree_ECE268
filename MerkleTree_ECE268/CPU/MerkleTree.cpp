#include "MerkleTree.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cassert>


// =============================================================================
// Placeholder Hash (XOR + bit rotation, same logic as Python version)
// Replace with KangarooTwelve or Poseidon later
// =============================================================================
std::vector<uint8_t> placeholderHash(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> result(HASH_SIZE, 0);
    for (size_t i = 0; i < data.size(); i++) {
        result[i % HASH_SIZE]       ^= data[i];
        result[(i + 1) % HASH_SIZE] ^= (data[i] << 1) & 0xFF;
    }
    return result;
}


// Helper: read one node from the flat array
std::vector<uint8_t> getNode(const MerkleTree& tree, size_t layer, size_t idx) {
    size_t start = tree.offsets[layer] + idx * HASH_SIZE;
    return std::vector<uint8_t>(
        tree.nodes.begin() + start,
        tree.nodes.begin() + start + HASH_SIZE
    );
}

// Helper: write one node into the flat array
void setNode(MerkleTree& tree, size_t layer, size_t idx,
             const std::vector<uint8_t>& hash) {
    size_t start = tree.offsets[layer] + idx * HASH_SIZE;
    std::copy(hash.begin(), hash.end(), tree.nodes.begin() + start);
}

// =============================================================================
// buildMerkleTree
// inData    : raw data blocks (each block is a byte vector)
// hashFunction : hash function to use
// Returns a MerkleTree with all layers stored in a flat array
// =============================================================================
MerkleTree buildMerkleTree(const std::vector<std::vector<uint8_t>>& inData,
                           HashFunction hashFunction) {
    // Pad leaf layer to even length by duplicating last element
    std::vector<std::vector<uint8_t>> leaves = inData;
    if (leaves.size() % 2 != 0) {
        leaves.push_back(leaves.back());
    }

    // Hash all leaves
    std::vector<std::vector<uint8_t>> hashedLeaves;
    for (const auto& leaf : leaves) {
        hashedLeaves.push_back(hashFunction(leaf));
    }

    // Calculate number of layers: ceil(log2(n)) + 1
    size_t n         = hashedLeaves.size();
    size_t numLayers = (size_t)std::ceil(std::log2((double)n)) + 1;

    // Pre-compute layer sizes and byte offsets for flat array allocation
    std::vector<size_t> layerSizes;
    std::vector<size_t> offsets;
    size_t currentSize = n;
    size_t byteOffset  = 0;

    for (size_t i = 0; i < numLayers; i++) {
        layerSizes.push_back(currentSize);
        offsets.push_back(byteOffset);
        byteOffset  += currentSize * HASH_SIZE;
        currentSize  = (currentSize + 1) / 2;  // ceil(currentSize / 2)
    }

    // Allocate flat array and fill in struct
    MerkleTree tree;
    tree.numLayers  = numLayers;
    tree.layerSizes = layerSizes;
    tree.offsets    = offsets;
    tree.nodes.resize(byteOffset, 0);

    // Store leaf layer (layer 0)
    for (size_t i = 0; i < hashedLeaves.size(); i++) {
        setNode(tree, 0, i, hashedLeaves[i]);
    }

    // Build each subsequent layer from the previous one
    for (size_t layer = 1; layer < numLayers; layer++) {
        // Read previous layer into a temp vector, padding if odd
        std::vector<std::vector<uint8_t>> prevLayer;
        for (size_t i = 0; i < tree.layerSizes[layer - 1]; i++) {
            prevLayer.push_back(getNode(tree, layer - 1, i));
        }
        if (prevLayer.size() % 2 != 0) {
            prevLayer.push_back(prevLayer.back());
        }

        // Hash each pair of nodes to produce the parent
        for (size_t i = 0; i < prevLayer.size(); i += 2) {
            std::vector<uint8_t> combined;
            combined.insert(combined.end(), prevLayer[i].begin(),     prevLayer[i].end());
            combined.insert(combined.end(), prevLayer[i+1].begin(),   prevLayer[i+1].end());
            setNode(tree, layer, i / 2, hashFunction(combined));
        }
    }

    return tree;
}

// =============================================================================
// proofGen
// inData       : the raw data block you want to prove membership of
// tree         : the Merkle tree built from buildMerkleTree
// hashFunction : same hash function used to build the tree
// Returns a list of (sibling hash, direction) pairs ending with the root
// =============================================================================
std::vector<ProofEntry> proofGen(const std::vector<uint8_t>& inData,
                                 const MerkleTree& tree,
                                 HashFunction hashFunction) {
    std::vector<uint8_t>  currNode = hashFunction(inData);
    std::vector<ProofEntry> proof;

    for (size_t layer = 0; layer < tree.numLayers - 1; layer++) {
        // Find index of currNode in this layer
        size_t idx   = 0;
        bool   found = false;
        for (size_t i = 0; i < tree.layerSizes[layer]; i++) {
            if (getNode(tree, layer, i) == currNode) {
                idx   = i;
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("Node not found in layer");

        std::vector<uint8_t> sibling;
        std::vector<uint8_t> combined;

        if (idx % 2 == 0) {
            // Even index: sibling is to the right
            sibling = getNode(tree, layer, idx + 1);
            proof.push_back({sibling, "right"});
            combined.insert(combined.end(), currNode.begin(), currNode.end());
            combined.insert(combined.end(), sibling.begin(),  sibling.end());
        } else {
            // Odd index: sibling is to the left
            sibling = getNode(tree, layer, idx - 1);
            proof.push_back({sibling, "left"});
            combined.insert(combined.end(), sibling.begin(),  sibling.end());
            combined.insert(combined.end(), currNode.begin(), currNode.end());
        }
        currNode = hashFunction(combined);
    }

    // Append root as final entry
    proof.push_back({getNode(tree, tree.numLayers - 1, 0), "root"});
    return proof;
}

// =============================================================================
// proofValidation
// inData       : the raw data block to verify
// proof        : proof list generated by proofGen
// hashFunction : same hash function used to build the tree
// Returns true if the leaf hashes up to the root correctly
// =============================================================================
bool proofValidation(const std::vector<uint8_t>& inData,
                     const std::vector<ProofEntry>& proof,
                     HashFunction hashFunction) {
    std::vector<uint8_t> root      = proof.back().first;
    std::vector<uint8_t> hashedVal = hashFunction(inData);

    for (size_t i = 0; i < proof.size() - 1; i++) {
        const std::vector<uint8_t>& sibling   = proof[i].first;
        const std::string&          direction = proof[i].second;

        std::vector<uint8_t> combined;
        if (direction == "left") {
            combined.insert(combined.end(), sibling.begin(),   sibling.end());
            combined.insert(combined.end(), hashedVal.begin(), hashedVal.end());
        } else {
            combined.insert(combined.end(), hashedVal.begin(), hashedVal.end());
            combined.insert(combined.end(), sibling.begin(),   sibling.end());
        }
        hashedVal = hashFunction(combined);
    }

    return hashedVal == root;
}