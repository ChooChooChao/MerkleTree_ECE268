#include "MerkleTree_GPU.cuh"
#include "MerkleTree.h"
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

// =============================================================================
// GPU Tests
// Each test builds the tree on GPU, copies back to CPU, then compares
// against the CPU reference implementation using the same placeholder hash
// =============================================================================
void runGPUTests()
{
    auto makeBlock = [](const std::string& s) {
        return std::vector<uint8_t>(s.begin(), s.end());
    };

    // ---- buildMerkleTree_GPU tests ----

    // Test 1: power of 2 inputs → 3 layers
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };
        MerkleTreeGPU gpuTree = buildMerkleTree_GPU(data);
        assert(gpuTree.numLayers == 3);
        std::cout << "Test 1 - GPU layers: " << gpuTree.numLayers << std::endl;
        freeMerkleTree_GPU(gpuTree);
    }

    // Test 2: odd number of inputs → 3 layers
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"), makeBlock("block2")
        };
        MerkleTreeGPU gpuTree = buildMerkleTree_GPU(data);
        assert(gpuTree.numLayers == 3);
        std::cout << "Test 2 - GPU layers: " << gpuTree.numLayers << std::endl;
        freeMerkleTree_GPU(gpuTree);
    }

    // Test 3: 6 inputs (intermediate odd layer) → 4 layers
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"), makeBlock("block2"),
            makeBlock("block3"), makeBlock("block4"), makeBlock("block5")
        };
        MerkleTreeGPU gpuTree = buildMerkleTree_GPU(data);
        assert(gpuTree.numLayers == 4);
        std::cout << "Test 3 - GPU layers: " << gpuTree.numLayers << std::endl;
        freeMerkleTree_GPU(gpuTree);
    }

    // Test 4: single input → 2 layers
    {
        std::vector<std::vector<uint8_t>> data = { makeBlock("block0") };
        MerkleTreeGPU gpuTree = buildMerkleTree_GPU(data);
        assert(gpuTree.numLayers == 2);
        std::cout << "Test 4 - GPU layers: " << gpuTree.numLayers << std::endl;
        freeMerkleTree_GPU(gpuTree);
    }

    // Test 5: GPU root matches CPU root (most important test)
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };

        // Build on CPU
        MerkleTree  cpuTree = buildMerkleTree(data, placeholderHash);

        // Build on GPU, copy back
        MerkleTreeGPU gpuTree    = buildMerkleTree_GPU(data);
        MerkleTree    gpuTreeCPU = copyTreeToHost(gpuTree);

        // Compare roots
        std::vector<uint8_t> cpuRoot = getNode(cpuTree,    cpuTree.numLayers    - 1, 0);
        std::vector<uint8_t> gpuRoot = getNode(gpuTreeCPU, gpuTreeCPU.numLayers - 1, 0);
        bool rootsMatch = (cpuRoot == gpuRoot);
        assert(rootsMatch);
        std::cout << "Test 5 - GPU root matches CPU root: "
                  << (rootsMatch ? "True" : "False") << std::endl;

        freeMerkleTree_GPU(gpuTree);
    }

    // Test 6: full tree matches CPU (all nodes, all layers)
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };

        MerkleTree    cpuTree    = buildMerkleTree(data, placeholderHash);
        MerkleTreeGPU gpuTree    = buildMerkleTree_GPU(data);
        MerkleTree    gpuTreeCPU = copyTreeToHost(gpuTree);

        bool nodesMatch = (cpuTree.nodes == gpuTreeCPU.nodes);
        assert(nodesMatch);
        std::cout << "Test 6 - Full GPU tree matches CPU: "
                  << (nodesMatch ? "True" : "False") << std::endl;

        freeMerkleTree_GPU(gpuTree);
    }

    // Test 7: GPU tree → CPU proofGen → proofValidation
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };

        MerkleTreeGPU gpuTree    = buildMerkleTree_GPU(data);
        MerkleTree    gpuTreeCPU = copyTreeToHost(gpuTree);

        // Use CPU proofGen and proofValidation on the GPU-built tree
        for (size_t i = 0; i < data.size(); i++) {
            auto proof  = proofGen(data[i], gpuTreeCPU, placeholderHash);
            bool result = proofValidation(data[i], proof, placeholderHash);
            assert(result);
            std::cout << "Test 7 - GPU tree leaf " << i << " validates: "
                      << (result ? "True" : "False") << std::endl;
        }

        freeMerkleTree_GPU(gpuTree);
    }

    // Test 8: tampered data fails against GPU tree
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };

        MerkleTreeGPU gpuTree    = buildMerkleTree_GPU(data);
        MerkleTree    gpuTreeCPU = copyTreeToHost(gpuTree);

        auto proof  = proofGen(data[0], gpuTreeCPU, placeholderHash);
        bool result = proofValidation(makeBlock("fakeblock"), proof, placeholderHash);
        assert(!result);
        std::cout << "Test 8 - GPU tree tampered data: "
                  << (result ? "True" : "False") << std::endl;

        freeMerkleTree_GPU(gpuTree);
    }

    // Test 9: odd number of leaves, GPU tree validates correctly
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"), makeBlock("block2")
        };

        MerkleTreeGPU gpuTree    = buildMerkleTree_GPU(data);
        MerkleTree    gpuTreeCPU = copyTreeToHost(gpuTree);

        auto proof  = proofGen(data[2], gpuTreeCPU, placeholderHash);
        bool result = proofValidation(data[2], proof, placeholderHash);
        assert(result);
        std::cout << "Test 9 - GPU odd leaves validates: "
                  << (result ? "True" : "False") << std::endl;

        freeMerkleTree_GPU(gpuTree);
    }
}

int main()
{
    // Check CUDA device is available
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    std::cout << "Running GPU Merkle Tree Tests..." << std::endl;
    runGPUTests();
    return 0;
}
