#include "MerkleTree.h"
#include <iostream>
#include <cassert>

void runTests() {
    auto makeBlock = [](const std::string& s) {
        return std::vector<uint8_t>(s.begin(), s.end());
    };

    // ---- buildMerkleTree tests ----

    // Test 1: power of 2 inputs -> 3 layers
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };
        MerkleTree tree = buildMerkleTree(data, placeholderHash);
        assert(tree.numLayers == 3);
        std::cout << "Test 1 - Number of layers: " << tree.numLayers << std::endl;
    }

    // Test 2: odd number of inputs -> 3 layers
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"), makeBlock("block2")
        };
        MerkleTree tree = buildMerkleTree(data, placeholderHash);
        assert(tree.numLayers == 3);
        std::cout << "Test 2 - Number of layers: " << tree.numLayers << std::endl;
    }

    // Test 3: 6 inputs (intermediate odd layer) -> 4 layers
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"), makeBlock("block2"),
            makeBlock("block3"), makeBlock("block4"), makeBlock("block5")
        };
        MerkleTree tree = buildMerkleTree(data, placeholderHash);
        assert(tree.numLayers == 4);
        std::cout << "Test 3 - Number of layers: " << tree.numLayers << std::endl;
    }

    // Test 4: single input -> 2 layers
    {
        std::vector<std::vector<uint8_t>> data = { makeBlock("block0") };
        MerkleTree tree = buildMerkleTree(data, placeholderHash);
        assert(tree.numLayers == 2);
        std::cout << "Test 4 - Number of layers: " << tree.numLayers << std::endl;
    }

    // Test 5: determinism
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };
        MerkleTree tree1 = buildMerkleTree(data, placeholderHash);
        MerkleTree tree2 = buildMerkleTree(data, placeholderHash);
        bool same = (tree1.nodes == tree2.nodes);
        assert(same);
        std::cout << "Test 5 - Deterministic: " << (same ? "True" : "False") << std::endl;
    }

    // ---- proofGen tests ----

    std::vector<std::vector<uint8_t>> data4 = {
        makeBlock("block0"), makeBlock("block1"),
        makeBlock("block2"), makeBlock("block3")
    };
    MerkleTree tree4 = buildMerkleTree(data4, placeholderHash);

    // Test 1: proof length for 4-leaf tree = 3 (2 siblings + root)
    {
        auto proof = proofGen(data4[0], tree4, placeholderHash);
        assert(proof.size() == 3);
        std::cout << "Test 1 - Proof length: " << proof.size() << std::endl;
    }

    // Test 2: last entry is root and matches tree root
    {
        auto proof       = proofGen(data4[0], tree4, placeholderHash);
        bool isRoot      = (proof.back().second == "root");
        bool matchesTree = (proof.back().first == getNode(tree4, tree4.numLayers - 1, 0));
        assert(isRoot && matchesTree);
        std::cout << "Test 2 - Last entry is root: "  << (isRoot      ? "True" : "False") << std::endl;
        std::cout << "Test 2 - Root matches tree: "   << (matchesTree ? "True" : "False") << std::endl;
    }

    // Test 3: every leaf produces a valid proof length
    {
        for (size_t i = 0; i < data4.size(); i++) {
            auto proof = proofGen(data4[i], tree4, placeholderHash);
            assert(proof.size() == 3);
            std::cout << "Test 3 - Leaf " << i << " proof length: " << proof.size() << std::endl;
        }
    }

    // Test 4: odd leaves
    {
        std::vector<std::vector<uint8_t>> data3 = {
            makeBlock("block0"), makeBlock("block1"), makeBlock("block2")
        };
        MerkleTree tree3 = buildMerkleTree(data3, placeholderHash);
        auto proof = proofGen(data3[0], tree3, placeholderHash);
        assert(proof.size() == 3);
        std::cout << "Test 4 - Odd leaves proof length: " << proof.size() << std::endl;
    }

    // ---- proofValidation tests ----

    // Test 5: valid proof returns true
    {
        auto proof  = proofGen(data4[0], tree4, placeholderHash);
        bool result = proofValidation(data4[0], proof, placeholderHash);
        assert(result);
        std::cout << "Test 5 - Valid proof: " << (result ? "True" : "False") << std::endl;
    }

    // Test 6: every leaf validates
    {
        for (size_t i = 0; i < data4.size(); i++) {
            auto proof  = proofGen(data4[i], tree4, placeholderHash);
            bool result = proofValidation(data4[i], proof, placeholderHash);
            assert(result);
            std::cout << "Test 6 - Leaf " << i << " validates: " << (result ? "True" : "False") << std::endl;
        }
    }

    // Test 7: tampered data returns false
    {
        auto proof  = proofGen(data4[0], tree4, placeholderHash);
        bool result = proofValidation(makeBlock("fakeblock"), proof, placeholderHash);
        assert(!result);
        std::cout << "Test 7 - Tampered data: " << (result ? "True" : "False") << std::endl;
    }

    // Test 8: wrong leaf returns false
    {
        auto proof  = proofGen(data4[0], tree4, placeholderHash);
        bool result = proofValidation(data4[1], proof, placeholderHash);
        assert(!result);
        std::cout << "Test 8 - Wrong leaf: " << (result ? "True" : "False") << std::endl;
    }

    // Test 9: odd leaves validates correctly
    {
        std::vector<std::vector<uint8_t>> data3 = {
            makeBlock("block0"), makeBlock("block1"), makeBlock("block2")
        };
        MerkleTree tree3 = buildMerkleTree(data3, placeholderHash);
        auto proof  = proofGen(data3[2], tree3, placeholderHash);
        bool result = proofValidation(data3[2], proof, placeholderHash);
        assert(result);
        std::cout << "Test 9 - Odd leaves validates: " << (result ? "True" : "False") << std::endl;
    }
}

int main() {
    runTests();
    return 0;
}