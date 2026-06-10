// =============================================================================
// MerkleTreeGPUTests_v2.cu
//
// Tests for MerkleTree_GPU_v2 (unified K12 + Poseidon kernels).
//
// K12 tests:  GPU tree compared node-by-node against the CPU K12 C++ library.
// Poseidon tests: GPU leaf hashes compared against the Python reference via
//                 subprocess; internal nodes verified by self-consistency.
//
// RUN FROM: MerkleTree_ECE268/GPU/
// (All relative paths below assume that working directory.)
// =============================================================================

#include "MerkleTree_GPU_v2.cuh"
#include "../CPU/MerkleTree.h"
#include "../../KangarooTwelve_ECE268/CPU/k12.h"
#include <array>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Paths (relative to MerkleTree_ECE268/GPU/)
// ---------------------------------------------------------------------------
#define POSEIDON_PARAMS_FILE "../../ECE268_Spring2026_Poseidon/GPU/montgomery_constants.txt"
#define POSEIDON_INPUT_FILE  "../../ECE268_Spring2026_Poseidon/input.txt"
// Command to run Python from the CPU folder so "../input.txt" resolves correctly
#ifdef _WIN32
#  define PYTHON_CMD "cd /D ..\\..\\ECE268_Spring2026_Poseidon\\CPU && python main.py"
#else
#  define PYTHON_CMD "cd ../../ECE268_Spring2026_Poseidon/CPU && python3 main.py"
#endif

// ---------------------------------------------------------------------------
// Leaf sizes
// ---------------------------------------------------------------------------
static constexpr size_t LEAF_SIZE_1MB = 1048576u;

// ---------------------------------------------------------------------------
// Tracking
// ---------------------------------------------------------------------------
static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define CHECK(cond, name)                                              \
    do {                                                               \
        ++g_tests_run;                                                 \
        if (cond) {                                                    \
            ++g_tests_passed;                                          \
            std::cout << "  [PASS] " << (name) << "\n";               \
        } else {                                                       \
            std::cout << "  [FAIL] " << (name) << "\n";               \
        }                                                              \
    } while (0)

// ---------------------------------------------------------------------------
// Utility: hex dump of a 32-byte hash
// ---------------------------------------------------------------------------
static void print_hash(const std::string& label, const std::vector<uint8_t>& h)
{
    std::cout << "    " << label << ": ";
    for (uint8_t b : h)
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b;
    std::cout << std::dec << "\n";
}

// ---------------------------------------------------------------------------
// Convert a std::string to a byte vector (for use as leaf data)
// ---------------------------------------------------------------------------
static std::vector<uint8_t> S(const std::string& s)
{
    return std::vector<uint8_t>(s.begin(), s.end());
}

// ===========================================================================
// Poseidon constant loading
// Mirrors load_parameters() from ECE268_Spring2026_Poseidon/GPU/main.cpp
// but calls poseidon_load_constants_v2() instead of load_constants_to_gpu().
// ===========================================================================
static void hex_to_limbs(std::string hex, uint64_t* limbs)
{
    if (hex.size() >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X'))
        hex = hex.substr(2);
    while (hex.size() < 64) hex = "0" + hex;
    for (int i = 0; i < 4; i++)
        limbs[i] = std::stoull(hex.substr(64 - (i + 1) * 16, 16), nullptr, 16);
}

static bool load_poseidon_params_v2(const std::string& filename)
{
    std::vector<uint64_t> rc, mds;
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "  Could not open Poseidon params: " << filename << "\n";
        return false;
    }
    std::string line;
    bool reading_rc = false, reading_mds = false;
    while (std::getline(file, line)) {
        if (line.find("Round constants") != std::string::npos)
            { reading_rc = true;  reading_mds = false; continue; }
        if (line.find("MDS matrix") != std::string::npos)
            { reading_mds = true; reading_rc  = false; continue; }
        size_t pos = 0;
        while ((pos = line.find("'0x", pos)) != std::string::npos) {
            size_t end = line.find("'", pos + 1);
            uint64_t limbs[4];
            hex_to_limbs(line.substr(pos + 1, end - pos - 1), limbs);
            if      (reading_rc)  for (int j = 0; j < 4; j++) rc.push_back(limbs[j]);
            else if (reading_mds) for (int j = 0; j < 4; j++) mds.push_back(limbs[j]);
            pos = end;
        }
    }
    poseidon_load_constants_v2(rc.data(), rc.size(), mds.data(), mds.size());
    return true;
}

// ===========================================================================
// CPU K12 hash wrapper (plugs into buildMerkleTree's HashFunction slot)
// ===========================================================================
static std::vector<uint8_t> k12HashCPU(const std::vector<uint8_t>& data)
{
    std::vector<uint8_t> out(HASH_SIZE);
    kangaroo_twelve(data.data(), data.size(),
                    nullptr, 0,
                    out.data(), HASH_SIZE);
    return out;
}

// ===========================================================================
// Python Poseidon helpers
// ===========================================================================

// Convert a Python hex string like "0x1a2b..." to the same 32-byte layout
// that poseidon_hash_device writes (4 uint64 limbs, each little-endian).
static std::vector<uint8_t> python_hex_to_gpu_bytes(const std::string& hex_str)
{
    std::string hex = hex_str;
    if (hex.size() >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X'))
        hex = hex.substr(2);

    // Left-pad to exactly 64 hex chars (32 bytes)
    while (hex.size() < 64) hex = "0" + hex;

    // Python's hex() is big-endian: hex[0..15] = limb[3] (MSB), ..., hex[48..63] = limb[0] (LSB)
    uint64_t limbs[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++)
        limbs[i] = std::stoull(hex.substr((3 - i) * 16, 16), nullptr, 16);

    // Serialise exactly as poseidon_hash_device does: each limb little-endian
    std::vector<uint8_t> result(32);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            result[i * 8 + j] = static_cast<uint8_t>((limbs[i] >> (j * 8)) & 0xFF);
    return result;
}

// Write lines to the file Python reads, run Python, return one 32-byte hash
// per input line in the same order.
// NOTE: Only valid for text inputs (ASCII/UTF-8).  Binary inputs (e.g. raw
//       hash bytes) cannot be passed through Python's text-mode reader.
static std::vector<std::vector<uint8_t>>
run_python_poseidon(const std::vector<std::string>& inputs)
{
    // Write inputs to input.txt where Python expects it
    {
        std::ofstream f(POSEIDON_INPUT_FILE);
        for (const auto& s : inputs)
            f << s << "\n";
    }

    // Invoke Python script
#ifdef _WIN32
    FILE* pipe = _popen(PYTHON_CMD, "r");
#else
    FILE* pipe = popen(PYTHON_CMD, "r");
#endif
    if (!pipe) {
        std::cerr << "  Could not launch Python. Is python in PATH?\n";
        return {};
    }

    // Parse "Line N Final hash: 0x..." lines
    std::vector<std::pair<int,std::string>> parsed; // (line_idx, hex_str)
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe)) {
        std::string row(buf);
        // Trim trailing newline
        while (!row.empty() && (row.back() == '\n' || row.back() == '\r'))
            row.pop_back();

        // Match "Line N Final hash: 0x..."
        if (row.find("Final hash:") == std::string::npos) continue;
        std::istringstream ss(row);
        std::string word;
        int idx = -1;
        std::string hex;
        // "Line N Final hash: 0x..."
        ss >> word;  // "Line"
        if (word != "Line") continue;
        ss >> idx;   // N
        ss >> word;  // "Final"
        ss >> word;  // "hash:"
        ss >> hex;   // "0x..."
        if (idx >= 0 && !hex.empty())
            parsed.push_back({idx, hex});
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    // Sort by line index and convert
    std::vector<std::vector<uint8_t>> result(inputs.size());
    for (auto& p : parsed) {
        size_t i = static_cast<size_t>(p.first);
        if (i < result.size())
            result[i] = python_hex_to_gpu_bytes(p.second);
    }
    return result;
}

// ===========================================================================
// K12 TESTS
// Compare GPU v2 (HashType::K12) against the CPU K12 C++ library.
// ===========================================================================
static void test_k12()
{
    std::cout << "\n=== K12 Tests ===\n";

    auto makeBlock = [](const std::string& s) { return S(s); };

    // ------------------------------------------------------------------
    // Structure tests: verify layer counts
    // ------------------------------------------------------------------
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };
        MerkleTreeGPU_v2 g = buildMerkleTree_GPU_v2(data, HashType::K12);
        CHECK(g.numLayers == 3, "K12 structure: 4 leaves → 3 layers");
        freeMerkleTree_GPU_v2(g);
    }
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"), makeBlock("block2")
        };
        MerkleTreeGPU_v2 g = buildMerkleTree_GPU_v2(data, HashType::K12);
        CHECK(g.numLayers == 3, "K12 structure: 3 leaves (odd) → 3 layers");
        freeMerkleTree_GPU_v2(g);
    }
    {
        std::vector<std::vector<uint8_t>> data = { makeBlock("block0") };
        MerkleTreeGPU_v2 g = buildMerkleTree_GPU_v2(data, HashType::K12);
        CHECK(g.numLayers == 2, "K12 structure: 1 leaf → 2 layers");
        freeMerkleTree_GPU_v2(g);
    }
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("b0"), makeBlock("b1"), makeBlock("b2"),
            makeBlock("b3"), makeBlock("b4"), makeBlock("b5")
        };
        MerkleTreeGPU_v2 g = buildMerkleTree_GPU_v2(data, HashType::K12);
        CHECK(g.numLayers == 4, "K12 structure: 6 leaves → 4 layers");
        freeMerkleTree_GPU_v2(g);
    }

    // ------------------------------------------------------------------
    // Correctness: GPU root matches CPU root
    // ------------------------------------------------------------------
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };

        MerkleTree    cpuTree  = buildMerkleTree(data, k12HashCPU);
        MerkleTreeGPU_v2 gpuRaw = buildMerkleTree_GPU_v2(data, HashType::K12);
        MerkleTree    gpuTree  = copyTreeToHost_v2(gpuRaw);

        auto cpuRoot = getNode(cpuTree, cpuTree.numLayers - 1, 0);
        auto gpuRoot = getNode(gpuTree, gpuTree.numLayers - 1, 0);
        CHECK(cpuRoot == gpuRoot, "K12 correctness: GPU root matches CPU root");
        if (cpuRoot != gpuRoot) {
            print_hash("CPU root", cpuRoot);
            print_hash("GPU root", gpuRoot);
        }
        freeMerkleTree_GPU_v2(gpuRaw);
    }

    // ------------------------------------------------------------------
    // Correctness: every node matches CPU (all layers)
    // ------------------------------------------------------------------
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("alpha"), makeBlock("beta"),
            makeBlock("gamma"), makeBlock("delta"),
            makeBlock("epsilon"), makeBlock("zeta"),
            makeBlock("eta"),  makeBlock("theta")
        };

        MerkleTree    cpuTree  = buildMerkleTree(data, k12HashCPU);
        MerkleTreeGPU_v2 gpuRaw = buildMerkleTree_GPU_v2(data, HashType::K12);
        MerkleTree    gpuTree  = copyTreeToHost_v2(gpuRaw);

        CHECK(cpuTree.nodes == gpuTree.nodes,
              "K12 correctness: all nodes match CPU (8 leaves)");
        freeMerkleTree_GPU_v2(gpuRaw);
    }

    // ------------------------------------------------------------------
    // Proof generation and validation (using CPU K12 hash)
    // ------------------------------------------------------------------
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };

        MerkleTreeGPU_v2 gpuRaw = buildMerkleTree_GPU_v2(data, HashType::K12);
        MerkleTree       gpuTree = copyTreeToHost_v2(gpuRaw);

        bool all_valid = true;
        for (size_t i = 0; i < data.size(); i++) {
            auto proof  = proofGen(data[i], gpuTree, k12HashCPU);
            bool result = proofValidation(data[i], proof, k12HashCPU);
            if (!result) all_valid = false;
        }
        CHECK(all_valid, "K12 proof: all 4 leaves validate");
        freeMerkleTree_GPU_v2(gpuRaw);
    }

    // ------------------------------------------------------------------
    // Proof validation: tampered data must fail
    // ------------------------------------------------------------------
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };

        MerkleTreeGPU_v2 gpuRaw = buildMerkleTree_GPU_v2(data, HashType::K12);
        MerkleTree       gpuTree = copyTreeToHost_v2(gpuRaw);

        auto proof  = proofGen(data[0], gpuTree, k12HashCPU);
        bool result = proofValidation(S("tampered_data"), proof, k12HashCPU);
        CHECK(!result, "K12 proof: tampered data is rejected");
        freeMerkleTree_GPU_v2(gpuRaw);
    }

    // ------------------------------------------------------------------
    // Odd leaf count: proof validates correctly after padding
    // ------------------------------------------------------------------
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"), makeBlock("block2")
        };

        MerkleTree    cpuTree  = buildMerkleTree(data, k12HashCPU);
        MerkleTreeGPU_v2 gpuRaw = buildMerkleTree_GPU_v2(data, HashType::K12);
        MerkleTree    gpuTree  = copyTreeToHost_v2(gpuRaw);

        auto cpuRoot = getNode(cpuTree, cpuTree.numLayers - 1, 0);
        auto gpuRoot = getNode(gpuTree, gpuTree.numLayers - 1, 0);
        CHECK(cpuRoot == gpuRoot, "K12 odd leaves: GPU root matches CPU root");

        auto proof  = proofGen(data[2], gpuTree, k12HashCPU);
        bool result = proofValidation(data[2], proof, k12HashCPU);
        CHECK(result, "K12 odd leaves: last leaf validates");
        freeMerkleTree_GPU_v2(gpuRaw);
    }

    // ------------------------------------------------------------------
    // Large leaves (1 MB): GPU must use K12 multi-chunk path (128 threads)
    // ------------------------------------------------------------------
    {
        std::cout << "  Building 4 x 1 MB leaves (may take a moment)...\n";
        std::vector<std::vector<uint8_t>> data(4);
        for (size_t i = 0; i < data.size(); i++) {
            data[i].resize(LEAF_SIZE_1MB);
            for (size_t j = 0; j < LEAF_SIZE_1MB; j++)
                data[i][j] = static_cast<uint8_t>((i + j) % 251);
        }

        MerkleTree     cpuTree  = buildMerkleTree(data, k12HashCPU);
        MerkleTreeGPU_v2 gpuRaw = buildMerkleTree_GPU_v2(data, HashType::K12);
        MerkleTree     gpuTree  = copyTreeToHost_v2(gpuRaw);

        CHECK(cpuTree.nodes == gpuTree.nodes,
              "K12 correctness: 4 x 1 MB leaves, all nodes match CPU");
        if (cpuTree.nodes != gpuTree.nodes) {
            auto cpuRoot = getNode(cpuTree, cpuTree.numLayers - 1, 0);
            auto gpuRoot = getNode(gpuTree, gpuTree.numLayers - 1, 0);
            print_hash("CPU root", cpuRoot);
            print_hash("GPU root", gpuRoot);
        }
        freeMerkleTree_GPU_v2(gpuRaw);
    }
}

// ===========================================================================
// POSEIDON TESTS
// Leaf hashes are compared against the Python CPU reference via subprocess.
// Internal nodes are verified by self-consistency (two builds produce the
// same tree).  Full proof validation is skipped because there is no C++
// Poseidon available for the required CPU-side re-hashing.
// ===========================================================================
static void test_poseidon()
{
    std::cout << "\n=== Poseidon Tests ===\n";

    auto makeBlock = [](const std::string& s) { return S(s); };

    // ------------------------------------------------------------------
    // Structure tests
    // ------------------------------------------------------------------
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };
        MerkleTreeGPU_v2 g = buildMerkleTree_GPU_v2(data, HashType::POSEIDON);
        CHECK(g.numLayers == 3, "Poseidon structure: 4 leaves → 3 layers");
        freeMerkleTree_GPU_v2(g);
    }
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"), makeBlock("block2")
        };
        MerkleTreeGPU_v2 g = buildMerkleTree_GPU_v2(data, HashType::POSEIDON);
        CHECK(g.numLayers == 3, "Poseidon structure: 3 leaves (odd) → 3 layers");
        freeMerkleTree_GPU_v2(g);
    }
    {
        std::vector<std::vector<uint8_t>> data = { makeBlock("block0") };
        MerkleTreeGPU_v2 g = buildMerkleTree_GPU_v2(data, HashType::POSEIDON);
        CHECK(g.numLayers == 2, "Poseidon structure: 1 leaf → 2 layers");
        freeMerkleTree_GPU_v2(g);
    }

    // ------------------------------------------------------------------
    // Self-consistency: two builds produce the identical tree
    // ------------------------------------------------------------------
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };

        MerkleTreeGPU_v2 g1 = buildMerkleTree_GPU_v2(data, HashType::POSEIDON);
        MerkleTreeGPU_v2 g2 = buildMerkleTree_GPU_v2(data, HashType::POSEIDON);
        MerkleTree t1 = copyTreeToHost_v2(g1);
        MerkleTree t2 = copyTreeToHost_v2(g2);

        CHECK(t1.nodes == t2.nodes,
              "Poseidon self-consistency: two builds produce identical tree");
        freeMerkleTree_GPU_v2(g1);
        freeMerkleTree_GPU_v2(g2);
    }

    // ------------------------------------------------------------------
    // Leaf correctness: GPU leaf hashes match Python CPU reference
    // ------------------------------------------------------------------
    {
        // These strings are the leaf inputs.  They are written as plain
        // UTF-8 lines to input.txt so Python can read them.
        std::vector<std::string> text_inputs = {
            "block0", "block1", "block2", "block3"
        };

        std::cout << "  Running Python Poseidon reference (may take a few seconds)...\n";
        std::vector<std::vector<uint8_t>> py_hashes =
            run_python_poseidon(text_inputs);

        if (py_hashes.empty() || py_hashes[0].empty()) {
            std::cout << "  [SKIP] Python not available or returned no output.\n";
        } else {
            // Build GPU tree and extract the leaf layer
            std::vector<std::vector<uint8_t>> data;
            for (const auto& s : text_inputs) data.push_back(S(s));

            MerkleTreeGPU_v2 gpuRaw = buildMerkleTree_GPU_v2(data, HashType::POSEIDON);
            MerkleTree       gpuTree = copyTreeToHost_v2(gpuRaw);

            bool all_match = true;
            for (size_t i = 0; i < text_inputs.size(); i++) {
                auto gpu_leaf = getNode(gpuTree, 0, i);  // layer 0 = leaves
                bool match = (!py_hashes[i].empty() && gpu_leaf == py_hashes[i]);
                if (!match) {
                    all_match = false;
                    std::cout << "  Leaf " << i << " MISMATCH:\n";
                    print_hash("  GPU", gpu_leaf);
                    print_hash("  Py ", py_hashes[i]);
                }
            }
            CHECK(all_match,
                  "Poseidon leaf correctness: GPU leaves match Python reference");
            freeMerkleTree_GPU_v2(gpuRaw);
        }
    }

    // ------------------------------------------------------------------
    // Determinism across different input sizes
    // ------------------------------------------------------------------
    {
        // Short input (< 31 bytes: fits in one field element)
        std::vector<std::string> short_inputs = { "hi", "ab", "xy", "zz" };
        std::vector<std::vector<uint8_t>> short_data;
        for (const auto& s : short_inputs) short_data.push_back(S(s));

        MerkleTreeGPU_v2 g1 = buildMerkleTree_GPU_v2(short_data, HashType::POSEIDON);
        MerkleTreeGPU_v2 g2 = buildMerkleTree_GPU_v2(short_data, HashType::POSEIDON);
        MerkleTree t1 = copyTreeToHost_v2(g1);
        MerkleTree t2 = copyTreeToHost_v2(g2);
        CHECK(t1.nodes == t2.nodes,
              "Poseidon determinism: short inputs (<31 bytes)");
        freeMerkleTree_GPU_v2(g1);
        freeMerkleTree_GPU_v2(g2);

        // Long input (> 62 bytes: needs two permutation calls per leaf)
        std::string long_str(80, 'A');
        std::vector<std::vector<uint8_t>> long_data = { S(long_str) };
        MerkleTreeGPU_v2 g3 = buildMerkleTree_GPU_v2(long_data, HashType::POSEIDON);
        MerkleTreeGPU_v2 g4 = buildMerkleTree_GPU_v2(long_data, HashType::POSEIDON);
        MerkleTree t3 = copyTreeToHost_v2(g3);
        MerkleTree t4 = copyTreeToHost_v2(g4);
        CHECK(t3.nodes == t4.nodes,
              "Poseidon determinism: long input (>62 bytes)");
        freeMerkleTree_GPU_v2(g3);
        freeMerkleTree_GPU_v2(g4);
    }

    // ------------------------------------------------------------------
    // Cross-hash: K12 and Poseidon produce DIFFERENT roots for the same data
    // (sanity check that the two code paths are actually distinct)
    // ------------------------------------------------------------------
    {
        std::vector<std::vector<uint8_t>> data = {
            makeBlock("block0"), makeBlock("block1"),
            makeBlock("block2"), makeBlock("block3")
        };

        MerkleTreeGPU_v2 gk12  = buildMerkleTree_GPU_v2(data, HashType::K12);
        MerkleTreeGPU_v2 gpos  = buildMerkleTree_GPU_v2(data, HashType::POSEIDON);
        MerkleTree        tk12  = copyTreeToHost_v2(gk12);
        MerkleTree        tpos  = copyTreeToHost_v2(gpos);

        auto rootK12 = getNode(tk12, tk12.numLayers - 1, 0);
        auto rootPos = getNode(tpos, tpos.numLayers - 1, 0);
        CHECK(rootK12 != rootPos,
              "Sanity: K12 and Poseidon produce different roots");

        freeMerkleTree_GPU_v2(gk12);
        freeMerkleTree_GPU_v2(gpos);
    }

    std::cout << "\n  Note: Poseidon proof generation/validation is skipped.\n"
              << "  Reason: proofValidation() requires a CPU-side hash function,\n"
              << "  and no C++ Poseidon implementation is available in this project.\n"
              << "  Leaf-level correctness is validated against the Python reference above.\n";
}

// ===========================================================================
// BENCHMARKS
//
//  CPU K12      — buildMerkleTree() with kangaroo_twelve() (single-threaded)
//  GPU K12      — buildMerkleTree_GPU_v2(HashType::K12)
//  CPU Poseidon — Python reference; Python's self-reported execution time is
//                 used so the ~3-5 s parameter-import startup is excluded.
//  GPU Poseidon — buildMerkleTree_GPU_v2(HashType::POSEIDON)
//
//  Text inputs are used for every configuration so Python can read them
//  directly from input.txt; the same bytes are fed to C++/CUDA.
//  K12 and GPU timings: 1 warmup + 5 runs, mean reported.
//  CPU Poseidon: single Python invocation, self-reported time parsed.
// ===========================================================================

using hrc = std::chrono::high_resolution_clock;
using ms  = std::chrono::duration<double, std::milli>;

// Generate num_leaves binary blobs each exactly leaf_bytes long.
// Uses a deterministic byte pattern (not ASCII) so large leaves are cheap to build.
static std::vector<std::vector<uint8_t>>
make_pattern_leaves(int num_leaves, size_t leaf_bytes)
{
    std::vector<std::vector<uint8_t>> data(num_leaves);
    for (int i = 0; i < num_leaves; i++) {
        data[i].resize(leaf_bytes);
        for (size_t j = 0; j < leaf_bytes; j++)
            data[i][j] = static_cast<uint8_t>((i + j) % 251);
    }
    return data;
}

// Generate num_leaves ASCII strings each exactly leaf_bytes long.
static std::vector<std::string>
make_text_inputs(int num_leaves, int leaf_bytes)
{
    std::vector<std::string> inputs(num_leaves);
    for (int i = 0; i < num_leaves; i++) {
        std::string s = "leaf" + std::to_string(i);
        while ((int)s.size() < leaf_bytes) s += 'x';
        inputs[i] = s.substr(0, leaf_bytes);
    }
    return inputs;
}

static std::vector<std::vector<uint8_t>>
text_to_bytes(const std::vector<std::string>& inputs)
{
    std::vector<std::vector<uint8_t>> data(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
        data[i] = std::vector<uint8_t>(inputs[i].begin(), inputs[i].end());
    return data;
}

// Write inputs to input.txt, invoke Python, parse and return
// "Total Python Execution Time: X ms".  Returns -1 on failure.
static double run_python_poseidon_timed(const std::vector<std::string>& inputs)
{
    {
        std::ofstream f(POSEIDON_INPUT_FILE);
        for (const auto& s : inputs) f << s << "\n";
    }

#ifdef _WIN32
    FILE* pipe = _popen(PYTHON_CMD, "r");
#else
    FILE* pipe = popen(PYTHON_CMD, "r");
#endif
    if (!pipe) return -1.0;

    double exec_ms = -1.0;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe)) {
        std::string row(buf);
        while (!row.empty() && (row.back() == '\n' || row.back() == '\r'))
            row.pop_back();
        if (row.find("Total Python Execution Time:") != std::string::npos) {
            size_t colon = row.find(':');
            if (colon != std::string::npos) {
                try { exec_ms = std::stod(row.substr(colon + 1)); }
                catch (...) {}
            }
        }
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return exec_ms;
}

// Returns {cpu_k12_ms, gpu_k12_ms, cpu_poseidon_ms, gpu_poseidon_ms}.
static std::array<double, 4>
run_one(const std::vector<std::string>& text_inputs, int warmup, int runs)
{
    auto data = text_to_bytes(text_inputs);
    std::vector<double> cpu_k12, gpu_k12, gpu_pos;

    for (int r = 0; r < warmup + runs; r++) {

        auto t0 = hrc::now();
        { auto t = buildMerkleTree(data, k12HashCPU); (void)t; }
        auto t1 = hrc::now();

        cudaDeviceSynchronize();
        auto t2 = hrc::now();
        { auto g = buildMerkleTree_GPU_v2(data, HashType::K12); freeMerkleTree_GPU_v2(g); }
        auto t3 = hrc::now();

        cudaDeviceSynchronize();
        auto t4 = hrc::now();
        { auto g = buildMerkleTree_GPU_v2(data, HashType::POSEIDON); freeMerkleTree_GPU_v2(g); }
        auto t5 = hrc::now();

        if (r >= warmup) {
            cpu_k12.push_back(ms(t1 - t0).count());
            gpu_k12.push_back(ms(t3 - t2).count());
            gpu_pos.push_back(ms(t5 - t4).count());
        }
    }

    double cpu_pos = run_python_poseidon_timed(text_inputs);

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    return { mean(cpu_k12), mean(gpu_k12), cpu_pos, mean(gpu_pos) };
}

// Format a millisecond value as a fixed-precision string, or "N/A".
static std::string fmt_ms(double v)
{
    if (v < 0) return "N/A";
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << v;
    return o.str();
}

// Format a speedup ratio as "X.XXx", or "N/A".
static std::string fmt_spdup(double cpu, double gpu)
{
    if (cpu <= 0 || gpu <= 0) return "N/A";
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << (cpu / gpu) << "x";
    return o.str();
}

static void benchmark()
{
    std::cout << "\n=== Benchmark: CPU vs GPU Merkle Tree Build Time ===\n";
    std::cout << "  K12      — CPU: single-threaded C++ | GPU: CUDA (1 warmup + 5 runs).\n";
    std::cout << "  Poseidon — CPU: Python self-reported time | GPU: CUDA (1 warmup + 5 runs).\n";
    std::cout << "  1 MB leaf configs use binary pattern data; Poseidon CPU is skipped.\n";
    std::cout << "  GPU times include host<->device memcpy.\n";
    std::cout << "  Speedup = CPU / GPU  (>1x means GPU is faster).\n\n";

    const int W = 12;
    std::cout << std::left
              << std::setw(10) << "Hash"
              << std::setw(8)  << "Leaves"
              << std::setw(9)  << "Leaf(B)"
              << std::setw(W)  << "CPU(ms)"
              << std::setw(W)  << "GPU(ms)"
              << std::setw(10) << "Speedup"
              << "\n";
    std::cout << std::string(59, '-') << "\n";

    const int configs[][2] = {
        {  64,       256 },
        {  64,      1024 },
        {  16,  1048576 },   // 1 MB leaves — K12 multi-chunk path
        {  64,  1048576 },
        { 256,  1048576 },
    };

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    auto print_row = [&](const std::string& hash_name,
                         int leaves, int leaf_b,
                         double cpu, double gpu) {
        std::cout << std::left
                  << std::setw(10) << hash_name
                  << std::setw(8)  << leaves
                  << std::setw(9)  << leaf_b
                  << std::setw(W)  << fmt_ms(cpu)
                  << std::setw(W)  << fmt_ms(gpu)
                  << std::setw(10) << fmt_spdup(cpu, gpu)
                  << "\n";
    };

    for (auto& cfg : configs) {
        int num_leaves = cfg[0], leaf_bytes = cfg[1];
        const bool large_leaves = leaf_bytes >= static_cast<int>(LEAF_SIZE_1MB);

        std::vector<std::vector<uint8_t>> data;
        std::vector<std::string> text_inputs;
        if (large_leaves) {
            std::cout << "  [" << num_leaves << "x" << leaf_bytes
                      << "B — allocating pattern leaves...]\n";
            data = make_pattern_leaves(num_leaves, static_cast<size_t>(leaf_bytes));
        } else {
            text_inputs = make_text_inputs(num_leaves, leaf_bytes);
            data = text_to_bytes(text_inputs);
        }

        // --- K12 ---
        std::vector<double> ck12_s, gk12_s;
        for (int r = 0; r < 6; r++) {   // run 0 = warmup, discarded
            auto t0 = hrc::now();
            { auto t = buildMerkleTree(data, k12HashCPU); (void)t; }
            auto t1 = hrc::now();

            cudaDeviceSynchronize();
            auto t2 = hrc::now();
            { auto g = buildMerkleTree_GPU_v2(data, HashType::K12); freeMerkleTree_GPU_v2(g); }
            auto t3 = hrc::now();

            if (r > 0) {
                ck12_s.push_back(ms(t1 - t0).count());
                gk12_s.push_back(ms(t3 - t2).count());
            }
        }

        print_row("K12", num_leaves, leaf_bytes, mean(ck12_s), mean(gk12_s));

        if (large_leaves) {
            std::cout << "  (Poseidon CPU/GPU skipped for 1 MB leaves)\n\n";
            continue;
        }

        // --- GPU Poseidon ---
        std::vector<double> gpos_s;
        for (int r = 0; r < 6; r++) {
            cudaDeviceSynchronize();
            auto t4 = hrc::now();
            { auto g = buildMerkleTree_GPU_v2(data, HashType::POSEIDON); freeMerkleTree_GPU_v2(g); }
            auto t5 = hrc::now();
            if (r > 0) gpos_s.push_back(ms(t5 - t4).count());
        }

        // --- CPU Poseidon (Python, one run) ---
        std::cout << "  [" << num_leaves << "x" << leaf_bytes << "B — invoking Python...]\n";
        double cpos = run_python_poseidon_timed(text_inputs);

        print_row("Poseidon", num_leaves, leaf_bytes, cpos, mean(gpos_s));
        std::cout << "\n";
    }
}

// ===========================================================================
// main
// ===========================================================================
int main()
{
    // Check that a CUDA device is present
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";

    // Load Poseidon constants before any Poseidon test runs
    std::cout << "Loading Poseidon constants from " << POSEIDON_PARAMS_FILE << "...\n";
    if (!load_poseidon_params_v2(POSEIDON_PARAMS_FILE)) {
        std::cerr << "Failed to load Poseidon constants. "
                  << "Poseidon tests may produce wrong results.\n";
    }

    // Run correctness tests
    test_k12();
    test_poseidon();

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "Results: " << g_tests_passed << " / " << g_tests_run
              << " tests passed.\n";
    std::cout << "========================================\n";

    // Run benchmarks (always, regardless of test pass/fail)
    benchmark();

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
