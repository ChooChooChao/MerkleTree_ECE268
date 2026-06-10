# MerkleTree (ECE 268)

CPU and GPU implementations of a binary Merkle tree with inclusion proofs, supporting KangarooTwelve (K12) and Poseidon hash functions. Includes CPU correctness tests, GPU vs CPU validation, and build-time benchmarks.

## Repository layout

| Directory | Contents |
|-----------|----------|
| `MerkleTree_ECE268/` | Merkle tree CPU/GPU code, tests, and benchmarks |
| `KangarooTwelve_ECE268/` | K12 hash submodule (CPU + GPU) |
| `ECE268_Spring2026_Poseidon/` | Poseidon hash submodule (CPU Python + GPU CUDA) |

## Dependencies

Clone with submodules from the repository root:

```bash
git submodule update --init --recursive
```

| Submodule | Used for |
|-----------|----------|
| `KangarooTwelve_ECE268/` | CPU and GPU K12 hashing |
| `ECE268_Spring2026_Poseidon/` | GPU Poseidon device primitives and Python CPU reference |

## Merkle tree layout

| Directory | Contents |
|-----------|----------|
| `MerkleTree_ECE268/CPU/` | Reference CPU Merkle tree (`MerkleTree.cpp`), Python prototype (`MerkleTree.py`), and placeholder-hash tests |
| `MerkleTree_ECE268/GPU/` | CUDA implementations (`MerkleTree_GPU.cu` — placeholder hash; `MerkleTree_GPU_v2.cu` — K12 + Poseidon) and unified test/benchmark binary |

Trees store all nodes in a flat byte array (GPU-ready). Each node is 32 bytes. Odd layers are padded by duplicating the last element.

## CPU correctness

CPU tests use a placeholder hash (XOR + bit rotation) and cover tree structure, determinism, proof generation, and proof validation (including tampered and wrong-leaf cases).

From `MerkleTree_ECE268/CPU/`:

```bash
g++ -std=c++17 -O2 -Wall MerkleTree.cpp MerkleTreeTests.cpp -o MerkleTreeTests
./MerkleTreeTests
```

Exits with code 0 when all assertions pass.

A Python reference with the same API is available in `MerkleTree.py`.

## GPU correctness

`MerkleTreeGPUTests_v2` is the main test binary. It compares the GPU tree against:

- **K12** — node-by-node against the CPU K12 C++ library (`KangarooTwelve_ECE268/CPU/`), plus inclusion proof validation
- **Poseidon** — leaf hashes against `ECE268_Spring2026_Poseidon/CPU/main.py` (one UTF-8 line per leaf); internal nodes checked for self-consistency

Requires a CUDA-capable GPU, `nvcc`, and `python3` on `PATH` for Poseidon leaf tests. Tests write leaf strings to `ECE268_Spring2026_Poseidon/input.txt`, then invoke Python from `ECE268_Spring2026_Poseidon/CPU/`.

From `MerkleTree_ECE268/GPU/`:

```bash
make          # build MerkleTreeGPUTests_v2
make run      # build + run tests and benchmarks
```

Or manually:

```bash
nvcc -std=c++14 \
  -I../CPU \
  -I../../KangarooTwelve_ECE268/CPU \
  -I../../KangarooTwelve_ECE268/GPU \
  -I../../ECE268_Spring2026_Poseidon/GPU \
  MerkleTreeGPUTests_v2.cu MerkleTree_GPU_v2.cu \
  ../CPU/MerkleTree.cpp \
  ../../KangarooTwelve_ECE268/CPU/k12.cpp \
  ../../KangarooTwelve_ECE268/CPU/turboshake128.cpp \
  ../../KangarooTwelve_ECE268/CPU/keccak_p.cpp \
  -o MerkleTreeGPUTests_v2
./MerkleTreeGPUTests_v2
```

Prints `Results: N / N tests passed` and exits with code 0 when all tests pass.

**Poseidon notes:**

- Constants are loaded from `../../ECE268_Spring2026_Poseidon/GPU/montgomery_constants.txt` at startup.
- Leaf tests require text inputs (ASCII/UTF-8); binary 1 MB pattern leaves are K12-only.
- If Python is unavailable, Poseidon leaf tests are skipped.

See [ECE268_Spring2026_Poseidon/README.md](ECE268_Spring2026_Poseidon/README.md) for standalone Poseidon usage.

## Timing benchmarks

Benchmarks run automatically at the end of `MerkleTreeGPUTests_v2` (after correctness tests). Each configuration uses 1 warmup run plus 5 timed runs (mean reported).

| Leaves | Leaf size | K12 | Poseidon |
|--------|-----------|-----|----------|
| 64 | 256 B | yes | yes |
| 64 | 1 KB | yes | yes |
| 16 | 1 MB | yes | skipped |
| 64 | 1 MB | yes | skipped |
| 256 | 1 MB | yes | skipped |

- **K12 CPU** — single-threaded C++ `buildMerkleTree()` with `kangaroo_twelve()`
- **K12 GPU** — `buildMerkleTree_GPU_v2(..., HashType::K12)`; includes host↔device memcpy
- **Poseidon CPU** — Python reference (self-reported time); skipped for 1 MB leaves (binary pattern data)
- **Poseidon GPU** — `buildMerkleTree_GPU_v2(..., HashType::POSEIDON)`; includes host↔device memcpy

Speedup is `CPU / GPU` (>1× means GPU is faster).

Large-leaf configs (1 MB) use binary pattern data (`byte[i] = (leaf_index + i) % 251`) so K12 exercises the multi-chunk GPU path (128 threads per leaf).
