# =============================================================================
# Makefile — MerkleTree GPU v2 tests
#
# Run from MerkleTree_ECE268/GPU/ :
#   make          # build
#   make run      # build + run
#   make clean    # remove binary
# =============================================================================

NVCC     := nvcc
STD      := -std=c++14
OUT      := MerkleTreeGPUTests_v2

# Source files
SRCS_CU  := MerkleTreeGPUTests_v2.cu \
             MerkleTree_GPU_v2.cu

SRCS_CPP := ../CPU/MerkleTree.cpp \
             ../../KangarooTwelve_ECE268/CPU/k12.cpp \
             ../../KangarooTwelve_ECE268/CPU/turboshake128.cpp \
             ../../KangarooTwelve_ECE268/CPU/keccak_p.cpp

# Include directories
INCLUDES := -I../CPU \
            -I../../KangarooTwelve_ECE268/CPU \
            -I../../KangarooTwelve_ECE268/GPU \
            -I../../ECE268_Spring2026_Poseidon/GPU

# ---------------------------------------------------------------------------
all: $(OUT)

$(OUT): $(SRCS_CU) $(SRCS_CPP)
	$(NVCC) $(STD) $(INCLUDES) $(SRCS_CU) $(SRCS_CPP) -o $(OUT)
	@echo "Build OK -> $(OUT)"

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT)

.PHONY: all run clean
