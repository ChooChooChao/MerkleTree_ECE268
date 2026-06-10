import math

def placeholder_hash(data: bytes) -> bytes:
    """
    Simple placeholder hash using XOR + bit rotation.
    Replace this with KangarooTwelve or Poseidon later.
    """
    result = bytearray(32)  # 32 byte output, same size real hashes will use
    
    for i, byte in enumerate(data):
        result[i % 32] ^= byte
        result[(i + 1) % 32] ^= (byte << 1) & 0xFF
    
    return bytes(result)

def buildMerkleTree(inData, hashFunction):
    """
    Given an input list of data values, build the merkle tree using the specified hash function passed.

    inData: list of values that will be used to build merkle tree
    hashFunction: the function that you want to use for hashing in the tree
    """
    tree = []
    leaves = []
    
    # if odd number of leaves duplicate last value
    if len(inData) % 2 != 0:
        leaves = inData + [inData[-1]]
    else:
        leaves = inData

    # calculate number of layers in the tree
    numLayers = math.ceil(math.log2(len(leaves))) + 1
    
    # create hash of each leaf
    hashedLeaves = [hashFunction(i) for i in leaves]
    tree.append(hashedLeaves)
    
    count = 1
    while count != numLayers:
        layer = []
        prevLayer = []

        # check intermediate layers if an odd number of leaves and duplicate as necessary
        if len(tree[count - 1]) % 2 != 0:
            prevLayer = tree[count - 1] + [tree[count - 1][-1]]
        else:
            prevLayer = tree[count - 1]
                                           
        # traverse the previous layers' leaves and conduct the hashing on the pairs
        for i in range(0, len(prevLayer), 2):
            layer.append(hashFunction(prevLayer[i] + prevLayer[i+1]))

        count += 1
        tree.append(layer) # append the layer to the total tree

    return tree

def proofGen(inData, merkleTree, hashFunction):
    """
    This fucntion generates the proof of the merkle tree, meaning it returns a list
    of the required (sibling node, direction) to reach the root hash from the input data.

    inData: string value that you want to start at
    merkleTree: the merkle tree list generated from buildMerkleTree
    hashFunction: the hash function that you used to build the merkle tree
    """
    startLeaf = hashFunction(inData)

    proof = []

    currNode = startLeaf
    for i in range(len(merkleTree) - 1):
        idx = merkleTree[i].index(currNode)

        # if even idx, sibling is to the right (left otherwise)
        if (idx % 2 == 0):
            proof.append((merkleTree[i][idx+1], "right"))
            currNode = hashFunction(currNode + merkleTree[i][idx+1])
        else:
            proof.append((merkleTree[i][idx-1], "left"))
            currNode = hashFunction(merkleTree[i][idx-1] + currNode)

    proof.append((merkleTree[-1][0], "root"))

    return proof


def proofValidation(inData, proofList, hashFunction):
    """
    This function takes a starting value and traverses the generated proof to check
    if you can reach the root hash for verification.

    inData: a string value that you want to verify
    proofList: generated proof list
    hashFunction: the hash function that you used to build the merkle tree
    """
    start = hashFunction(inData)

    root = proofList[-1][0]

    hashedVal = start
    for node in proofList[0:len(proofList)-1]:
        if (node[1] == "left"):
            hashedVal = hashFunction(node[0] + hashedVal)
        else:
            hashedVal = hashFunction(hashedVal + node[0])

    return (hashedVal == root)



## BUILD MERKLETREE TESTS
# ------------------------------------------------------------------------------
# Test 1 - Power of 2 inputs (clean case)
data = [b'block0', b'block1', b'block2', b'block3']
tree = buildMerkleTree(data, placeholder_hash)
print("Test 1 - Number of layers:", len(tree))  # expected: 3
print("Root:", tree[-1])                          # expected: single hash

# Test 2 - Odd number of inputs (tests padding)
data = [b'block0', b'block1', b'block2']
tree = buildMerkleTree(data, placeholder_hash)
print("Test 2 - Number of layers:", len(tree))  # expected: 3
print("Root:", tree[-1])                          # expected: single hash

# Test 3 - 6 inputs (tests intermediate odd layer)
data = [b'block0', b'block1', b'block2', b'block3', b'block4', b'block5']
tree = buildMerkleTree(data, placeholder_hash)
print("Test 3 - Number of layers:", len(tree))  # expected: 4
print("Root:", tree[-1])                          # expected: single hash

# Test 4 - Single input edge case
data = [b'block0']
tree = buildMerkleTree(data, placeholder_hash)
print("Test 4 - Number of layers:", len(tree))  # expected: 2
print("Root:", tree[-1])                          # expected: single hash

# Test 5 - Determinism check (same input should always give same root)
data = [b'block0', b'block1', b'block2', b'block3']
tree1 = buildMerkleTree(data, placeholder_hash)
tree2 = buildMerkleTree(data, placeholder_hash)
print("Test 5 - Deterministic:", tree1[-1] == tree2[-1])  # expected: True


## PROOF GEN TESTS
# ------------------------------------------------------------------------------
# Test 1 - Basic proof generation, check proof length
# For n=4 leaves, proof should have 2 siblings + 1 root = 3 entries
data = [b'block0', b'block1', b'block2', b'block3']
tree = buildMerkleTree(data, placeholder_hash)
proof = proofGen(data[0], tree, placeholder_hash)
print("Test 1 - Proof length:", len(proof))  # expected: 3

# Test 2 - Last entry should always be the root
proof = proofGen(data[0], tree, placeholder_hash)
print("Test 2 - Last entry is root:", proof[-1][1] == "root")  # expected: True
print("Test 2 - Root matches tree:", proof[-1][0] == tree[-1][0])  # expected: True

# Test 3 - Check proof for every leaf, all should generate valid proofs
data = [b'block0', b'block1', b'block2', b'block3']
tree = buildMerkleTree(data, placeholder_hash)
for i, block in enumerate(data):
    proof = proofGen(block, tree, placeholder_hash)
    print(f"Test 3 - Leaf {i} proof length:", len(proof))  # expected: 3 for all

# Test 4 - Odd number of leaves
data = [b'block0', b'block1', b'block2']
tree = buildMerkleTree(data, placeholder_hash)
proof = proofGen(data[0], tree, placeholder_hash)
print("Test 4 - Odd leaves proof length:", len(proof))  # expected: 3


# PROOF VALIDAION TESTS
# ------------------------------------------------------------------------------
# Test 5 - Valid proof should return True
data = [b'block0', b'block1', b'block2', b'block3']
tree = buildMerkleTree(data, placeholder_hash)
proof = proofGen(data[0], tree, placeholder_hash)
print("Test 5 - Valid proof:", proofValidation(data[0], proof, placeholder_hash))  # expected: True

# Test 6 - Every leaf should validate successfully
for i, block in enumerate(data):
    proof = proofGen(block, tree, placeholder_hash)
    result = proofValidation(block, proof, placeholder_hash)
    print(f"Test 6 - Leaf {i} validates:", result)  # expected: True for all

# Test 7 - Tampered data should return False
proof = proofGen(data[0], tree, placeholder_hash)
print("Test 7 - Tampered data:", proofValidation(b'fakeblock', proof, placeholder_hash))  # expected: False

# Test 8 - Proof from one leaf should not validate another leaf
proof_leaf0 = proofGen(data[0], tree, placeholder_hash)
print("Test 8 - Wrong leaf:", proofValidation(data[1], proof_leaf0, placeholder_hash))  # expected: False

# Test 9 - Odd number of leaves validates correctly
data = [b'block0', b'block1', b'block2']
tree = buildMerkleTree(data, placeholder_hash)
proof = proofGen(data[2], tree, placeholder_hash)
print("Test 9 - Odd leaves validates:", proofValidation(data[2], proof, placeholder_hash))  # expected: True
                             
                             

    