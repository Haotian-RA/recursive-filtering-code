import math
import numpy as np

# IIR basic function by multi-block filtering in a SIMD manner.

def permute(blocks):
    """Permute samples in multiple blocks in a SIMD manner.
    
    Parameters
    ----------
    blocks : list of ndarray
        A list of N blocks, each of size L (must be power of 2),
        and each block is a 1D array. N should be a multiple of L but no need to be 
        a power of 2.
        
    Returns
    -------
    blocks_bar : list of ndarray
        A list of blocks where samples are permuted (interleaved) in a space of N.
    """
    # Note:
    # 1. (De)permutation takes log2L rounds, every round takes N shuffles.
    # 2. Always swap the bottom left and top right corners among all blocks.
    # 3. Your current-round permutation is based on half-size blocks of the last round.
    # 4. after one-round permutation, you always first generate the new blocks from 
    # the top samples of 2 blocks (with a spacing of N) and the bottom samples of 2 blocks.
    # for example: assume N = 8 blocks
    # start:       [0,1,2,3], [4,5,6,7] ... [16,17,18,19], [20,21,22,23] ...
    # first round: [0,1,16,17], [2,3,18,19], [4,5,20,21], [6,7,22,23] ... [8,9,24,25] ...
    # last round: [0,8,16,24],[1,9,24,25] ... 
    # this function might be easier written due to structure of SIMD blend (blend<>(block1,block2)).
    
    N = len(blocks)
    L = len(blocks[0])
    assert math.log2(L).is_integer(), "L must be a power of 2"
    
    blocks_bar = np.zeros_like(blocks)

    rounds = int(np.log2(L))

    for l in range(rounds):
        stride = L//(2**(l+1))

        for n in range(N//2):

            even = [] # corresponds to even-numbered blocks (the top half)
            odd = [] #  corresponds to even-numbered blocks (the bottom half)

            for i in range(0, L, 2*stride):
                even.append(blocks[n][i:i+stride])
                even.append(blocks[n+N//2][i:i+stride])
                odd.append(blocks[n][i+stride:i+2*stride])
                odd.append(blocks[n+N//2][i+stride:i+2*stride])

            blocks_bar[2*n] = np.concatenate(even)
            blocks_bar[2*n+1] = np.concatenate(odd)

        blocks = blocks_bar.copy()

    return blocks_bar



def depermute(blocks_bar):
    """Depermute samples among blocks back to continuous order in a SIMD manner.

    Parameters
    ----------
    blocks_bar : list of ndarray
        A list of N permuted blocks, each of size L.

    Returns
    -------
    blocks : list of ndarray
        The original blocks before permutation.
    """
    # Note:
    # 1. (De)permutation takes log2L rounds, every round takes N shuffles.
    # 2. Depermutation is not the simple inverse operation of permutation due to retangular form.
    # 3. Always swap the bottom left and top right corners among all blocks.
    # 4. Your current-round permutation is based on half-size blocks of the last round.
    # 5. The first-round permutation permutes within L blocks and re-arranges blocks in N/2 sections, the rest
    # rounds permute blocks in N/4, N/8 ... sections. 
    # for example: assume N = 16 blocks
    # start:       [0,16,32,48], [1,17,33,49] ... [4,20,36,52], [5,21,37,53] ...
    # first round: [0,16,2,18], [4,20,6,22], [8,24,10,26], [12,28,14,30] [1,17,3,19] ... [13,29,15,31] | ... 
    # last round: [0,1,2,3],[4,5,6,7] ... 
    # this function might be easier written due to structure of SIMD blend (blend<>(block1,block2)).
    
    N = len(blocks_bar)
    L = len(blocks_bar[0])
    assert math.log2(L).is_integer(), "L must be a power of 2"

    blocks = np.zeros_like(blocks_bar)

    rounds = int(np.log2(L))

    for l in range(rounds):
        stride = L//(2**(l+1)) 
        
        # first round operation and rest-round operations are inversed.
        if l == 0:
            K = L//2 # seperation of top and bottom
            R = N//L # blocks partition
        else:
            K = 2**l # blocks partition
            R = N//(2**(l+1)) # seperation of top and bottom

        for k in range(K): 
            for n in range(R):

                top = []
                bottom = []
                
                for i in range(0, L, 2*stride):

                    if l == 0:
                        top.append(blocks_bar[k+n*2*K][i:i+stride])  # select top of every L block
                        top.append(blocks_bar[k+K+n*2*K][i:i+stride]) # select top of every L+l/2 block
                        bottom.append(blocks_bar[k+n*2*K][i+stride:i+2*stride]) # select bottom of every L block
                        bottom.append(blocks_bar[k+K+n*2*K][i+stride:i+2*stride]) # select bottom of every L block
                    else:
                        top.append(blocks_bar[n+k*2*R][i:i+stride]) # within each block partition, select top of every N//(2**l) block
                        top.append(blocks_bar[n+R+k*2*R][i:i+stride]) # within each block partition, select top of every N//(2**l)+N//(2**(l+1))
                        bottom.append(blocks_bar[n+k*2*R][i+stride:i+2*stride]) # within each block partition, select bottom of every N//(2**l) block
                        bottom.append(blocks_bar[n+R+k*2*R][i+stride:i+2*stride]) # within each block partition, select bottom of every N//(2**l) block
                        
                if l == 0:
                    blocks[n+k*R] = np.concatenate(top) # put shuffled blocks in order 0,1,2 ...
                    blocks[n+N//2+k*R] = np.concatenate(bottom)
                else:
                    blocks[n+k*2*R] = np.concatenate(top) # don't change the order
                    blocks[n+R+k*2*R] = np.concatenate(bottom)
    
        blocks_bar = blocks.copy()
        
    return blocks


def prepare_X(X_bar,b2,b1,xi2,xi1):
    """Compute the non-recursive part of second order recursive equation.

    Parameters
    ----------
    X_bar : list of ndarray
        A list of N data blocks of x, each of size L. 
        
    b2 : float
        The coefficient for input initial state x_{n-2}.
        
    b1 : float
        The coefficient for input initial state x_{n-1}.
        
    xi2: float
        The second input initial state.
        
    xi1: float
        The first input initial state.

    Returns
    -------
    B_X_bar : list of ndarray
        Input blocks with non-recursive part added.
    """
    
    xvi2 = np.concatenate(([xi2],X_bar[-2][:-1]),dtype=float)
    xvi1 = np.concatenate(([xi1],X_bar[-1][:-1]),dtype=float)
    B_X_bar = np.zeros_like(X_bar,dtype=float)

    for n in range(len(X_bar)):
        if n == 0:
            B_X_bar[0] = X_bar[0] + b1*xvi1 + b2*xvi2
        elif n == 1:
            B_X_bar[1] = X_bar[1] + b1*X_bar[0] + b2*xvi1
        else:
            B_X_bar[n] = X_bar[n] + b1*X_bar[n-1] + b2*X_bar[n-2]

    return B_X_bar