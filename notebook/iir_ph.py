import numpy as np
from iir_function import permute,depermute,prepare_X 

# compute IIR recursive function by multiple blocks using particular and homogeneous factorization.

def block_particular_solution(B_X_bar,a2,a1):
    """Compute the particular solution among blocks.

    Parameters
    ----------
    B_X_bar : list of ndarray
        A list of N data blocks of x after feedforwarding, each of size L. 
        
    a2 : float
        The coefficient for input initial state y_{n-2}.
        
    a1 : float
        The coefficient for input initial state y_{n-1}.

    Returns
    -------
    W_bar : list of ndarray
        Input blocks with particular part added.
    """
    # note: particular solution is computed based on blocks not samples. 
    # The particular solution of the elements of the first block, i.e., W_bar[0], except for the first one,
    # is different from the one in sample base. For example, W_bar[0][1] = x[N] is not the real particular solution
    # x[N] - a1*w[N-1] - a2*w[N-2] in sample base, where w contains the particular solution of prior inputs.
    
    W_bar = np.zeros_like(B_X_bar,dtype=float)

    for n in range(len(B_X_bar)):
        
        if n == 0:
            W_bar[0] = B_X_bar[0]
        elif n == 1:
            W_bar[1] = B_X_bar[1] - a1*W_bar[0]
        else:
            W_bar[n] = B_X_bar[n] - a1*W_bar[n-1] - a2*W_bar[n-2]

    return W_bar


def iir_h(N,a2,a1):
    """Compute the impulse response of second order recursive equation.

    Parameters
    ----------
    N : int
        The length of the (truncated) impulse response.
        
    a2 : float
        The coefficient for input initial state y_{n-2}.
        
    a1 : float
        The coefficient for input initial state y_{n-1}.

    Returns
    -------
    h2 : N x 1 ndarray
        IIR impulse response for the second output initial state, i.e., yi2.
        
    h1 : N x 1 ndarray
        IIR impulse response for the first output initial state, i.e., yi1.
    """
    
    h1 = np.zeros(N,dtype=float)
    h2 = np.zeros(N,dtype=float)
    
    for n in range(N):
        
        if n == 0:
            h2[0] = a2
            h1[0] = a1
        elif n == 1:
            h2[1] = -a1*h2[0]
            h1[1] = -a1*h1[0] + h2[0]
        else:
            h2[n] = -a1*h2[n-1] - a2*h2[n-2]
            h1[n] = -a1*h1[n-1] - a2*h1[n-2]
                
    return (h2,h1)



def C_rd(L,h2=0,h1=0,C=None,c_cross=False):
    """Compute the required multiplier for block recursive doubling.

    Parameters
    ----------
    L  : int
        The block size. L is a power of 2, which caters to the SIMD length.
    
    h2 : N x 1 ndarray
            IIR impulse response for the second output initial state, i.e., yi2.
        
    h1 : N x 1 ndarray
        IIR impulse response for the first output initial state, i.e., yi1.

    C  : 2 x 2 ndarray
        Block recursive parameter for Y_n and Y_{n-1}, where Y_n is a 2 x 1 ndarray 
        that includes the two output initial states. For example, for N = 4, 
        [y_2 y_3]^T = [x_2 x_3]^T + C*[y_{-2} y_{-1}]^T, and next one is 
        [y_6 y_7]^T = [x_6 x_7]^T + C*[y_2 y_3]^T.

    Returns
    -------
    C_dict : dictionary of ndarray
        A dictionary of the block recursive doubling multiplier, which has the following directory:
                                   key                                           value
        first layer:   l (the number of recursive round)               second layer dictionary.
        second layer: '-2' or '-1'(C matrix for xv2 or xv1)              third layer ndarray
        third layer:   0 or 1 (elements in C added to xv2 or xv1)             L x 1 ndarray
        
    C_list[-1] : 2 x 2 ndarray
        Block recursive parameter for cross core computation.
    """     
    
    # multiplier for the last samples, which are also the initial states for next block.
    if C is None:
        C = np.array([[h2[-2],h1[-2]],[h2[-1],h1[-1]]]) 
    
    C_list = []
    
    for l in range(L): # block recursive doubling is processed with one block size. 
        if l == 0:
            C_list.append(C)
        else:
            C_list.append(-C @ C_list[l-1]) # C, C^2, C^3 ... C^L
            
    C_dict = {}

    rounds = int(np.log2(L))
    
    for l in range(rounds+1):
        
        K = 2**(l-1)
        repeats = L//(2*K)
        
        C_dict[l] = {}
        C_dict[l]['-2'] = []
        C_dict[l]['-1'] = []
        
        for m in range(2): # for second order

            # for appending initial output state 
            if l == 0: 
                C_dict[0]['-2'].append(np.array([C[0][m] for C in C_list]))
                C_dict[0]['-1'].append(np.array([C[1][m] for C in C_list])) 
            # for appending recursive part that only involves inputs
            else:
                tmp2 = [0]*K + [C[0][m] for C in C_list[:K]]
                tmp1 = [0]*K + [C[1][m] for C in C_list[:K]]
                C_dict[l]['-2'].append(tmp2*repeats)
                C_dict[l]['-1'].append(tmp1*repeats)

    if c_cross:
        return C_list[-1]
    else:
        return C_dict



class block_homogeneous_solution:
    """Compute the homogeneous solution among blocks.
    
    Parameters
    ----------
    N  : int
        The number of blocks. N is a multiple of L.

    L  : int
        The block size. L is a power of 2, which caters to the SIMD length.
    
    a2 : float
        The coefficient for input initial state y_{n-2}.
        
    a1 : float
        The coefficient for input initial state y_{n-1}.

    yi2: float
        The second output initial state.
    
    yi1: float
        The first output initial state.

    W_bar : list of ndarray
        A list of input permuted blocks.

    Returns
    -------
    h2 : N x 1 ndarray
        IIR impulse response for the second output initial state, i.e., yi2.
        
    h1 : N x 1 ndarray
        IIR impulse response for the first output initial state, i.e., yi1.

    C_dict : dictionary of ndarray
        A dictionary of the block recursive doubling multiplier, which has the following directory:
                                   key                                           value
        first layer:   l (the number of recursive round)               second layer dictionary.
        second layer: '-2' or '-1'(C matrix for xv2 or xv1)              third layer ndarray
        third layer:   0 or 1 (elements in C added to xv2 or xv1)             L x 1 ndarray
        
    Y_bar : list of ndarray
        A list of completed output permuted blocks.
    """

    def __init__(self,N,L,a2,a1,yi2,yi1):
        
        self.a2 = a2     
        self.a1 = a1
        self.yi2 = yi2
        self.yi1 = yi1
        
        self.h2,self.h1 = iir_h(N,a2,a1)
        self.C_dict = C_rd(L,self.h2,self.h1)


    def block_recursive_doubling(self,C_dict,xv2,xv1,yi2=0,yi1=0,pre_rd=True,add_init=True):
        """
        Perform recursive doubling between samples inside a SIMD vector.
        
        Parameters
        ----------
        C_dict : a dictionary of ndarray
            The pre-computed block recursive doubling multiplier for SOS.
                                    key                                           value
            first layer:   l (the number of recursive round)               second layer dictionary.
            second layer: '-2' or '-1'(C matrix for xv2 or xv1)              third layer ndarray
            third layer:   0 or 1 (elements in C added to xv2 or xv1)             L x 1 ndarray
    
        xv2 : L x 1 ndarray
            The second input block where every sample has a recursive relationship.
    
        xv1 : L x 1 ndarray
            The first input block where every sample has a recursive relationship.
    
        yi2 : float
            The second output initial state.
            
        yi1 : float
            The first output initial state.
    
        Returns
        -------
        yv2 : L x 1 ndarray
            The second output block after adding the recursive solution from the initial state.
        
        yv1 : L x 1 ndarray
            The first output block after adding the recursive solution from the initial state.
        """
        # block recursive doubling is mainly to be used in computing the recursive section in IIR.
        # In second order system, once the computation of input (x) is done, the relationship between
        # output (Y_n) and initial state (Y_{n-1}) can be represented by
        # Y_n = X_n - CY_{n-1}, where C is a 2 by 2 matrix, and Y_n is a 2 by 1 vector that contains 
        # the initial states for next Y_{n+1}.
        # Without block recursive doubling, the number of computation takes M^2*L = 4L number of operations,
        # sequential traversing L samples in a block. While with block recursive doubling, the number of computation
        # takes M^2*log2(L) FMAs + M*log2(L) shuffles.
        # The difficulty is that the samples in two blocks, i.e., xv2 and xv1, have the recursive property and we want 
        # to use SIMD manner and don't break up the two blocks. 
        # THE SOLUTION IS TO SHUFFLE THE REQUIRED BLOCKS FOR BLOCK COMPUTATION FOR EVERY ROUND.
        # Note: this function is seperated into two parts:
        # 1. pre-rd: compute recursive part that only involves x.
        # 2. add_init: the last step to add the initial on pre-processed x.
    
        L = len(xv2)
    
        rounds = int(np.log2(L))
        
        # compute recursive part that only involves x
        if pre_rd:
            for l in range(1,rounds+1): 
                
                K = 2**(l-1)
                repeats = L//(2*K)
                tmp2 = []
                tmp1 = []
                
                for r in range(repeats):
    
                    tmp2 += [0]*K + [xv2[K-1+2*K*r]]*K  # for the first round, tmp2 = [0 x_0 0 x_2 ...]. This can be done simpler in C++ SIMD.
                    tmp1 += [0]*K + [xv1[K-1+2*K*r]]*K
    
                xv2 -= (C_dict[l]['-2'][0]*np.array(tmp2) + C_dict[l]['-2'][1]*np.array(tmp1))
                xv1 -= (C_dict[l]['-1'][0]*np.array(tmp2) + C_dict[l]['-1'][1]*np.array(tmp1))
    
        # the last step to add the initial on pre-processed x
        if add_init:
            yv2 = xv2 - C_dict[0]['-2'][0]*yi2 - C_dict[0]['-2'][1]*yi1
            yv1 = xv1 - C_dict[0]['-1'][0]*yi2 - C_dict[0]['-1'][1]*yi1
    
            return (yv2,yv1)
        else:
            return (xv2,xv1)


    def forward(self,h2,h1,W_bar,yv2,yv1,yi2,yi1):
        """Compute all output blocks by filtering with initial blocks. 
    
        Parameters
        ----------
        h2 : N x 1 ndarray
            IIR impulse response for the second output initial state, i.e., yi2.
            
        h1 : N x 1 ndarray
            IIR impulse response for the first output initial state, i.e., yi1.
    
        W : list of ndarray
            A list of ndarray of particular solution.
            
        yv2 : L x 1 ndarray
            The completed second last block, i.e., Y[n-2], computed by block recursive doubling.
    
        yv1 : L x 1 ndarray
            The completed second last block, i.e., Y[n-1], computed by block recursive doubling.
            
        yi2 : float
            The second output initial state.
            
        yi1 : float
            The first output initial state.
    
        Returns
        -------
        Y_bar : list of ndarray
            The completed output blocks.
        """

        # prepare for the two initial blocks
        yvi2 = np.concatenate(([yi2],yv2[:-1]))
        yvi1 = np.concatenate(([yi1],yv1[:-1]))
    
        Y_bar = np.zeros_like(W_bar)
        Y_bar[-2] = yv2
        Y_bar[-1] = yv1
        
        for n in range(len(Y_bar)-2):
            Y_bar[n] = W_bar[n] - h2[n]*yvi2 - h1[n]*yvi1
            
        return Y_bar
    
    
    def compute(self,W_bar):

        yv2,yv1 = self.block_recursive_doubling(self.C_dict,W_bar[-2],W_bar[-1],self.yi2,self.yi1)
        Y_bar = self.forward(self.h2,self.h1,W_bar,yv2,yv1,self.yi2,self.yi1)

        return Y_bar



class cross_core_block_homogeneous_solution(block_homogeneous_solution):
    """Compute the homogeneous solution of multiple blocks in multiple cores simultaneously.
    
    Parameters
    ----------
    N  : int
        The number of blocks. N is a multiple of L.

    L  : int
        The block size. L is a power of 2, which caters to the SIMD length.
    
    a2 : float
        The coefficient for input initial state y_{n-2}.
        
    a1 : float
        The coefficient for input initial state y_{n-1}.

    yi2: float
        The second output initial state.
    
    yi1: float
        The first output initial state.

    Returns
    -------
    C_cross : 2 x 2 ndarray
        Block recursive parameter for cross core computation.

    C_cross_dict : dictionary of ndarray
        A dictionary of the block recursive doubling multiplier for multi-core processing, which has the following directory:
                                   key                                           value
        first layer:   l (the number of recursive round)               second layer dictionary.
        second layer: '-2' or '-1'(C_cross matrix for xv2 or xv1)         third layer ndarray
        third layer:   0 or 1 (elements in C added to xv2 or xv1)             L x 1 ndarray

    Y_bar_list : list of list of ndarray
        A list of "multiple completed output blocks" computed by each core.
    """    

    
    def __init__(self,N,L,a2,a1,yi2,yi1):
        super().__init__(N,L,a2,a1,yi2,yi1)
        
        C_cross = C_rd(L,self.h2,self.h1,c_cross=True)
        self.C_dict_cross = C_rd(L=L,C=C_cross)

    def compute(self,W_list):

        L = len(W_list)

        wv2_cross = np.zeros(L,dtype=float)
        wv1_cross = np.zeros(L,dtype=float)
               
        W_last_list = []

        # step 1: compute recursive part that only involves W in each core
        for l in range(L):
            
            wv2,wv1 = super().block_recursive_doubling(self.C_dict,W_list[l][-2],W_list[l][-1],add_init=False)

            wv2_cross[l] = wv2[-1] 
            wv1_cross[l] = wv1[-1]
            
            W_last_list.append((wv2,wv1))

        # step 2: cross-core recursive doubling (with initial appending) to get the initial states for all cores.
        yv2_cross,yv1_cross = super().block_recursive_doubling(self.C_dict_cross,wv2_cross,wv1_cross,self.yi2,self.yi1)        
        
        Y_bar_list = []

        # step 3: inner-core recursive doubling to get all output blocks
        for l in range(L):
            
            if l == 0:
                yi2 = self.yi2
                yi1 = self.yi1
            else:
                yi2 = yv2_cross[l-1]
                yi1 = yv1_cross[l-1]

            # inner-core block filtering (with initial appending) to get the last two blocks 
            yv2,yv1 = super().block_recursive_doubling(self.C_dict,W_last_list[l][0],W_last_list[l][1],yi2,yi1,pre_rd=False)
            
            # forward the rest blocks
            Y_bar_list.append(super().forward(self.h2,self.h1,W_list[l],yv2,yv1,yi2,yi1))

        return Y_bar_list