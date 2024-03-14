"""
Music 422 Marina Bosi

- mdct.py -- Computes a reasonably fast MDCT/IMDCT using the FFT/IFFT

-----------------------------------------------------------------------
© 2009-2024 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
from scipy.fft import fft, ifft
import time

### Problem 1.a ###
def MDCTslow(data, a, b, isInverse=False):
    """
    Slow MDCT algorithm for window length a+b following pp. 130 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    and where the 2/N factor is included in the forward transform instead of inverse.
    a: left half-window length
    b: right half-window length
    """
    n_0 = (b + 1)/2
    N = a + b

    if isInverse is False:
        X = []
        # don't need to subtract 1 because it is N/2 (exclusive)
        for k in range(0,int(N/2)):
            sum = np.sum(data * np.cos((2*np.pi/N)*(np.arange(0, N, 1) + n_0)*(k + 0.5)))
            X.append(2/N * sum)
        
        return X
    elif isInverse is True:
        x = []
        for n in range(0,N):
            sum = np.sum(data * np.cos((2*np.pi/N)*(n + n_0)*(np.arange(0, int(N/2), 1) + 0.5)))
            x.append(2 * sum)
        return x

### Problem 1.c ###
# Fixed via the previous homework solutions
def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    and where the 2/N factor is included in forward transform instead of inverse.
    a: left half-window length
    b: right half-window length
    """

    N = a+b
    halfN = N//2
    no = (b+1.)/2

    if not isInverse:
        # forward transform (data are N windowed time samples)
        preTwiddle = np.arange(N, dtype=np.float64)
        phase = -1j*np.pi/N
        preTwiddle = np.exp(phase*preTwiddle)*data

        postTwiddle = np.linspace(0.5,int(halfN-0.5),int(halfN))
        phase = -2j*np.pi*no/N
        # factor of 2./N is shifted from inverse
        postTwiddle = np.exp(phase*postTwiddle)*2./N

        return (postTwiddle*fft(preTwiddle)[:int(halfN)]).real
    else:
        # inverse transform (data are N/2 MDCT coeffs)
        preTwiddle = np.arange(N, dtype=np.float64)
        phase = 2j * np.pi*no/N
        preTwiddle = np.exp(phase*preTwiddle)

        postTwiddle = np.linspace(no, N+no-1, N)
        phase = 1j*np.pi/N

        # N was 2 before shifting 2./N to forward
        postTwiddle = N*np.exp(phase*postTwiddle)

        return ( postTwiddle * ifft(preTwiddle*np.concatenate((data, -data[::-1])))).real

def IMDCT(data,a,b):

    return MDCT(data, a, b, isInverse=True)

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    # Part (a) - testing
    # uncomment:
    x = [0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 3, 1,-1,-3, 0, 0, 0, 0]
    x1 = x[0:8]
    x2 = x[4:12]
    x3 = x[8:16]
    x4 = x[12:20]
    X1 = MDCTslow(x1, 4, 4)
    X2 = MDCTslow(x2, 4, 4)
    X3 = MDCTslow(x3, 4, 4)
    X4 = MDCTslow(x4, 4, 4)
    x1_t = MDCTslow(X1, 4, 4, isInverse=True)
    x2_t = MDCTslow(X2, 4, 4, isInverse=True)
    x3_t = MDCTslow(X3, 4, 4, isInverse=True)
    x4_t = MDCTslow(X4, 4, 4, isInverse=True)
    print(X1)
    print(X2)
    print(X3)
    print(X4)
    print(x1_t)
    print(x2_t)
    print(x3_t)
    print(x4_t)

    # Part (b) - testing
    # uncomment:
    x = [0, 1, 2, 3, 4, 4, 4, 4, 3, 1,-1,-3]
    prior_block = np.zeros(4)
    prior_imdct = np.zeros(4)
    follow_block = np.zeros(4)
    current_block = np.zeros(4)
    result = []
    
    x = np.concatenate((x, follow_block))
    for idx, block in enumerate(x):
        if idx % 4 != 0:
            continue
        current_block = np.concatenate((prior_block, x[idx:idx + 4]))
        prior_block = x[idx:idx + 4]

        mdct = MDCTslow(current_block, 4, 4)
        imdct = MDCTslow(mdct, 4, 4, isInverse=True)

        imdct = np.divide(imdct, 2)

        sum = np.round(np.add(prior_imdct[-4:], imdct[0:4]))
        if idx == 0:
            result = []
        else:
            result.append(sum)
        prior_imdct = imdct[4:]
    
    result = np.array(result).flatten()

    # Part (c) - testing
    # uncomment:
    rand_block = np.random.rand(2048)

    tic = time.time()
    mdct = MDCT(rand_block, 1024, 1024)
    imdct = IMDCT(mdct, 1024, 1024)
    toc = time.time()
    print("MDCT/IMDCT: ", toc - tic)

    tic = time.time()
    mdct = MDCTslow(rand_block, 1024, 1024)
    imdct = MDCTslow(mdct, 1024, 1024, isInverse=True)
    toc = time.time()
    print("MDCT/IMDCT: ", toc - tic)