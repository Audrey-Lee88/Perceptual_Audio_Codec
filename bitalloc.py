"""
Music 422
-----------------------------------------------------------------------
(c) 2009-2024 Marina Bosi  -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np

# Question 1.c)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformly distributed for the mantissas.
    """
    # We calculated 2/3 earlier from part 1a and 1b.
    return 2*np.ones(nBands, dtype = np.int)
    # return 3*np.ones(nBands, dtype = np.int)

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    # 128
    array = [11,15,13,14,14,10,9,13,9,6,5,5,4,3,3,3,2,8,11,1,0,10,1,0,0]
    # 192
    # array = [11,15,13,14,14,10,9,13,9,6,5,5,4,4,3,3,2,8,12,1,0,10,1,0,0]
    return np.array(array, dtype = int)

def BitAllocConstNMR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
    # 128
    array = [4,10,7,9,10,6,6,10,6,5,7,9,10,10,10,7,3,6,10,1,0,9,0,0,0]
    # 192
    # array = [7,12,10,12,12,9,9,12,9,8,10,12,13,13,13,10,6,9,12,4,3,12,2,0,0]
    return np.array(array, dtype = int)

# Question 2.a)
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band

        Returns:
            bits[nBands] is number of bits allocated to each scale factor band

        
    """
    threshold = np.max(SMR)

    bit_alloc = np.zeros_like(SMR, dtype=int)

    while bitBudget > 0:
        # Subtract 6 dB from the threshold
        threshold -= 6
        # Check if we should add at this threshold at all:
        subtraction_factor = np.sum(nLines[np.logical_and(SMR > threshold, bit_alloc < 16)])
        # Check if the bit budget would be exhausted
        bitBudget -= subtraction_factor
        if bitBudget >= 0:
            bitBudget += subtraction_factor
            # iterate through all sub-bands
            for b in range(nBands):
                if SMR[b] > threshold:
                    if bit_alloc[b] < 16:
                        bit_alloc[b] += 1
                        bitBudget -= nLines[b]
        else:
            bitBudget += subtraction_factor
            for b in range(nBands):
                if SMR[b] > threshold:
                    if bit_alloc[b] < 16:
                        bitBudget -= nLines[b]
                        if bitBudget >= 0:
                            bit_alloc[b] += 1

    # Check for lonely bits
    for b in range(nBands):
        if bit_alloc[b] == 1:
            if b == 0:
                if (bit_alloc[b+1] != 0): bit_alloc[b+1] += 1
            else:
                if (bit_alloc[b-1] != 0): bit_alloc[b-1] += 1
            bit_alloc[b] = 0
        
    return bit_alloc

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    pass