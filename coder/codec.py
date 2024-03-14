"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2019-2024 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from window import *  # current window used for MDCT -- implement KB-derived?
from mdct import MDCT,IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)

# used only by Encode
from psychoac import CalcSMRs  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc  #allocates bits to scale factor bands given SMRs
from helper import *

def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    rescaleLevel = 1.*(1<<overallScaleFactor)
    # prepare various constants
    if codingParams.currBlock == 0:
        halfN = codingParams.nMDCTLines_l
    elif codingParams.currBlock == 1:
        halfN = codingParams.nMDCTLines_t
    elif codingParams.currBlock == 2:
        halfN = codingParams.nMDCTLines_s
    elif codingParams.currBlock == 3:
        halfN = codingParams.nMDCTLines_t
    N = 2*halfN

    if codingParams.currBlock == 0:
        sfBands = codingParams.sfBands_l
    elif codingParams.currBlock == 1:
        sfBands = codingParams.sfBands_t
    elif codingParams.currBlock == 2:
        sfBands = codingParams.sfBands_s
    elif codingParams.currBlock == 3:
        sfBands = codingParams.sfBands_t

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN,dtype=np.float64)
    iMant = 0
    for iBand in range(sfBands.nBands):
        nLines = sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            mdctLine[iMant:(iMant+nLines)]=vDequantize(scaleFactor[iBand], mantissa[iMant:(iMant+nLines)],codingParams.nScaleBits, bitAlloc[iBand])
            # print(mdctLine[iMant:(iMant+nLines)],'mdct')
        iMant += nLines
    mdctLine /= rescaleLevel  # put overall gain back to original level


    # IMDCT and window the data for this channel
    if codingParams.currBlock == 0:
        data = KBDWindow(IMDCT(mdctLine, halfN, halfN))
    elif codingParams.currBlock == 1:
        data = LongtoShortWindow(IMDCT(mdctLine, codingParams.nMDCTLines_l, codingParams.nMDCTLines_s),codingParams.nMDCTLines_l,codingParams.nMDCTLines_s)
    elif codingParams.currBlock == 2:
        data = SineWindow(IMDCT(mdctLine, halfN, halfN))
    elif codingParams.currBlock == 3:
        data = ShorttoLongWindow(IMDCT(mdctLine, codingParams.nMDCTLines_s, codingParams.nMDCTLines_l),codingParams.nMDCTLines_s,codingParams.nMDCTLines_l)
    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data


def Encode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []

    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        (s,b,m,o) = EncodeSingleChannel(data[iCh],codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
    # return results bundled over channels
    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor)


def EncodeSingleChannel(data,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    if codingParams.currBlock == 0:
        halfN = codingParams.nMDCTLines_l
    elif codingParams.currBlock == 1:
        halfN = codingParams.nMDCTLines_t
    elif codingParams.currBlock == 2:
        halfN = codingParams.nMDCTLines_s
    elif codingParams.currBlock == 3:
        halfN = codingParams.nMDCTLines_t
    N = 2*halfN
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    if codingParams.currBlock == 0:
        sfBands = codingParams.sfBands_l
    elif codingParams.currBlock == 1:
        sfBands = codingParams.sfBands_t
    elif codingParams.currBlock == 2:
        sfBands = codingParams.sfBands_s
    elif codingParams.currBlock == 3:
        sfBands = codingParams.sfBands_t

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data

    if codingParams.currBlock == 0:
        mdctTimeSamples = KBDWindow(data)
        mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]
    elif codingParams.currBlock == 1:
        mdctTimeSamples = LongtoShortWindow(data,codingParams.nMDCTLines_l,codingParams.nMDCTLines_s)
        mdctLines = MDCT(mdctTimeSamples, codingParams.nMDCTLines_l, codingParams.nMDCTLines_s)[:halfN]
    elif codingParams.currBlock == 2:
        mdctTimeSamples = SineWindow(data)
        mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]
    elif codingParams.currBlock == 3:
        mdctTimeSamples = ShorttoLongWindow(data,codingParams.nMDCTLines_s,codingParams.nMDCTLines_l)
        mdctLines = MDCT(mdctTimeSamples, codingParams.nMDCTLines_s, codingParams.nMDCTLines_l)[:halfN]

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)
    # perform bit allocation using SMR results
    bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)
    # bitAlloc = BitAllocWaterFilling( bitBudget , maxMantBits , sfBands.nBands , sfBands.nLines , SMRs )

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            # print(mdctLines[lowLine:highLine],'mdct_encode')
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)



