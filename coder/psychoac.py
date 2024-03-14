"""
psychoac.py -- masking models implementation

-----------------------------------------------------------------------
(c) 2011-2024 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np
from window import *
import scipy

FULLSCALESPL = 96.
SPLFLOOR = -30

def SPL(intensity):
    """
    Returns the SPL corresponding to intensity 
    """
    # spl = 96 + 10*np.log10(intensity)
    # spl = np.array(spl)
    # spl[spl < -30] = -30

    # return spl
    return np . maximum ( SPLFLOOR , FULLSCALESPL + 10.* np . log10 ( intensity + 1e-100))

def Intensity(spl):
    """
    Returns the intensity  for SPL spl
    """
    # intensity = 10**((spl - 96)/10)
    # return intensity
    return np . power (10. ,( spl - FULLSCALESPL )/10.)

def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    # f = np.array(f)
    # f[f < 20] = 20

    # f_thresh = (3.64 * (f / 1e3)**(-0.8)) - (6.5 * np.exp(-0.6 * (f / 1e3 - 3.3)**2)) + (10**(-3) * (f / 1e3)**4)

    # return f_thresh
    f = np . maximum (f ,20.)
    return 3.64* np . power ( f /1000. , -0.8) - 6.5* np . exp ( -0.6*( f /1000. -3.3) * \
    ( f /1000. -3.3)) + 0.001* np . power ( f /1000. ,4)

def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    # bark = 13 * np.arctan(0.76 * f / 1e3) + 3 * np.arctan((f / 7.5e3)**2)
    # return bark
    return 13.0* np . arctan (0.76* f /1000.)+3.5* np . arctan (( f /7500.)*( f /7500.))

DBTOBITS = 6.02
MASKTONALDROP = 16
MASKNOISEDROP = 6
class Masker:
    """
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the
    masker frequency
    """

    def __init__(self,f,SPL,isTonal=True):
        """
        initialized with the frequency and SPL of a masker and whether or not
        it is Tonal
        """
        # self.f = f
        # self.SPL = SPL
        # self.isTonal = isTonal
        self . SPL = SPL # SPL of the masker
        self . f = f # frequency of the masker
        self . z = Bark ( f ) # frequency in Bark scale of masker
        self . drop = MASKTONALDROP
        if not isTonal : self . drop = MASKNOISEDROP

    def IntensityAtFreq(self,freq):
        """The intensity at frequency freq"""
        return self.IntensityAtBark(Bark(freq))

    def IntensityAtBark(self,z):
        """The intensity at Bark location z"""
        # dz = z - Bark(self.f)

        # # Defining the piecewise function
        # if dz >= -0.5 or dz <= 0.5:
        #     piece_out = 0
        # elif dz < -0.5:
        #     piece_out = -27 * (np.abs(dz) - 0.5)
        # elif dz > 0.5:
        #     piece_out = (-27 + 0.367 * np.max([self.SPL - 40, 0])) * (np.abs(dz) - 0.5)
        
        # if self.isTonal:
        #     drop_delta = 16
        # else:
        #     drop_delta = 6

        # slope = piece_out - drop_delta
        # slope += self.SPL
        # return Intensity(slope)
        maskedDB = self . SPL - self . drop
        # if more than half a critical band away , drop lower at appropriate
        # spreading function rate
        if abs ( self .z - z ) >0.5 :
            if self .z > z : # masker above maskee
                maskedDB -= 27.*( self .z -z -0.5)
            else : # masker below maskee
                iEffect = self . SPL -40.
                if np . abs ( iEffect )!= iEffect : iEffect =0.
                maskedDB -= (27. -0.367* iEffect )*( z - self .z -0.5)
        # return resulting intensity
        return Intensity ( maskedDB )
    
    def vIntensityAtBark(self,zVec):
        # """The intensity at vector of Bark locations zVec"""
        # dz_array = zVec - Bark(self.f)
        # piece_out = np.zeros_like(dz_array)

        # piece_out[dz_array < -0.5] = -27 * (np.abs(dz_array[dz_array < -0.5]) - 0.5)
        # piece_out[dz_array > -0.5] = (-27 + 0.367 * np.max([self.SPL - 40, 0])) * (np.abs(dz_array[dz_array > -0.5]) - 0.5)

        # if self.isTonal:
        #     drop_delta = 16
        # else:
        #     drop_delta = 6

        # slope = piece_out - drop_delta
        # slope += self.SPL
        # return Intensity(slope)
        """ The intensity of this masker at vector of Bark locations zVec """
        # start at dB near - masking level for type of masker
        maskedDB = np . empty ( np . size ( zVec ) , np . float64 )
        maskedDB . fill ( self . SPL - self . drop )
        # if more than half a critical band away , drop lower at appropriate
        # spreading function rate adjust lower frequencies ( beyond 0.5 Bark away )
        v = (( self .z -0.5) > zVec )
        maskedDB [ v ] -= 27.*( self .z - zVec [ v ] -0.5)
        # adjust higher frequencies ( beyond 0.5 Bark away )
        iEffect = self . SPL -40.
        if iEffect <0.: iEffect =0.
        v = (( self . z +0.5) < zVec )
        maskedDB [ v ] -= (27. -0.367* iEffect )*( zVec [ v ] -( self . z +0.5))
        # return resulting intensity
        return Intensity ( maskedDB )


# Default data for 25 scale factor bands based on the traditional 25 critical bands
# 24kHz is the upper limit of the human hearing range
cbFreqLimits = [100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 24000]

def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    # max_f = 0.5*sampleRate
    # lines = np.linspace(0,max_f,nMDCTLines)

    # lines_per_band = np.zeros(len(flimit))
    # band = 0
    # lines_num = 0
    # for _, line in enumerate(lines):
    #     if line <= flimit[band]:
    #         lines_num += 1
    #     else:
    #         lines_per_band[band] = lines_num
    #         # increment the band
    #         band += 1
    #         # set up the next band number of line numbers
    #         lines_num = 1

    # lines_per_band[band] = lines_num
    # return lines_per_band
    lineToFreq = 0.5* sampleRate / nMDCTLines
    maxFreq = ( nMDCTLines -1+0.5)* lineToFreq
    # first get upper line for each band
    nLines = [ ]
    iLast = -1 # the last line before the start of this group
    # ( -1 when we start at zero )
    for iLine in range ( len ( flimit )):
        if flimit [ iLine ] > maxFreq :
            nLines . append ( nMDCTLines -1 - iLast )
            break
            # otherwise , find last lin in this band , compute number , and save last
        # for next loop
        iUpper = int ( flimit [ iLine ]/ lineToFreq -0.5)
        nLines . append ( iUpper - iLast )
        iLast = iUpper
    return nLines

class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """

    def __init__(self,nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        """
        # Need to dtype=int in order for the pacfile file to work
        # self.nLines = np.array(nLines, dtype=int)
        # self.nBands = len(self.nLines)
        # self.lowerLine = []
        # self.upperLine = []

        # band = 0
        # for i, _ in enumerate(nLines):
        #     self.lowerLine.append(band)
        #     self.upperLine.append(band + nLines[i] - 1)
        #     band += nLines[i]

        # self.lowerLine = np.array(self.lowerLine, dtype=int)
        # self.upperLine = np.array(self.upperLine, dtype=int)

        self . nBands = len ( nLines )
        self . nLines = np . array ( nLines , dtype = np . uint16 )
        self . lowerLine = np . empty ( self . nBands , dtype = np . uint16 )
        self . upperLine = np . empty ( self . nBands , dtype = np . uint16 )
        self . lowerLine [0]=0
        self . upperLine [0]= nLines [0] -1
        for iBand in range (1 , self . nBands ):
            self . lowerLine [ iBand ]= self . upperLine [ iBand -1]+1
            self . upperLine [ iBand ]= self . upperLine [ iBand -1]+ nLines [ iBand ]
       


def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    N = len(data)
    nLines = N//2
    lineToFreq = sampleRate/N
    nBands = sfBands.nBands

    fftData = fft(HanningWindow(data))[:nLines]
    fftIntensity = 32./3./N/N*np.abs(fftData)**2
    fftSPL = SPL(fftIntensity)

    maskers = []    
    for i in range(2, nLines-2):
        if fftIntensity[i]>fftIntensity[i-1] and \
            fftIntensity[i]>fftIntensity[i+1]:

            spl = fftIntensity[i]+fftIntensity[i-1]+fftIntensity[i+1]

            f = lineToFreq*(i*fftIntensity[i]+(i-1)+fftIntensity[i-1] + \
                            (i+1)*fftIntensity[i+1])/spl
            spl = SPL(spl)

            if spl > Thresh(f):
                maskers.append(Masker(f,spl))

    fline = lineToFreq*np.linspace(0.5,nLines-0.5,nLines)
    zline = Bark(fline)

    maskedSPL = np.zeros(nLines, dtype=np.float64)

    for m in maskers: maskedSPL += m.vIntensityAtBark(zline)
    maskedSPL += Intensity(Thresh(fline))
    maskedSPL = SPL(maskedSPL)
    
    return maskedSPL

def CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Set SMR for each critical band in sfBands.

    Arguments:
                data:       is an array of N time domain samples
                MDCTdata:   is an array of N/2 MDCT frequency coefficients for the time domain samples
                            in data; note that the MDCT coefficients have been scaled up by a factor
                            of 2^MDCTscale
                MDCTscale:  corresponds to an overall scale factor 2^MDCTscale for the set of MDCT
                            frequency coefficients
                sampleRate: is the sampling rate of the time domain samples
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band

    Returns:
                SMR[sfBands.nBands] is the maximum signal-to-mask ratio in each
                                    scale factor band

    Logic:
                Performs an FFT of data[N] and identifies tonal and noise maskers.
                Combines their relative masking curves and the hearing threshold
                to calculate the overall masked threshold at the MDCT frequency locations. 
				Then determines the maximum signal-to-mask ratio within
                each critical band and returns that result in the SMR[] array.
    """
    # mdct_data_spl = SPL(2 / (3/8) * np.abs(MDCTdata / (2**MDCTscale))**2)

    # thresh = getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands)
    # smr_array = []
    # for i in range(sfBands.nBands):
        
    #     low_bound = int(sfBands.lowerLine[i])
    #     up_bound = int(sfBands.upperLine[i] + 1)

    #     mask = mdct_data_spl[low_bound:up_bound] - thresh[low_bound:up_bound]

    #     if len(mask) > 0:
    #         smr_array.append(np.max(mask))
    #     else:
    #         smr_array.append(mask[low_bound])
    
    # return np.array(smr_array)
    nBands = sfBands . nBands # number of sf bands spanned by MDCT lines
    # also get spectral densitites from MDCT data ( in SPL to compute SMR later )
    dtemp2 = DBTOBITS * MDCTscale # adjust MDCTdata level for any overall
    # scale factor
    mdctSPL = 4.* MDCTdata **2 # 8/ N ^2 for MDCT Parsevals * 2 for sine
    # window but 4/ N ^2 already in MDCT forward
    mdctSPL = SPL ( mdctSPL ) - dtemp2
    maskedSPL = getMaskedThreshold( data , MDCTdata , MDCTscale , sampleRate , sfBands )
    # Compute and return SMR for each scale factor band as max value for
    # lines in band
    SMR = np . empty ( nBands , dtype = np . float64 )
    for i in range ( nBands ) :
        lower = sfBands . lowerLine [ i ]
        upper = sfBands . upperLine [ i ]+1 # slices don ’t include last item in range
        SMR [ i ]= np . max ( mdctSPL [ lower : upper ] - maskedSPL [ lower : upper ] )
    return SMR

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    print(len(cbFreqLimits))
