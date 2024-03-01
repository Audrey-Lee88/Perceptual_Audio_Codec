"""
psychoac.py -- masking models implementation

-----------------------------------------------------------------------
(c) 2011-2024 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np
from window import *
import scipy

def SPL(intensity):
    """
    Returns the SPL corresponding to intensity 
    """
    spl = 96 + 10*np.log10(intensity)
    spl = np.array(spl)
    spl[spl < -30] = -30

    return spl

def Intensity(spl):
    """
    Returns the intensity  for SPL spl
    """
    intensity = 10**((spl - 96)/10)
    return intensity

def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    f = np.array(f)
    f[f < 20] = 20

    f_thresh = (3.64 * (f / 1e3)**(-0.8)) - (6.5 * np.exp(-0.6 * (f / 1e3 - 3.3)**2)) + (10**(-3) * (f / 1e3)**4)

    return f_thresh

def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    bark = 13 * np.arctan(0.76 * f / 1e3) + 3 * np.arctan((f / 7.5e3)**2)
    return bark

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
        self.f = f
        self.SPL = SPL
        self.isTonal = isTonal

    def IntensityAtFreq(self,freq):
        """The intensity at frequency freq"""
        return self.IntensityAtBark(Bark(freq))

    def IntensityAtBark(self,z):
        """The intensity at Bark location z"""
        dz = z - Bark(self.f)

        # Defining the piecewise function
        if dz >= -0.5 or dz <= 0.5:
            piece_out = 0
        elif dz < -0.5:
            piece_out = -27 * (np.abs(dz) - 0.5)
        elif dz > 0.5:
            piece_out = (-27 + 0.367 * np.max([self.SPL - 40, 0])) * (np.abs(dz) - 0.5)
        
        if self.isTonal:
            drop_delta = 16
        else:
            drop_delta = 6

        slope = piece_out - drop_delta
        slope += self.SPL
        return Intensity(slope)
    
    def vIntensityAtBark(self,zVec):
        """The intensity at vector of Bark locations zVec"""
        dz_array = zVec - Bark(self.f)
        piece_out = np.zeros_like(dz_array)

        piece_out[dz_array < -0.5] = -27 * (np.abs(dz_array[dz_array < -0.5]) - 0.5)
        piece_out[dz_array > -0.5] = (-27 + 0.367 * np.max([self.SPL - 40, 0])) * (np.abs(dz_array[dz_array > -0.5]) - 0.5)

        if self.isTonal:
            drop_delta = 16
        else:
            drop_delta = 6

        slope = piece_out - drop_delta
        slope += self.SPL
        return Intensity(slope)


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
    max_f = 0.5*sampleRate
    lines = np.linspace(0,max_f,nMDCTLines)

    lines_per_band = np.zeros(len(flimit))
    band = 0
    lines_num = 0
    for _, line in enumerate(lines):
        if line <= flimit[band]:
            lines_num += 1
        else:
            lines_per_band[band] = lines_num
            # increment the band
            band += 1
            # set up the next band number of line numbers
            lines_num = 1

    lines_per_band[band] = lines_num
    return lines_per_band

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
        self.nLines = np.array(nLines, dtype=int)
        self.nBands = len(self.nLines)
        self.lowerLine = []
        self.upperLine = []

        band = 0
        for i, _ in enumerate(nLines):
            self.lowerLine.append(band)
            self.upperLine.append(band + nLines[i] - 1)
            band += nLines[i]

        self.lowerLine = np.array(self.lowerLine, dtype=int)
        self.upperLine = np.array(self.upperLine, dtype=int)
       


def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    windowed_data_f = np.abs(fft(HanningWindow(data)))

    # Find the peaks in the first half
    windowed_data_f = windowed_data_f[:len(windowed_data_f)//2]
    k = scipy.signal.find_peaks(windowed_data_f)[0]

    # find intensity values and calculate SPL of signal
    N = len(data)
    window_intensity = (4 / (N**2 * (3/8))) * windowed_data_f**2

    # Implement Equation 4 in the handout
    f_p_val = []
    A_2_val = []

    # calculate A2 and fp for peaks at kp
    for k_p in k:
        low_bound = k_p - 1
        up_bound = k_p + 2

        m = np.arange(low_bound, up_bound, 1)
        X = windowed_data_f[low_bound:up_bound]

        f_p = (sampleRate/N)*np.sum(m*X**2) / np.sum(X**2)

        a_2 = np.sum(window_intensity[low_bound:up_bound])

        f_p_val.append(f_p)
        A_2_val.append(a_2)

    maskingCurves = []
    f = np.linspace(0, sampleRate/2, N//2)
    for i in range(len(k)):
        this_masker = Masker(f_p_val[i], SPL(A_2_val[i]))
        this_curve_spl = SPL(this_masker.vIntensityAtBark(Bark(f)))
        maskingCurves.append(this_curve_spl)

    maskingCurves_array = np.array(maskingCurves)
    quiet_thresh = Thresh(f)
    thresh = []
    for n in range(len(f)):
        thresh.append(np.max(maskingCurves_array[:,n] + quiet_thresh[n]))
    
    return np.array(thresh)


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
    mdct_data_spl = SPL(2 / (3/8) * np.abs(MDCTdata / (2**MDCTscale))**2)

    thresh = getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands)
    smr_array = []
    for i in range(sfBands.nBands):
        
        low_bound = int(sfBands.lowerLine[i])
        up_bound = int(sfBands.upperLine[i] + 1)

        mask = mdct_data_spl[low_bound:up_bound] - thresh[low_bound:up_bound]

        if len(mask) > 0:
            smr_array.append(np.max(mask))
        else:
            smr_array.append(mask[low_bound])
    
    return np.array(smr_array) 

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    print(len(cbFreqLimits))
