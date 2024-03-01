"""

Music 422  Marina Bosi

window.py -- Defines functions to window an array of discrete-time data samples

-----------------------------------------------------------------------
© 2009-2024 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------


"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift
import mdct
import matplotlib.pyplot as plt

### Problem 1.d ###
def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    N = len(dataSampleArray)
    window = np.sin(np.pi * (np.arange(0, N, 1) + 0.5) / N)
    return window * dataSampleArray


def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    N = len(dataSampleArray)
    window = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(0, N, 1) + 0.5) / N))

    return window * dataSampleArray


### Problem 1.d - OPTIONAL ###
def KBDWindow(dataSampleArray,alpha=4.):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following the KDB Window handout in the 
	Canvas Files/Assignments/HW3 folder
    """

    ### YOUR CODE STARTS HERE ###

    return np.zeros_like(dataSampleArray) # CHANGE THIS
    ### YOUR CODE ENDS HERE ###

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    N = 1024
    n = np.arange(0, N, 1)
    x = np.cos(2* np.pi * 7000 * n / 44100)

    sine_x = SineWindow(x)
    hanning_x = HanningWindow(x)

    sine_fft = fftshift(fft(sine_x))
    hanning_fft = fftshift(fft(hanning_x))
    sine_mdct = mdct.MDCT(sine_x, 512, 512)

    sine_co = 4/(N**2 * 0.5**2)
    hanning_co = 4/(N**2 * 0.375**2)
    mdct_sine_co = 8/(N**2 * 0.5**2)

    sine_SPL = (96 + 10 * np.log10(sine_co * np.abs(sine_fft)**2))[int(N/2):]

    hanning_SPL = (96 + 10 * np.log10(hanning_co * np.abs(hanning_fft)**2))[int(N/2):]

    mdct_SPL = 96 + 10 * np.log10(mdct_sine_co * np.abs(sine_mdct)**2)

    f = np.linspace(0, 44.1e3/2, 512)
    plt.subplot(2,1,1)

    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.suptitle('SPL of Windowed and Tranformed Signals')
    plt.xlabel('Frequency (Hz)')

    axs[0].plot(f, sine_SPL)
    axs[0].set_title('Sine Windowed and DFT')
    axs[0].set_ylabel('SPL (dB)')
    axs[1].plot(f, hanning_SPL)
    axs[1].set_title('Hanning Windowed and DFT')
    axs[1].set_ylabel('SPL (dB)')
    axs[2].plot(f, mdct_SPL)
    axs[2].set_title('Sine Windowed and MDCT')
    axs[2].set_ylabel('SPL (dB)')

    plt.savefig('part_f_plot.png')
