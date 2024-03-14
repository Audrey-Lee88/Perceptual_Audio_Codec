""" Helper functions for block switching """
import numpy as np
from window import *
from scipy.fft import fft
import scipy.signal as sig
from scipy.io import wavfile
from scipy import stats
from psychoac import *
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times']})
rc('text')

# used https://stackoverflow.com/questions/39032325/python-high-pass-filter
def butter_highpass(cutoff, fs=44100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = sig.filtfilt(b, a, data)
    return y

def TFSFM(block, fs, cutoff = 8e3):
    N = len(block)
    K = N//2
    # TFSFM Implementation
    # SFM Implementation
    X_k = np.power(np.abs(fft(butter_highpass_filter(block, cutoff, fs))), 2)
    SFM = stats.gmean(X_k)/np.average(X_k)+1e-30


    # TFM Implementation
    x_n = np.power(np.abs(butter_highpass_filter(block, cutoff, fs)), 2)
    TFM = stats.gmean(x_n)/np.average(x_n)+1e-30

    TFSFM = SFM/TFM
    return TFSFM

def HFE(block, fs, cutoff=8e3):
    N = len(block)
    K = int(N * (cutoff/fs))

    block_fft = fft(KBDWindow(block))
    HFE = 0
    for i in np.arange(K, N//2, 1):
        const = 8/N**2
        HFE += const * np.abs(block_fft[i])**2
    HFE = SPL(HFE)
    return HFE


def containsTransient(prev_block, curr_block, fs):
    TFSFM_curr = TFSFM(curr_block, fs)
    TFSFM_prev = TFSFM(prev_block, fs)
    TFSFM_diff = np.abs(TFSFM_curr - TFSFM_prev)
    
    HFE_curr = HFE(curr_block, fs)
    HFE_prev = HFE(prev_block, fs)
    HFE_diff = np.abs(HFE_curr - HFE_prev)

    if HFE_diff > 10 and TFSFM_diff > 0.75:
        print("!", end="")
        return True
    else:
        print( ".",end="")
        return False
    
def HFETest(prev_block, curr_block, fs):
    HFE_curr = HFE(curr_block, fs)
    HFE_prev = HFE(prev_block, fs)
    HFE_diff = np.abs(HFE_curr - HFE_prev)

    if HFE_diff > 10:
        return True
    else:
        return False

def TFSFMTest(prev_block, curr_block, fs):
    TFSFM_curr = TFSFM(curr_block, fs)
    TFSFM_prev = TFSFM(prev_block, fs)
    TFSFM_diff = np.abs(TFSFM_curr - TFSFM_prev)

    if TFSFM_diff > 0.75:
        return True
    else:
        return False
   
if __name__ == "__main__":
    fs, audio = wavfile.read('test/glockenspiel.wav')
    
    block_size = 2048
    # transient_bools = np.zeros_like(audio)
    HFE_x = []
    HFE_y = []
    TFSFM_x = []
    TFSFM_y = []
    audio = audio[0:204800,:]

    prev_block =audio[0:block_size]
    for n in range(block_size, len(audio), block_size):
        start = n
        end = n+block_size
        curr_block = audio[start:end]
        for iCh in range(2):
            HFE_res = False
            HFE_res = HFETest(prev_block[:,iCh], curr_block[:,iCh], fs)
            if HFE_res == True:
                break
        if HFE_res == True:
            HFE_x.append(start+block_size//2)
            HFE_y.append(1)

        for iCh in range(2):
            TFSFM_res = False
            TFSFM_res = TFSFMTest(prev_block[:,iCh], curr_block[:,iCh], fs)
            if TFSFM_res == True:
                break
        if TFSFM_res == True:
            TFSFM_x.append(start+block_size//2)
            TFSFM_y.append(1)
        prev_block = curr_block


    plt.plot(audio[:,1], color='k', label='Audio Wav File')
    plt.scatter(HFE_x, np.multiply(1.3, np.multiply(HFE_y,np.max(audio[:, 1]))), color='k', marker='.', label="HFE Determined Transients")
    plt.scatter(TFSFM_x, np.multiply(1.1, np.multiply(TFSFM_y,np.max(audio[:, 1]))), color='k', marker='v', label="TFSFM Determined Transients")
    plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1),
          fancybox=True, shadow=True)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Glockenspiel Transient Detection')
    plt.savefig('test/figures/glockenspiel_transient_image.jpg', bbox_inches='tight', dpi=200)
    plt.show()