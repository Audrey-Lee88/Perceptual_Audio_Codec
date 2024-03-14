import abc
import copy
from typing import List

import numpy as np

from prob_dist import Frequencies


class FreqModelBase(abc.ABC):
    """Base Freq Model

    The Arithmetic Entropy Coding (AEC) encoder can be thought of consisting of two parts:
    1. The probability model
    2. The "lossless coding" algorithm which uses these probabilities

    Note that the probabilities/frequencies coming from the probability model are fixed in the simplest Arithmetic coding
    version, but they can be modified as we parse each symbol.
    This class represents a generic "probability Model", but using frequencies (or counts), and hence the name FreqModel.
    Frequencies are used, mainly because floating point values can be unpredictable/uncertain on different platforms.

    Some typical examples of Freq models are:

    a) FixedFreqModel -> the probability model is fixed to the initially provided one and does not change
    b) AdaptiveIIDFreqModel -> starts with some initial probability distribution provided
        (the initial distribution is typically uniform)
        The Adaptive Model then updates the model based on counts of the symbols it sees.

    Args:
        freq_initial -> the frequencies used to initialize the model
        max_allowed_total_freq -> to limit the total_freq values of the frequency model
    """

    def __init__(self, freqs_initial: Frequencies, max_allowed_total_freq):
        # initialize the current frequencies using the initial freq.
        # NOTE: the deepcopy here is needed as we modify the frequency table internally
        # so, if it is used elsewhere externally, then it can cause unexpected issued
        self.freqs_current = copy.deepcopy(freqs_initial)
        self.max_allowed_total_freq = max_allowed_total_freq