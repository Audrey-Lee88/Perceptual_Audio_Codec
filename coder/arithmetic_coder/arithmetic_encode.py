from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any
from bitarray_util import BitArray, uint_to_bitarray, bitarray_to_uint
from data_block import DataBlock
from prob_dist import Frequencies
from data_encoder_decoder import DataEncoder
from probability_models import FreqModelBase

@dataclass
class AECParams:
    """AEC hyper parameters

    These are a couple of parameters used by the AEC encoder. more details inline
    """

    # represents the number of bits used to represent the size of the input data_block
    DATA_BLOCK_SIZE_BITS: int = 32

    # number of bits used to represent the arithmetic coder range
    PRECISION: int = 32

    def __post_init__(self):
        self.FULL: int = 1 << self.PRECISION
        self.HALF: int = 1 << (self.PRECISION - 1)
        self.QTR: int = 1 << (self.PRECISION - 2)
        self.MAX_ALLOWED_TOTAL_FREQ: int = self.QTR
        self.MAX_BLOCK_SIZE: int = 1 << self.DATA_BLOCK_SIZE_BITS

class ArithmeticEncoder(DataEncoder):
    """Finite precision Arithmetic encoder

    The encoder and decoders are based on the following sources:
    - https://youtu.be/ouYV3rBtrTI: This series of videos on Arithmetic coding are a very gradual but a great
    way to understand them
    - Charles Bloom's blog: https://www.cbloom.com/algs/statisti.html#A5
    - There is of course the original paper:
        https://web.stanford.edu/class/ee398a/handouts/papers/WittenACM87ArithmCoding.pdf
    """

    def __init__(
        self, params: AECParams, freq_model_params: Frequencies, freq_model_cls: FreqModelBase
    ):
        self.params = params

        # define the probability model used by the AEC
        # the model can get updated when we call update_model(s) after every step
        if isinstance(freq_model_params, tuple):
            # if freq_model_params is a tuple, we unpack the parameters
            self.freq_model = freq_model_cls(*freq_model_params, params.MAX_ALLOWED_TOTAL_FREQ)
        else:
            self.freq_model = freq_model_cls(freq_model_params, params.MAX_ALLOWED_TOTAL_FREQ)

    @classmethod
    def shrink_range(cls, freqs: Frequencies, s: Any, low: int, high: int) -> Tuple[int, int]:
        """shrinks the range (low, high) based on the symbol s

        Args:
            s (Any): symbol to encode

        Returns:
            Tuple[int, int]: (low, high) ranges returned after shrinking
        """

        # compute some intermediate variables: rng, c, d
        rng = high - low
        c = freqs.cumulative_freq_dict[s]
        d = c + freqs.frequency(s)

        # perform shrinking of low, high
        # NOTE: this is the basic Arithmetic coding step implemented using integers
        high = low + (rng * d) // freqs.total_freq
        low = low + (rng * c) // freqs.total_freq
        return (low, high)

    def encode_block(self, data_block: DataBlock):
        """Encode block function for arithmetic coding"""

        # ensure data_block.size is not too big
        err_msg = "choose a larger DATA_BLOCK_SIZE_BITS, as data_block.size is too big"
        assert data_block.size < (1 << self.params.MAX_BLOCK_SIZE), err_msg

        # initialize the low and high states
        low = 0
        high = self.params.FULL

        # initialize the output
        encoded_bitarray = BitArray("")

        # add the data_block size at the beginning
        # NOTE: Arithmetic decoding needs a way to indicate where to stop the decoding
        # One way is to add a character at the end which signals EOF (end-of-file). This requires us to
        # change the probabilities of the other symbols. Another way is to just signal the size of the
        # block. These two approaches add a bit of overhead. We use the second approach as it is much more transparent.
        encoded_bitarray = uint_to_bitarray(data_block.size, self.params.DATA_BLOCK_SIZE_BITS)

        # initialize counter for mid-range re-adjustments
        # used to ensure that the range doesn't become too small avoiding finite precision problem in AEC
        num_mid_range_readjust = 0

        # start the encoding
        for s in data_block.data_list:
            # ensure freqs.total_freq is not too big
            err_msg = """the frequency total is too large, which might cause stability issues. 
            Please increase the precision (or reduce the total_freq)"""
            assert (
                self.freq_model.freqs_current.total_freq < self.params.MAX_ALLOWED_TOTAL_FREQ
            ), err_msg

            # shrink range
            # i.e. the core Arithmetic encoding step
            low, high = ArithmeticEncoder.shrink_range(self.freq_model.freqs_current, s, low, high)
            # update the freq model for encoding the next symbol
            self.freq_model.update_model(s)

            # perform re-normalizing range
            # NOTE: the low, high values need to be re-normalized as else they will keep shrinking
            # and after a few iterations things will be infeasible.
            # The goal of re-normalizing is to not let the range (high - low) get smaller than self.params.QTR

            # CASE I, II -> simple cases where low, high are both in the same half of the range
            while (high < self.params.HALF) or (low > self.params.HALF):
                if high < self.params.HALF:
                    # output 1's corresponding to prior mid-range readjustments
                    encoded_bitarray.extend("0" + "1" * num_mid_range_readjust)

                    # re-adjust range, and reset params
                    low = low << 1
                    high = high << 1
                    num_mid_range_readjust = 0  # reset the mid-range readjustment counter

                elif low > self.params.HALF:
                    # output 0's corresponding to prior mid-range readjustments
                    encoded_bitarray.extend("1" + "0" * num_mid_range_readjust)

                    # re-adjust range, and reset params
                    low = (low - self.params.HALF) << 1
                    high = (high - self.params.HALF) << 1
                    num_mid_range_readjust = 0  # reset the mid-range readjustment counter

            # CASE III -> the more complex case where low, high straddle the midpoint
            while (low > self.params.QTR) and (high < 3 * self.params.QTR):
                # increment the mid-range adjustment counter
                num_mid_range_readjust += 1
                low = (low - self.params.QTR) << 1
                high = (high - self.params.QTR) << 1

        # Finally output a few bits to signal the final range + any remaining mid range readjustments
        num_mid_range_readjust += 1  # this increment is mainly to output either 01, 10
        if low <= self.params.QTR:
            # output 0's corresponding to prior mid-range readjustments
            encoded_bitarray.extend("0" + num_mid_range_readjust * "1")
        else:
            # output 1's corresponding to prior mid-range readjustments
            encoded_bitarray.extend("1" + num_mid_range_readjust * "0")

        return encoded_bitarray