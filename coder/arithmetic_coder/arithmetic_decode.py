from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any
from bitarray_util import BitArray, uint_to_bitarray, bitarray_to_uint
from data_block import DataBlock
from prob_dist import Frequencies
from arithmetic_encode import AECParams, ArithmeticEncoder
from data_encoder_decoder import DataDecoder
from probability_models import FreqModelBase


class ArithmeticDecoder(DataDecoder):
    """Finite precision Arithmetic decoder

    The encoder and decoders are based on the following sources:
    - https://youtu.be/ouYV3rBtrTI: This series of videos on Arithmetic coding are a very gradual but a great
    way to understand them
    - Charles Bloom's blog: https://www.cbloom.com/algs/statisti.html#A5
    """

    def __init__(
        self, params: AECParams, freq_model_params: Frequencies, freq_model_cls: FreqModelBase
    ):
        self.params = params
        if isinstance(freq_model_params, tuple):
            # if freq_model_params is a tuple, we unpack the parameters
            self.freq_model = freq_model_cls(*freq_model_params, params.MAX_ALLOWED_TOTAL_FREQ)
        else:
            self.freq_model = freq_model_cls(freq_model_params, params.MAX_ALLOWED_TOTAL_FREQ)

    def decode_step_core(self, low: int, high: int, state: int, freqs: Frequencies):
        """Core Arithmetic decoding function

        We cut the [low, high) range bits proportional to the cumulative probability of each symbol
        the function locates the bin in which the state lies
        NOTE: This is exactly same as the decoding function of the theoretical arithmetic decoder,
        except implemented using integers

        Args:
            low (int): range low point
            high (int): range high point
            state (int): the arithmetic decoder state

        Returns:
            s : the decoded symbol
        """

        # FIXME: simplify this search.
        rng = high - low
        search_list = (
            low + (np.array(list(freqs.cumulative_freq_dict.values())) * rng) // freqs.total_freq
        )
        start_bin = np.searchsorted(search_list, state, side="right") - 1
        s = freqs.alphabet[start_bin]
        return s

    def decode_block(self, encoded_bitarray: BitArray):

        data_block_size_bitarray = encoded_bitarray[: self.params.DATA_BLOCK_SIZE_BITS]
        encoded_bitarray = encoded_bitarray[self.params.DATA_BLOCK_SIZE_BITS :]

        # get data size
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)

        # initialize return variables
        decoded_data_list = []
        num_bits_consumed = 0

        # initialize intermediate state vars etc.
        low = 0
        high = self.params.FULL
        state = 0
        arith_bitarray_size = len(encoded_bitarray)

        # initialize the state
        while (num_bits_consumed < self.params.PRECISION) and (
            num_bits_consumed < arith_bitarray_size
        ):
            bit = encoded_bitarray[num_bits_consumed]
            if bit:
                state += 1 << (self.params.PRECISION - num_bits_consumed - 1)
            num_bits_consumed += 1
        num_bits_consumed = self.params.PRECISION

        # main decoding loop
        while True:
            # decode the next symbol
            s = self.decode_step_core(low, high, state, self.freq_model.freqs_current)
            low, high = ArithmeticEncoder.shrink_range(self.freq_model.freqs_current, s, low, high)
            decoded_data_list.append(s)

            # update the freq_model
            self.freq_model.update_model(s)

            # break when we have decoded all the symbols in the data block
            if len(decoded_data_list) == input_data_block_size:
                break

            while (high < self.params.HALF) or (low > self.params.HALF):
                if high < self.params.HALF:
                    # re-adjust range, and reset params
                    low = low << 1
                    high = high << 1
                    state = state << 1

                elif low > self.params.HALF:
                    # re-adjust range, and reset params
                    low = (low - self.params.HALF) << 1
                    high = (high - self.params.HALF) << 1
                    state = (state - self.params.HALF) << 1

                if num_bits_consumed < arith_bitarray_size:
                    bit = encoded_bitarray[num_bits_consumed]
                    state += bit
                num_bits_consumed += 1

            while (low > self.params.QTR) and (high < 3 * self.params.QTR):
                # increment the mid-range adjustment counter
                low = (low - self.params.QTR) << 1
                high = (high - self.params.QTR) << 1
                state = (state - self.params.QTR) << 1

                if num_bits_consumed < arith_bitarray_size:
                    bit = encoded_bitarray[num_bits_consumed]
                    state += bit
                num_bits_consumed += 1

        # NOTE: we might have loaded in additional bits not added by the arithmetic encoder
        # (which are present in the encoded_bitarray).
        # This block of code determines the extra bits and subtracts it from num_bits_consumed
        for extra_bits_read in range(self.params.PRECISION):
            state_low = (state >> extra_bits_read) << extra_bits_read
            state_high = state_low + (1 << extra_bits_read)
            if (state_low < low) or (state_high > high):
                break
        num_bits_consumed -= extra_bits_read - 1

        # add back the bits corresponding to the num elements
        num_bits_consumed += self.params.DATA_BLOCK_SIZE_BITS

        return DataBlock(decoded_data_list), num_bits_consumed