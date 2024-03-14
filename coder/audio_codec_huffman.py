import sys
sys.path.append('huffman_coder')

from huffman_coder.prob_dist import ProbabilityDist
from huffman_coder.huffman_coding import HuffmanEncoder, HuffmanDecoder
from huffman_coder.prob_dist import *
# from huffman_coder.bitarray_util import BitArray, uint_to_bitarray, bitarray_to_uint
# import pickle
# from huffman_coder.data_block import DataBlock
# import numpy as np
   
class huffman_audio:
    def __init__(self, mant_dict):
        mant_freq = Frequencies(mant_dict)
        self.prob_dist = mant_freq.get_prob_dist()

    def audio_huffman_encode(self,mant_vals):
        """
        Huffman encodes the quantized spectra using the Stanford Compression Library's class
        """
        huff_en = HuffmanEncoder(self.prob_dist)
        encoded_bitarray = huff_en.encode_symbol(mant_vals)
        return encoded_bitarray

    def audio_huffman_decode(self,encoded_bit,):
        """
        Huffman decodes the encoded bitarray using the Stanford Compression Library's class
        to get back the quantized spectra
        """
        huff_dec = HuffmanDecoder(self.prob_dist)
        decoded_block, num_bits_consumed = huff_dec.decode_symbol(encoded_bit)
        return decoded_block, num_bits_consumed

# def encode_prob_dist(prob_dist: ProbabilityDist) -> BitArray:
#     """Encode a probability distribution as a bit array

#     Args:
#         prob_dist (ProbabilityDist): probability distribution over 0, 1, 2, ..., 255
#             (note that some probabilities might be missing if they are 0).

#     Returns:
#         BitArray: encoded bit array
#     """
#     # pickle prob dist and convert to bytes
#     pickled_bits = BitArray()
#     pickled_bits.frombytes(pickle.dumps(prob_dist))
#     len_pickled = len(pickled_bits)
#     # encode length of pickle
#     length_bitwidth = 32
#     length_encoding = uint_to_bitarray(len_pickled, bit_width=length_bitwidth)

#     encoded_probdist_bitarray = length_encoding + pickled_bits

#     return encoded_probdist_bitarray

# def decode_prob_dist(bitarray: BitArray):
#     """Decode a probability distribution from a bit array

#     Args:
#         bitarray (BitArray): bitarray encoding probability dist followed by arbitrary data

#     Returns:
#         prob_dit (ProbabilityDist): the decoded probability distribution
#         num_bits_read (int): the number of bits read from bitarray to decode probability distribution
#     """
#     # first read 32 bits from start to get the length of the pickled sequence
#     length_bitwidth = 32
#     length_encoding = bitarray[:length_bitwidth]
#     len_pickled = bitarray_to_uint(length_encoding)
#     # bits to bytes
#     pickled_bytes = bitarray[length_bitwidth: length_bitwidth + len_pickled].tobytes()

#     prob_dist = pickle.loads(pickled_bytes)
#     num_bits_read = length_bitwidth + len_pickled
#     return prob_dist, num_bits_read