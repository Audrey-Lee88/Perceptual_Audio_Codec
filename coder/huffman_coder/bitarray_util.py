import bitarray
from bitarray.util import ba2int, int2ba
import numpy as np
from typing import Tuple

BitArray = bitarray.bitarray

def uint_to_bitarray(x: int, bit_width=None) -> BitArray:
    assert isinstance(x, (int, np.integer))
    return int2ba(int(x), length=bit_width)

def bitarray_to_uint(bit_array: BitArray) -> int:
    return ba2int(bit_array)