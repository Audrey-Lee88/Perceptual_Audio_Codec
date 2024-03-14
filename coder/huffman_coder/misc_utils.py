import typing
import numpy as np




def is_power_of_two(x: float):
    exp = float(np.log2(x))
    return exp.is_integer()