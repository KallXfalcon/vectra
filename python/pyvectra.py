
import sys
sys.path.append("./python/binding/build")

from typing import Optional
from .python import tensor_ops as to
from enum import Enum
import warnings

class DType(Enum):
    int16 = "int16"
    int32 = "int32"
    float32 = "float32"
    float64 = "float64"


class vectra:
    int16 = DType.int16
    int32 = DType.int32
    float32 = DType.float32
    float64 = DType.float64

class dispatcher:
    @staticmethod
    def _dispatch_add(a, b):
        if a.DType is DType.int16 and b.DType is DType.int16:
            return to.vectra_add_int16(a, b)
        elif a.DType is DType.int32 and b.DType is DType.int32:
            return to.vectra_add_int32(a, b)
        elif a.DType is DType.float32 and b.DType is DType.float32:
            return to.vectra_add_float32(a, b)
        elif a.DType is DType.float64 and b.DType is DType.float64:
            return to.vectra_add_float64(a, b)
        else:
            raise TypeError(f"pyvectra : add() Unsupported DType: {a.DType}")
        
    @staticmethod
    def _dispatch_sub(a, b):
        if a.DType is DType.int16 and b.DType is DType.int16:
            return to.vectra_sub_int16(a, b)
        elif a.DType is DType.int32 and b.DType is DType.int32:
            return to.vectra_sub_int32(a, b)
        elif a.DType is DType.float32 and b.DType is DType.float32:
            return to.vectra_sub_float32(a, b)
        elif a.DType is DType.float64 and b.DType is DType.float64:
            return to.vectra_sub_float64(a, b)
        else:
            raise TypeError(f"pyvectra : sub() Unsupported DType: {a.DType}")
        
    @staticmethod
    def _dispatch_mul(a, b):
        if a.DType is DType.int16 and b.DType is DType.int16:
            return to.vectra_mul_int16(a, b)
        elif a.DType is DType.int32 and b.DType is DType.int32:
            return to.vectra_mul_int32(a, b)
        elif a.DType is DType.float32 and b.DType is DType.float32:
            return to.vectra_mul_float32(a, b)
        elif a.DType is DType.float64 and b.DType is DType.float64:
            return to.vectra_mul_float64(a, b)
        else:
            raise TypeError(f"pyvectra : mul() Unsupported DType: {a.DType}")
        
    @staticmethod
    def _dispatch_div(a, b):
        if a.DType is DType.float32 and b.DType is DType.float32:
            return to.vectra_div_float32(a, b)
        elif a.DType is DType.float64 and b.DType is DType.float64:
            return to.vectra_div_float64(a, b)
        else:
            raise TypeError(f"pyvectra : div() Unsupported DType: {a.DType}")
        
    @staticmethod
    def _dispatch_dot(a, b):
        if a.DType is DType.int16 and b.DType is DType.int16:
            return to.vectra_dot_int16(a, b)
        elif a.DType is DType.int32 and b.DType is DType.int32:
            return to.vectra_dot_int32(a, b)
        elif a.DType is DType.float32 and b.DType is DType.float32:
            return to.vectra_dot_float32(a, b)
        elif a.DType is DType.float64 and b.DType is DType.float64:
            return to.vectra_dot_float64(a, b)
        else:
            raise TypeError(f"pyvectra : dot() Unsupported DType: {a.DType}")
        
    @staticmethod
    def _dispatch_max(x):
        if x.DType is DType.int16:
            return to.vectra_max_int16(x)
        elif x.DType is DType.int32:
            return to.vectra_max_int32(x)
        elif x.DType is DType.float32:
            return to.vectra_max_float32(x)
        elif x.DType is DType.float64:
            return to.vectra_max_float64(x)
        else:
            raise TypeError(f"pyvectra : dot() Unsupported DType: {x.DType}")
        
    @staticmethod
    def _dispatch_flatten(x):
        if x.DType is DType.int16:
            return to.vectra_flatten_int16(x)
        elif x.DType is DType.int32:
            return to.vectra_flatten_int32(x)
        elif x.DType is DType.float32:
            return to.vectra_flatten_float32(x)
        elif x.DType is DType.float64:
            return to.vectra_flatten_float64(x)
        else:
            raise TypeError(f"pyvectra : flatten() Unsupported DType: {x.DType}")
        
    @staticmethod
    def _dispatch_sum(x):
        if x.DType is DType.int16:
            return to.vectra_sum_int16(x)
        elif x.DType is DType.int32:
            return to.vectra_sum_int32(x)
        elif x.DType is DType.float32:
            return to.vectra_sum_float32(x)
        elif x.DType is DType.float64:
            return to.vectra_sum_float64(x)
        else:
            raise TypeError(f"pyvectra : sum() Unsupported DType: {x.DType}")
        
    @staticmethod
    def _dispatch_exp(x):
        if x.DType is DType.int16:
            return to.vectra_exp_int16(x)
        elif x.DType is DType.int32:
            return to.vectra_exp_int32(x)
        elif x.DType is DType.float32:
            return to.vectra_exp_float32(x)
        elif x.DType is DType.float64:
            return to.vectra_exp_float64(x)
        else:
            raise TypeError(f"pyvectra : exp() Unsupported DType: {x.DType}")

    




def full(shape, value, dtype=None):
    if isinstance(shape, list):
        raise TypeError("pyvectra : full() shape must be a tuple, not a list")
    if not isinstance(shape, tuple):
        raise TypeError(
            f"pyvectra : full() shape must be a tuple, got {DType(shape).__name__}"
        )
    if len(shape) > 32:
        raise ValueError("pyvectra : full() maximum supported dimension is 32")

    if dtype is None:
        if isinstance(value, int):
            dtype = DType.int32
        elif isinstance(value, float):
            dtype = DType.float32
        else:
            raise TypeError(
                "pyvectra : full() cannot infer DType from value "
                f"of DType {DType(value).__name__}"
            )

    if dtype is DType.int16:
        return to.vectra_full_int16(shape, value)
    elif dtype is DType.float32:
        return to.vectra_full_int32(shape, value)
    elif dtype is DType.float32:
        return to.vectra_full_float32(shape, value) 
    elif dtype is DType.float64:
        return to.vectra_full_float64(shape, value)
    else:
        raise TypeError(f"pyvectra : full() Unsupported DType: {dtype}")


def zeros(shape, dtype: str = None):
    if isinstance(shape, list):
        raise TypeError("pyvectra : zeros() shape must be tuple, not a list")
    if not isinstance(shape, tuple):
        raise TypeError(f"pyvectra : zeros() shape must be a tuple, got {DType(shape).__name__}")
    if len(shape) > 32:
        raise ValueError("pyvectra : zeros() maximum supported dimension is 32")

    if dtype is None:
        dtype = DType.float32

    if dtype == DType.int16:
        return to.vectra_zeros_int16(shape)
    elif dtype == DType.int32:
        return to.vectra_zeros_int32(shape)
    elif dtype == DType.float32:
        return to.vectra_zeros_float32(shape)
    elif dtype == DType.float64:
        return to.vectra_zeros_float64(shape)
    else:
        raise TypeError(f"pyvectra : zeros() Unsupported DType: {dtype}")

def ones(shape, dtype: str = None):
    if isinstance(shape, list):
        raise TypeError("pyvectra : ones() shape must be tuple, not a list")
    if not isinstance(shape, tuple):
        raise TypeError(f"pyvectra : ones() shape must be a tuple, got {DType(shape).__name__}")
    if len(shape) > 32:
        raise ValueError("pyvectra : ones() maximum supported dimension is 32")

    if dtype is None:
        dtype = DType.float32

    if dtype == DType.int16:
        return to.vectra_ones_int16(shape)
    elif dtype == DType.int32:
        return to.vectra_ones_int32(shape)
    elif dtype == DType.float32:
        return to.vectra_ones_float32(shape)
    elif dtype == DType.float64:
        return to.vectra_ones_float64(shape)
    else:
        raise TypeError(f"pyvectra : ones() Unsupported DType: {dtype}")

def twos(shape, dtype: str = None):
    if isinstance(shape, list):
        raise TypeError("pyvectra : twos() shape must be tuple, not a list")
    if not isinstance(shape, tuple):
        raise TypeError(f"pyvectra : twos() shape must be a tuple, got {DType(shape).__name__}")
    if len(shape) > 32:
        raise ValueError("pyvectra : twos() maximum supported dimension is 32")

    if dtype is None:
        dtype = DType.float32

    if dtype == DType.int16:
        return to.vectra_twos_int16(shape)
    elif dtype == DType.int32:
        return to.vectra_twos_int32(shape)
    elif dtype == DType.float32:
        return to.vectra_twos_float32(shape)
    elif dtype == DType.float64:
        return to.vectra_twos_float64(shape)
    else:
        raise TypeError(f"pyvectra : twos() Unsupported DType: {dtype}")

def rand(shape, dtype: str = None):
    if isinstance(shape, (list, int, float)):
        raise TypeError("pyvectra : rand() shape must be tuple, not int, float, or list")
    if not isinstance(shape, tuple):
        raise TypeError(f"pyvectra : rand() shape must be a tuple, got {DType(shape).__name__}")
    if len(shape) > 32:
        raise ValueError("pyvectra : rand() maximum supported dimension is 32")

    if dtype is None:
        dtype = DType.float32

    if dtype in (DType.int16, DType.int32):
        raise TypeError("pyvectra : rand() does not support int types right now!")
    elif dtype == DType.float32:
        return to.vectra_rand_float32(shape)
    elif dtype == DType.float64:
        return to.vectra_rand_float64(shape)
    else:
        raise TypeError(f"pyvectra : rand() Unsupported DType: {dtype}")

def randn(shape, dtype: str = None):
    if isinstance(shape, (list, int, float)):
        raise TypeError("pyvectra : randn() shape must be tuple, not int, float, or list")
    if not isinstance(shape, tuple):
        raise TypeError(f"pyvectra : randn() shape must be a tuple, got {DType(shape).__name__}")
    if len(shape) > 32:
        raise ValueError("pyvectra : randn() maximum supported dimension is 32")

    if dtype is None:
        dtype = DType.float32

    if dtype in (DType.int16, DType.int32):
        raise TypeError("pyvectra : randn() does not support int types right now!")
    elif dtype == DType.float32:
        return to.vectra_randn_float32(shape)
    elif dtype == DType.float64:
        return to.vectra_randn_float64(shape)
    else:
        raise TypeError(f"pyvectra : randn() Unsupported DType: {dtype}")
    
def add(a, b):
    if isinstance(a, list) or isinstance(b, list):
        raise TypeError("pyvectra : add() Shape not supported it was a list. Must be a tuple")
    if not isinstance(a, tuple) or not isinstance(b, tuple):
        raise TypeError("pyvectra : add() func only support tuple!")
    return dispatcher._dispatch_add(a, b)

def sub(a, b):
    if isinstance(a, list) or isinstance(b, list):
        raise TypeError("pyvectra : sub() Shape not supported it was a list. Must be a tuple")
    if not isinstance(a, tuple) or not isinstance(b, tuple):
        raise TypeError("pyvectra : sub() func only support tuple!")
    return dispatcher._dispatch_sub(a, b)

def mul(a, b):
    if isinstance(a, list) or isinstance(b, list):
        raise TypeError("pyvectra : mul() Shape not supported it was a list. Must be a tuple")
    if not isinstance(a, tuple) or not isinstance(b, tuple):
        raise TypeError("pyvectra : mul() func only support tuple!")
    return dispatcher._dispatch_mul(a, b)

def div(a, b):
    if b == 0:
        raise ValueError("pyvectra : div() cannot divide by zero")
    if isinstance(a, list) or isinstance(b, list):
        raise TypeError("pyvectra : div() Shape not supported it was a list. Must be a tuple")
    if not isinstance(a, tuple) or not isinstance(b, tuple):
        raise TypeError("pyvectra : div() func only support tuple!")
    return dispatcher._dispatch_div(a, b)

def dot(a, b):
    if isinstance(a, list) or isinstance(b, list):
        raise TypeError("pyvectra : dot() Shape not supported for list. Must be a tuple")
    if not isinstance(a, tuple) or not isinstance(b, tuple):
        raise TypeError("pyvectra : dot() must be a tuple")
    return dispatcher._dispatch_dot(a, b)

def max(x):
    if isinstance(x, list) or not isinstance(x, tuple):
        raise TypeError("pyvectra : max() Shape must be a tuple")
    return dispatcher._dispatch_max(x)

def flatten(x):
    if isinstance(x, list) or not isinstance(x, tuple):
        raise TypeError("pyvectra : flatten() Shape must be a tuple")
    return dispatcher._dispatch_flatten(x)

def sum(x):
    if isinstance(x, list) or not isinstance(x, tuple):
        raise TypeError("pyvectra : sum() Shape must be a tuple")
    return dispatcher._dispatch_sum(x)

def exp(x):
    if isinstance(x, list) or not isinstance(x, tuple):
        raise TypeError("pyvectra : exp() Shape must be a tuple")
    return dispatcher._dispatch_exp(x)