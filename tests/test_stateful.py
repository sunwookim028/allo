import allo
from allo.ir.types import int32, float32, Stateful

# Scalar stateless
def test_stateless_scalar(x: int32) -> int32:
    acc: int32 = 0
    acc = acc + x
    return acc

# Scalar stateful
def test_stateful_scalar(x: int32) -> int32:
    acc: Stateful[int32] = 0
    acc = acc + x
    return acc

# Array stateless
def test_stateless_array(x: float32) -> float32:
    buffer: float32[10] = 0.0
    buffer[0] = buffer[1] + buffer[2]
    return buffer[0]

# Array stateful
def test_stateful_array(x: float32) -> float32:
    buffer: Stateful[float32[10]] = 0.0
    buffer[0] = buffer[1] + buffer[2]
    return buffer[0]

s1 = allo.customize(test_stateless_scalar)
print(s1.module)  # Should show alloc

s2 = allo.customize(test_stateful_scalar)
print(s2.module)  # Should show memref.global

s3 = allo.customize(test_stateless_array)
print(s3.module)  # Should show alloc with shape

s2 = allo.customize(test_stateful_array)
print(s2.module)  # Should show memref.global with shape
