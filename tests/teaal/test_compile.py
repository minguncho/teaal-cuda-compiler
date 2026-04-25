from copy import deepcopy

from teaal.parse import *
from teaal.trans.hifiber import HiFiber

spmv_yaml = """
einsum:
  declaration:
    A: [M, K]
    B: [K]
    Z: [M]
  expressions:
    - Z[m] = A[m, k] * B[k]
mapping:
  rank-order:
    A: [M, K]
    B: [K]
    Z: [M]
  partitioning:
    Z:
      M: [uniform_shape(NUM_TILES), uniform_shape(NUM_THREADS)]
  loop-order:
    Z: [M2, M1, M0, K]
  spacetime:
    Z:
      space: [M0]
      time: [M2, M1, K]
"""

spmv_work_yaml = """
einsum:
  declaration:
    A: [M, K]
    B: [K]
    Z: [M]
  expressions:
    - Z[m] = A[m, k] * B[k]
mapping:
  rank-order:
    A: [M, K]
    B: [K]
    Z: [M]
  partitioning:
    Z:
      (M, K): [flatten()]
      MK: [uniform_occupancy(A.WORK_PER_THREAD)]
  loop-order:
    Z: [MK1, MK0]
    # MK1: Number of iterations to process all NNZ in parallel
    # MK0: Number of NNZ assigned to each thread = WORK_PER_THREAD
  spacetime:
    Z:
      space: [MK1]
      time: [MK0]
"""

spmm_yaml = """
einsum:
  declaration:
    A: [M, K]
    B: [K, N]
    Z: [M, N]
  expressions:
    - Z[m, n] = A[m, k] * B[k, n]
mapping:
  rank-order:
    A: [M, K]
    B: [K, N]
    Z: [M, N]
  partitioning:
    Z:
      M: [uniform_shape(2)]
      K: [uniform_shape(2)]
  loop-order:
    Z: [M1, M0, N, K1, K0]
  spacetime:
    Z:
      space: [M1, K0]
      time: [M0, K1, N]
"""


def test_compile():
    str_yaml = spmm_yaml
    einsum = Einsum.from_str(str_yaml)
    mapping = Mapping.from_str(str_yaml)
    arch = Architecture.from_str(str_yaml)
    bindings = Bindings.from_str(str_yaml)
    format_ = Format.from_str(str_yaml)

    hifiber = HiFiber(einsum, mapping, arch, bindings, format_)

    output_file = "./outputs/" + "teaal_output.txt"

    with open(output_file, "w") as f:
        f.write(str(hifiber))

    assert (1)
