from copy import deepcopy

from teaal.parse import *
from teaal.trans.hifiber import HiFiber

teaal_yaml = """
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


def test_compile():

    einsum = Einsum.from_str(teaal_yaml)
    mapping = Mapping.from_str(teaal_yaml)
    arch = Architecture.from_str(teaal_yaml)
    bindings = Bindings.from_str(teaal_yaml)
    format_ = Format.from_str(teaal_yaml)

    hifiber = HiFiber(einsum, mapping, arch, bindings, format_)

    output_file = "./outputs/" + "teaal_output.txt"

    with open(output_file, "w") as f:
        f.write(str(hifiber))

    assert (1)
