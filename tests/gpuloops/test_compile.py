from copy import deepcopy

# from fibertree_bootstrap import *

import subprocess  # For formatting output file

from gpuspec.parse import *
from gpuspec.trans.gpuloops import GPULoops

spmv_coord_coord_yaml = """
einsum:
  declaration:
    A: [M, K]
    B: [K]
    Z: [M]
  expressions:
    - Z[m] = A[m, k] * B[k]
mapping:
  work_quark: [M, K]
  work_atom:
    M: [uniform_shape(2)]
    K: [uniform_shape(2)]
  work_tile:
    M1: [uniform_shape(1)]
scheduler:
  thread_mapped
"""

'''spmv_coordinate_yaml = """
einsum:
  declaration:
    A: [M, K]
    B: [K]
    Z: [M]
  expressions:
    - Z[m] = A[m, k] * B[k]
mapping:
  work_quark: [M, K]
  work_atom:
    M: [uniform_shape(2)]
    K: [uniform_shape(2)]
  work_tile:
    M1: []
scheduler:
  thread_mapped
"""

spmv_position_yaml = """
einsum:
  declaration:
    A: [M, K]
    B: [K]
    Z: [M]
  expressions:
    - Z[m] = A[m, k] * B[k]
mapping:
  work_quark: [M, K]
  work_atom:
    (M, K): [flatten()]
    MK: [uniform_occupancy(A.4)]
  work_tile:
    MK1: [uniform_shape(2)]
scheduler:
  thread_mapped
"""'''


def test_compile():
    str_yaml = spmv_coord_coord_yaml
    einsum = Einsum.from_str(str_yaml)
    mapping = Mapping.from_str(str_yaml)
    schedulerParser = SchedulerParser.from_str(str_yaml)

    output_file = "./outputs/" + schedulerParser.get_scheduler() \
        + "_edge.cu"
    problem_type = "SpMV"  # SpMV, SpMM, SpGEMM
    N = 10  # Size of rank N
    tracker_enabled = True

    '''print("Work atom: ", mapping.get_work_atom())
    print("Work unit: ", mapping.get_work_unit())
    print("Work tile: ", mapping.get_work_tile())'''

    gpuloops = GPULoops(einsum, mapping, schedulerParser,
                        problem_type, N, tracker_enabled)

    with open(output_file, "w") as f:
        f.write(str(gpuloops))
    subprocess.run(["clang-format", "-i", output_file])

    assert (1)
