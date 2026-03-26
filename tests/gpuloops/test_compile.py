from copy import deepcopy

# from fibertree_bootstrap import *

import subprocess  # For formatting output file

from gpuspec.parse import *
from gpuspec.trans.gpuloops import GPULoops

# Referece, do not actually use the teaal_yaml as an input
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

loops_yaml = """
einsum:
  declaration:
    A: [M, K]
    B: [K]
    Z: [M]
  expressions:
    - Z[m] = A[m, k] * B[k]
mapping:
  work_atom: [M, K]
  work_unit: [M, K]
  work_tile: [M]
scheduler:
  thread_mapped
"""


def test_compile():

    einsum = Einsum.from_str(loops_yaml)
    mapping = Mapping.from_str(loops_yaml)
    scheduler = Scheduler.from_str(loops_yaml)

    output_file = "./outputs/output.cu"
    problem_type = "SpMV"  # SpMV, SpMM, SpGEMM
    N = 10  # Size of rank N

    '''print("Work atom: ", mapping.get_work_atom())
    print("Work unit: ", mapping.get_work_unit())
    print("Work tile: ", mapping.get_work_tile())'''

    gpuloops = GPULoops(einsum, mapping, scheduler,
                        problem_type, N)

    with open(output_file, "w") as f:
        f.write(str(gpuloops))
    subprocess.run(["clang-format", "-i", output_file])

    assert (1)


"""Notes:

    Suppose solving SpMV, so ranks are [M, K]

    Option for work_atom:
    1. [M, K]

    Options for work_unit:
    1. [M] --> Each work unit is a row of A.
    2. [K] --> Each work unit is a column of A.
    3. [M, K] --> Each work unit is a NZ of A.
    4. M: uniform_shape(M0) --> Each work unit is a M0 number of rows of A.
    5. K: uniform_shape(K0) --> Each work unit is a K0 number of columns of A.
    6. M: uniform_shape(M0), K: uniform_shape(K0) --> Each work unit is a M0 number of rows and K0 number of columns of A (static tile).
    7. (M, K): [flatten()], MK: uniform_occupancy(MK0) --> Each work unit is MK0 number of entries of A (dynamic tile, DRT).

    Options for work_tile:
    1. [M]
"""
