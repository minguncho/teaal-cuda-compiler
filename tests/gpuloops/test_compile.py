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

    output_file = "./outputs/" + scheduler.get_scheduler() \
        + "_edge.cu"
    problem_type = "SpMV"  # SpMV, SpMM, SpGEMM
    N = 10  # Size of rank N
    tracker_enabled = True

    '''print("Work atom: ", mapping.get_work_atom())
    print("Work unit: ", mapping.get_work_unit())
    print("Work tile: ", mapping.get_work_tile())'''

    gpuloops = GPULoops(einsum, mapping, scheduler,
                        problem_type, N, tracker_enabled)

    with open(output_file, "w") as f:
        f.write(str(gpuloops))
    subprocess.run(["clang-format", "-i", output_file])

    assert (1)
