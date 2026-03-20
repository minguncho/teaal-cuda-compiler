from copy import deepcopy

# from fibertree_bootstrap import *

from gpuspec.parse import *
from gpuspec.trans.hifiber import HiFiber
from gpuspec.trans.gpuloops import GPULoops

yaml = """
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
    # After First Partitioning:
    # M1: Size of each tile = TILE_SIZES
    # M0: Number of tiles = NUM_TILES

    # After Second Partitioning:
    # M2: TILE_SIZE
    # M1: Number of tiles each thread will be working on
    # M0: NUM_THREADS
  spacetime:
    Z:
      space: [M0]
      time: [M2, M1, K]
scheduler:
  thread-mapped
"""


def test_compile():
    str_yaml = yaml

    einsum = Einsum.from_str(str_yaml)
    mapping = Mapping.from_str(str_yaml)
    scheduler = Scheduler.from_str(str_yaml)
    arch = Architecture.from_str(str_yaml)
    bindings = Bindings.from_str(str_yaml)
    format_ = Format.from_str(str_yaml)

    output_path = "./outputs/"
    file_names = {
        "loops_fname": "output.cu",
    }

    gpuloops = GPULoops(
        einsum,
        mapping,
        scheduler,
        file_names,
        arch,
        bindings,
        format_)
    files = gpuloops.get_files()

    with open(output_path + file_names["loops_fname"], "w") as f:
        f.write(files["loops_fname"])

    '''hifiber = HiFiber(einsum, mapping, arch, bindings, format_)

    with open(output_path + "output_hifiber.txt", "w") as f:
        f.write(str(hifiber))'''
    assert (1)
