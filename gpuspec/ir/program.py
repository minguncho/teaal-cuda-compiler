"""
INSERT LICENSE HERE

Top-level GPULoops program representation
"""

from lark.tree import Tree
from typing import cast, List, Set, Optional

from gpuspec.gpuloops import *
from gpuspec.parse import *
from gpuspec.ir.tensor import Tensor
from gpuspec.ir.partitioning import Partitioning


class Program:
    """
    Top-level GPULoops program representation
    """

    def __init__(self,
                 einsum: Einsum,
                 mapping: Mapping,
                 schedulerParser: SchedulerParser,
                 problem_type: str,
                 N: int,
                 tracker_enabled: Optional[bool] = False,
                 block_size: Optional[str] = None,
                 grid_size: Optional[str] = None) -> None:
        """
        Construct the metadata
        """
        # Initialize with arguments
        self.einsum = einsum
        self.mapping = mapping
        self.schedulerParser = schedulerParser
        self.problem_type = problem_type
        self.N = N
        self.tracker_enabled = tracker_enabled

        # Default kernel parameters
        self.block_size = "128"
        self.grid_size = "(partitioner.get_num_tiles() + block_size - 1) / block_size"
        self.kernel_name = self.schedulerParser.get_scheduler() + "_edge"

        if block_size:
            self.block_size = block_size
        if grid_size:
            self.grid_size = grid_size

        # Loops parameters
        self.threads_per_block = 1
        self.threads_per_tile = 1

        # Check for length of einsum expressions
        if len(einsum.get_expressions()) != 1:
            raise ValueError(
                "Current version only supports 1 Einsum expression!")

        # Typenames and default type
        self.typenames = {
            "setup_t": None,
            "index_t": "int",
            "offset_t": "int",
            "type_t": "float",
            "quarks_t": "std::size_t"
        }
        self.gpu_typenames = ["setup_t", "index_t", "type_t"]

        # Get all tensors
        self.tensors = {}
        declaration = self.einsum.get_declaration()
        for ten_name in declaration:
            tensor = Tensor(ten_name, declaration[ten_name])
            self.tensors[tensor.root_name()] = tensor

        # Get all einsums
        self.einsums = []
        for expr in self.einsum.get_expressions():
            self.einsums.append(
                str(next(expr.find_data("output")).children[0]))

        # Apply partitioning
        self.partitioning = Partitioning(self.tensors)
        self.partitioning.partition_atoms(self.mapping.get_work_atom())
        self.partitioning.partition_tiles(self.mapping.get_work_tile())

    def __get_all_ranks(self) -> Set[str]:
        """
        Get the set of all ranks of current tensors
        """

        # Get all ranks
        ranks = set()
        for tensor in self.tensors.values():
            ranks.update(tensor.get_ranks())

        return ranks

    """
        Get functions
    """

    def get_scheduler_type(self) -> str:
        return self.schedulerParser.get_scheduler()

    def get_problem_type(self) -> str:
        return self.problem_type

    def get_N(self) -> int:
        return self.N

    def get_tracker_enabled(self) -> bool | None:
        return self.tracker_enabled

    def get_block_size(self) -> str:
        return self.block_size

    def get_grid_size(self) -> str:
        return self.grid_size

    def get_kernel_name(self) -> str:
        return self.kernel_name

    def get_threads_per_block(self) -> int:
        return self.threads_per_block

    def get_threads_per_tile(self) -> int:
        return self.threads_per_tile

    def get_typenames(self) -> Dict[str, Optional[str]]:
        return self.typenames

    def get_gpu_typenames(self) -> List[str]:
        return self.gpu_typenames

    def get_partitioning(self) -> Partitioning:
        return self.partitioning
