"""
INSERT LICENSE HERE

Top-level GPULoops program representation
"""

from typing import cast, List, Optional

from gpuspec.gpuloops import *
from gpuspec.parse import *


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

        # TODO: Add implementation for this instead of hard coding
        self.partition_method = "coordinate"
        self.atoms_M0 = "1"
        self.atoms_K0 = "1"
        self.tiles_M1 = "1"
        self.tiles_K1 = "A.cols"
        self.atoms_nnz = self.tiles_num_atoms = "1"

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
