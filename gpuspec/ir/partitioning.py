"""
INSERT LICENSE HERE

Intermediate representation of the partitioning information
"""

from lark.tree import Tree
from typing import cast, Dict, List, Set, Optional

from gpuspec.ir.tensor import Tensor


class Partitioning:
    """
    An abstract representation of the partitioning information
    """

    def __init__(self,
                 tensors: Dict[str, Tensor]) -> None:

        self.tensors = tensors
        self.partitioned_ranks: Dict[str, str] = {}

        self.atom_partition_method = ""
        self.tile_partition_method = ""

    def partition_atoms(self,
                        work_atom: Dict[Tree, List[Tree]]) -> None:
        """
        Apply partition based on given work_atom
        """
        for p_key, p_val in work_atom.items():
            rank_node = p_key.children[0]
            rank = str(
                rank_node.value) if hasattr(
                rank_node,
                'value') else str(rank_node)

            # Validate the partitioned rank
            if rank not in self.__get_all_ranks():
                raise ValueError(
                    "partition_atoms(): Invalid rank to partition!")

            p_type = p_val[0].data
            val_node = p_val[0].children[0].children[0]
            val = str(
                val_node.value) if hasattr(
                val_node,
                'value') else str(val_node)

            if p_type == "uniform_shape":
                self.atom_partition_method = "coordinate"
                self.partitioned_ranks[rank + "0"] = val
            elif p_type == "uniform_occupancy":
                self.atom_partition_method = "position"
                self.partitioned_ranks[rank + "0"] = val
            else:
                raise ValueError(
                    "partition_atoms(): Invalid partition syntax!", p_type)

            # Update tensors' new ranks
            self.__update_ranks(rank, [rank + "1", rank + "0"])

    def partition_tiles(self,
                        work_tile: Dict[Tree, List[Tree]]) -> None:
        """
        Apply partition based on given work_tile
        """
        for p_key, p_val in work_tile.items():
            rank_node = p_key.children[0]
            rank = str(
                rank_node.value) if hasattr(
                rank_node,
                'value') else str(rank_node)

            # Validate the partitioned rank
            if rank not in self.__get_all_ranks():
                raise ValueError(
                    "partition_tiles(): Invalid rank to partition!")

            p_type = p_val[0].data
            val_node = p_val[0].children[0].children[0]
            val = str(
                val_node.value) if hasattr(
                val_node,
                'value') else str(val_node)

            if p_type == "uniform_shape":
                self.tile_partition_method = "coordinate"
                self.partitioned_ranks[rank] = val
            elif p_type == "uniform_occupancy":
                self.tile_partition_method = "position"
                self.partitioned_ranks[rank] = val
            else:
                raise ValueError(
                    "partition_atoms(): Invalid partition syntax!", p_type)

            # Update tensors' new ranks
            self.__update_ranks(rank, [rank[:-1] + "2", rank])

    def __update_ranks(self,
                       rank: str,
                       partitioned_ranks: List[str]) -> None:
        for tensor in self.tensors.values():
            new_ranks = []

            for prev_rank in tensor.get_ranks():
                if prev_rank == rank:
                    new_ranks.extend(partitioned_ranks)
                else:
                    new_ranks.append(prev_rank)

            if new_ranks != tensor.get_ranks():
                tensor.update_ranks(new_ranks)

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

    def get_partitioned_ranks(self) -> Dict[str, str]:
        return self.partitioned_ranks

    def get_atom_partition_method(self) -> str:
        return self.atom_partition_method

    def get_tile_partition_method(self) -> str:
        return self.tile_partition_method
