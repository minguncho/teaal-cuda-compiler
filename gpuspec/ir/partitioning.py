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
                 ranks: List[str]) -> None:

        self.ranks = ranks
        self.partitioned_ranks: Dict[str, str] = {}

        self.atom_partition_method = ""
        self.tile_partition_method = ""
        self.flatten_ranks = False

    def partition_atoms(self,
                        work_atom: Dict[Tree, List[Tree]]) -> None:
        """
        Apply partition based on given work_atom
        """
        for p_key, p_val in work_atom.items():

            if len(p_key.children) == 1:
                rank_node = p_key.children[0]
                rank = str(
                    rank_node.value) if hasattr(
                    rank_node,
                    'value') else str(rank_node)

                # Validate the partitioned rank
                if rank not in self.ranks:
                    raise ValueError(
                        "partition_atoms(): Invalid rank to partition!", rank)

                p_type = p_val[0].data

                if p_type == "uniform_shape":
                    val_node = p_val[0].children[0].children[0]
                    val = str(
                        val_node.value) if hasattr(
                        val_node,
                        'value') else str(val_node)

                    if self.atom_partition_method == "position":
                        raise ValueError(
                            "partition_atoms(): Cannot partition atoms with both by uniform_shape and uniform_occupancy!")

                    self.atom_partition_method = "coordinate"
                    self.partitioned_ranks[rank + "0"] = val
                elif p_type == "uniform_occupancy":
                    val_node = p_val[0].children[1].children[0]
                    val = str(
                        val_node.value) if hasattr(
                        val_node,
                        'value') else str(val_node)

                    if self.atom_partition_method == "coordinate":
                        raise ValueError(
                            "partition_atoms(): Cannot partition atoms with both by uniform_shape and uniform_occupancy!")

                    self.atom_partition_method = "position"
                    self.partitioned_ranks[rank + "0"] = val
                else:
                    raise ValueError(
                        "partition_atoms(): Invalid partition syntax! ", p_type)

                self.__update_ranks([rank], [rank + "1", rank + "0"])

            elif len(p_key.children) == 2:  # Flatten 2 ranks
                rank_node1 = p_key.children[0]
                rank1 = str(
                    rank_node1.value) if hasattr(
                    rank_node1,
                    'value') else str(rank_node1)

                rank_node2 = p_key.children[1]
                rank2 = str(
                    rank_node2.value) if hasattr(
                    rank_node2,
                    'value') else str(rank_node2)

                # Validate the partitioned rank
                if rank1 not in self.ranks:
                    raise ValueError(
                        "partition_atoms(): Invalid rank to partition!", rank1)

                if rank2 not in self.ranks:
                    raise ValueError(
                        "partition_atoms(): Invalid rank to partition!", rank2)

                new_rank = rank1 + rank2
                self.atom_partition_method = "position"
                self.flatten_ranks = True
                self.__update_ranks(
                    [rank1, rank2], [new_rank])

            else:
                raise ValueError(
                    "partition_atoms(): Current version only supports flattening 2 ranks ", len(
                        p_key.children))

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
            if rank not in self.ranks:
                raise ValueError(
                    "partition_tiles(): Invalid rank to partition!", rank)

            p_type = p_val[0].data

            if p_type == "uniform_shape":
                val_node = p_val[0].children[0].children[0]
                val = str(
                    val_node.value) if hasattr(
                    val_node,
                    'value') else str(val_node)

                self.tile_partition_method = "coordinate"
                self.partitioned_ranks[rank] = val
            elif p_type == "uniform_occupancy":
                val_node = p_val[0].children[1].children[0]
                val = str(
                    val_node.value) if hasattr(
                    val_node,
                    'value') else str(val_node)

                self.tile_partition_method = "position"
                self.partitioned_ranks[rank] = val
            else:
                raise ValueError(
                    "partition_atoms(): Invalid partition syntax!", p_type)

            # Update tensors' new ranks
            self.__update_ranks([rank], [rank[:-1] + "2", rank])

    def __update_ranks(self,
                       target_ranks: List[str],
                       partitioned_ranks: List[str]) -> None:

        new_ranks = []

        for prev_rank in self.ranks:
            if prev_rank not in target_ranks:
                new_ranks.append(prev_rank)
        new_ranks.extend(partitioned_ranks)
        self.ranks = new_ranks

    """
        Get functions
    """

    def get_partitioned_ranks(self) -> Dict[str, str]:
        return self.partitioned_ranks

    def get_atom_partition_method(self) -> str:
        return self.atom_partition_method

    def get_tile_partition_method(self) -> str:
        return self.tile_partition_method

    def get_flatten_ranks(self) -> bool:
        return self.flatten_ranks
