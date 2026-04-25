"""
INSERT LICENSE HERE

Translate the partitioning specification
"""

from gpuspec.gpuloops import *
from gpuspec.ir.program import Program


class Partitioner:
    """
    Generate the GPUloops code for the partitioner information
    """

    def __init__(self,
                 program: Program) -> None:

        self.partitioning = program.get_partitioning()
        self.atoms_M0 = ""
        self.atoms_K0 = ""
        self.tiles_M1 = ""
        self.tiles_K1 = ""

    def create_partitioner(self) -> Statement:
        stmts = SBlock([])
        stmts.add(SAssignObj(
            ANewVar(
                "Partitioner<index_t, type_t, quarks_t>", " partitioner"),
            EFunc("", [AJust(EVar("A"))])))

        # Generate for atom partitioning
        if self.partitioning.get_atom_partition_method() == "":
            raise ValueError(
                "Empty atom partition method!")

        partitioned_ranks = self.partitioning.get_partitioned_ranks()
        if self.partitioning.get_atom_partition_method() == "coordinate":
            self.atoms_M0 = "A.rows"
            self.atoms_K0 = "A.cols"

            if "M0" in partitioned_ranks:
                self.atoms_M0 = partitioned_ranks["M0"]

            if "K0" in partitioned_ranks:
                self.atoms_K0 = partitioned_ranks["K0"]

            stmts.add(SExpr(
                EMethod(EVar("partitioner"),
                        "partition_atoms_coordinate_space",
                        [AJust(EVar(self.atoms_M0)),
                         AJust(EVar(self.atoms_K0))])))
        elif self.partitioning.get_atom_partition_method() == "position":
            atoms_nnz = "1"

            print("Current version does not support position partitioned method!")

            stmts.add(SExpr(
                EMethod(EVar("partitioner"),
                        "partition_atoms_position_space",
                        [AJust(EVar(atoms_nnz))])))
        else:
            raise ValueError(
                "Invalid atom partition method!, Must be 'coordinate' or 'position")

        # Generate for tile partitioning
        if self.partitioning.get_tile_partition_method() == "coordinate":
            self.tiles_M1 = "(A.rows" + " + " + self.atoms_M0 + \
                " - 1)" + " / " + self.atoms_M0
            self.tiles_K1 = "(A.cols" + " + " + self.atoms_K0 + \
                " - 1)" + " / " + self.atoms_K0

            if "M1" in partitioned_ranks:
                self.tiles_M1 = partitioned_ranks["M1"]

            if "K1" in partitioned_ranks:
                self.tiles_K1 = partitioned_ranks["K1"]

            stmts.add(SExpr(
                EMethod(EVar("partitioner"),
                        "partition_tiles_coordinate_space",
                        [AJust(EVar(self.tiles_M1)),
                         AJust(EVar(self.tiles_K1))])))
        elif self.partitioning.get_tile_partition_method() == "position":
            tiles_num_atoms = "1"

            print("Current version does not support position partitioned method!")

            stmts.add(SExpr(
                EMethod(EVar("partitioner"),
                        "partition_tiles_position_space",
                        [AJust(EVar(tiles_num_atoms))])))
        else:
            raise ValueError(
                "Invalid atom partition method!, Must be 'coordinate' or 'position")

        stmts.add(SExpr(
            EMethod(EVar("partitioner"), "prepare_gpu", [])))

        return stmts
