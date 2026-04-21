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

        self.program = program

        # TODO: Add implementation for this instead of hard coding
        self.partition_method = "coordinate"
        self.atoms_M0 = "1"
        self.atoms_K0 = "1"
        self.tiles_M1 = "1"
        self.tiles_K1 = "A.cols"
        self.atoms_nnz = self.tiles_num_atoms = "1"

    def create_partitioner(self) -> Statement:
        stmts = SBlock([])
        stmts.add(SAssignObj(
            ANewVar(
                "Partitioner<index_t, type_t, quarks_t>", " partitioner"),
            EFunc("", [AJust(EVar("A"))])))
        if self.partition_method == "coordinate":
            stmts.add(SExpr(
                EMethod(EVar("partitioner"),
                        "partition_coordinate_space",
                        [AJust(EVar(self.atoms_M0)),
                         AJust(EVar(self.atoms_K0)),
                         AJust(EVar(self.tiles_M1)),
                         AJust(EVar(self.tiles_K1))])))
        elif self.partition_method == "position":
            stmts.add(SExpr(
                EMethod(EVar("partitioner"),
                        "partition_atoms_dynamic",
                        [AJust(EVar(self.atoms_nnz)),
                         AJust(EVar(self.tiles_num_atoms))])))
        else:
            raise ValueError(
                "Invalid partition method!, Must be 'coordinate' or 'position")
        stmts.add(SExpr(
            EMethod(EVar("partitioner"), "prepare_gpu", [])))

        return stmts

    def get_partition_method(self) -> str:
        return self.partition_method
