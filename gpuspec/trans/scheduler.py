"""
INSERT LICENSE HERE

Translate the Scheduling specification
"""

from gpuspec.gpuloops import *
from gpuspec.ir.program import Program


class Scheduler:
    """
    Generate the GPUloops code for the scheduler information
    """

    def __init__(self,
                 program: Program) -> None:

        self.program = program

    def create_scheduler(self) -> Statement:
        stmts = SBlock([])

        if self.program.get_scheduler_type() == "thread_mapped":
            stmts.add(
                SAssignTypename(AVar("setup_t"),
                                EVar(
                                    "schedule_edge::setup<schedule_edge::algorithms_t::thread_mapped, "
                                    f"{str(self.program.get_threads_per_block())}, "
                                    f"{str(self.program.get_threads_per_tile())}, "
                                    "WorkTile<quarks_t>>")))
            stmts.add(SAssignObj(
                ANewVar(
                    "setup_t", "config"),
                EFunc("",
                      [AJust(EMethod(EMethod(EMethod(EVar("partitioner"), "get_work_tiles", []),
                                             "data", []), "get", [])),
                       AJust(EMethod(EVar("partitioner"), "get_num_tiles", []))])))
        elif self.program.get_scheduler_type() == "group_mapped":
            # TODO: Implement this
            pass
        else:  # work_oriented
            # TODO: Implement this
            pass
        return stmts
