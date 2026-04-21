"""
INSERT LICENSE HERE

Translate the Scheduling specification
"""

from gpuspec.ir.program import Program


class Scheduler:
    """
    Generate the GPUloops code for the scheduler information
    """

    def __init__(self,
                 program: Program) -> None:

        self.program = program

    def construct_expr(self) -> str:
        expr = ""
        if self.program.get_scheduler_type() == "thread_mapped":
            expr = (
                "schedule_edge::setup<schedule_edge::algorithms_t::thread_mapped, " f"{
                    str(self.program.get_threads_per_block())}, {
                    str(self.program.get_threads_per_tile())}, " "WorkTile<quarks_t>>")
        elif self.program.get_scheduler_type() == "group_mapped":
            # TODO: Implement this
            pass
        else:  # work_oriented
            # TODO: Implement this
            pass
        return expr
