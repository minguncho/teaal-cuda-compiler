"""
INSERT LICENSE HERE

Translate an Einsum to the corresponding GPULoops code
"""

from typing import cast, List, Optional

from gpuspec.gpuloops import *
from gpuspec.parse import *
from gpuspec.ir.program import Program
from gpuspec.trans.gpukernel import GPUKernel
from gpuspec.trans.hostfn import HostFn
from gpuspec.trans.partitioner import Partitioner
from gpuspec.trans.scheduler import Scheduler


class GPULoops:
    """
    Translate a given Einsum into the corresponding GPULoops code
    """

    def __init__(
            self,
            einsum: Einsum,
            mapping: Mapping,
            schedulerParser: SchedulerParser,
            problem_type: str,
            N: int,
            tracker_enabled: Optional[bool] = False,
            block_size: Optional[str] = None,
            grid_size: Optional[str] = None) -> None:
        """
        Perform the Einsum to GPULoops translation
        """

        self.program = Program(einsum, mapping, schedulerParser,
                               problem_type, N, tracker_enabled,
                               block_size, grid_size)

        self.gpuloops = self.__generate_gpuloops()

    def __generate_gpuloops(self) -> Statement:
        """
        Main framework for loops file
        """

        # Construct translator objects
        scheduler = Scheduler(self.program)
        partitioner = Partitioner(self.program)
        gpuKernel = GPUKernel(self.program)
        hostFn = HostFn(self.program, scheduler, partitioner)

        stmts = SBlock([])

        # Step 1: Add file description
        comments = [
            "/**",
            " * Generated Loops code",
            "*/"
        ]
        stmts.add(self.__add_file_description(comments))

        # Step 2: Add preprocessor directives (#include, #define, etc)
        pragmas: list[str] = []
        includes = ["\"helpers.hxx\"",
                    "<loops/container/formats.hxx>",
                    "<loops/container/vector.hxx>",
                    "<loops/memory.hxx>",
                    "<loops/schedule_edge.hxx>",
                    "<loops/util/launch.hxx>",
                    "<loops/util/device.hxx>",
                    "<loops/util/partitioner.hxx>",
                    "<loops/util/tracker.hxx>",
                    "<iostream>"]
        macros: list[str] = []

        stmts.add(self.__add_preproc_dir(pragmas, includes, macros))

        # Step 3: Add namespaces
        namespaces = ["loops"]
        stmts.add(self.__add_namespace(namespaces))

        # Step 4: Add GPU kernel
        stmts.add(gpuKernel.add_gpu_kernel())

        # Step 5: Add host main function
        stmts.add(hostFn.add_main_fn())

        return stmts

    def __add_file_description(self, comments: List[str]) -> Statement:
        """
        Add file description
        """
        stmts = SBlock([])

        for comment in comments:
            stmts.add(SDecl(DDefn(comment)))
        stmts.add(SNewEmptyLine())  # Adding a new empty line

        return stmts

    def __add_preproc_dir(self,
                          pragmas: List[str],
                          includes: List[str],
                          macros: List[str]) -> Statement:
        """
        Add preprocessor directives
        """
        stmts = SBlock([])

        # Add pragmas
        for pragma in pragmas:
            stmts.add(SDecl(DDefn(pragma)))
        if pragmas:
            stmts.add(SNewEmptyLine())  # Adding a new empty line

        # Add includes
        for include in includes:
            stmts.add(SDecl(DDefn(f"#include {include}")))
        stmts.add(SNewEmptyLine())  # Adding a new empty line

        # Add macros
        for macro in macros:
            stmts.add(SDecl(DDefn(macro)))
        if macros:
            stmts.add(SNewEmptyLine())  # Adding a new empty line

        return stmts

    def __add_namespace(self, namespaces: List[str]) -> Statement:
        """
        Add file description
        """
        stmts = SBlock([])

        for namespace in namespaces:
            stmts.add(SDecl(DDefn(f"using namespace {namespace};")))
        if namespaces:
            stmts.add(SNewEmptyLine())  # Adding a new empty line

        return stmts

    def __str__(self) -> str:
        """
        Return the string representation of this GPULoops program
        """

        return self.gpuloops.gen(0)
