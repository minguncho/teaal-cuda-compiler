"""
INSERT LICENSE HERE

Translate an Einsum to the corresponding GPULoops code
"""

from typing import cast, List, Optional

from gpuspec.gpuloops import *
from gpuspec.ir.flow_graph import FlowGraph
from gpuspec.ir.flow_nodes import *
from gpuspec.ir.fusion import Fusion
from gpuspec.ir.hardware import Hardware
from gpuspec.ir.iter_graph import IterationGraph
from gpuspec.ir.metrics import Metrics
from gpuspec.ir.node import Node
from gpuspec.ir.program import Program
from gpuspec.parse import *
from gpuspec.trans.collector import Collector
from gpuspec.trans.graphics import Graphics
from gpuspec.trans.equation import Equation
from gpuspec.trans.footer import Footer
from gpuspec.trans.header import Header
from gpuspec.trans.partitioner import Partitioner
from gpuspec.trans.utils import TransUtils

"""
    TODO:
    - Come back for arch, binding, and hardware stuff
"""


class GPULoops:
    """
    Translate a given Einsum into the corresponding GPULoops code
    """

    def __init__(
            self,
            einsum: Einsum,
            mapping: Mapping,
            scheduler: Scheduler,
            file_names: dict[str, str],
            arch: Optional[Architecture] = None,
            bindings: Optional[Bindings] = None,
            format_: Optional[Format] = None) -> None:
        """
        Perform the Einsum to GPULoops translation
        """
        self.program = Program(einsum, mapping)
        self.scheduler = scheduler

        self.hardware: Optional[Hardware] = None
        self.format = format_
        if arch and bindings and arch.get_spec():
            self.hardware = Hardware(arch, bindings, self.program)
            self.fusion = Fusion(self.hardware)

        self.trans_utils = TransUtils(self.program)

        # Typenames and default type
        self.typenames = {
            "setup_t": None,
            "index_t": "int",
            "offset_t": "int",
            "type_t": "float"
        }

        # Check if file names exist
        required_fnames = ["loops_fname"]
        for fname in required_fnames:
            if fname not in file_names:
                raise KeyError(
                    f"Missing required file name configuration: '{fname}'")
        self.file_names = file_names

        self.loops_file = self.__generate_loops_file()

    def __generate_loops_file(self) -> Statement:
        """
        Add statements for main loops file
        """

        stmts = SBlock([])

        # Step 1: Add file description
        comments = [
            "/**",
            " * @file " + self.file_names["loops_fname"],
            " * Generated Loops code",
            "*/"
        ]
        stmts.add(self.__add_file_description(comments))

        # Step 2: Add preprocessor directives (#include, #define, etc)
        pragmas: list[str] = []
        includes = ["\"helpers.hxx\"",
                    "<loops/schedule.hxx>",
                    "<loops/container/formats.hxx>",
                    "<loops/container/vector.hxx>",
                    "<loops/util/launch.hxx>",
                    "<loops/util/device.hxx>",
                    "<loops/memory.hxx>",
                    "<iostream>"]
        macros: list[str] = []

        stmts.add(self.__add_preproc_dir(pragmas, includes, macros))

        # Step 3: Add namespaces
        namespaces = ["loops"]
        stmts.add(self.__add_namespace(namespaces))

        # Step 4: Add GPU kernel
        stmts.add(self.__add_gpu_kernel())

        # Step 5: Add host main function
        stmts.add(self.__add_main_fn())

        return stmts

    def __add_file_description(self, comments: List[str]) -> Statement:
        """
        Add file description
        """
        stmts = SBlock([])

        for comment in comments:
            stmts.add(SDecl(DDefn(comment)))
        stmts.add(SDecl(DDefn("")))  # Adding an empty line

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
            stmts.add(SDecl(DDefn("")))  # Adding an empty line

        # Add includes
        for include in includes:
            stmts.add(SDecl(DDefn(f"#include {include}")))
        stmts.add(SDecl(DDefn("")))  # Adding an empty line

        # Add macros
        for macro in macros:
            stmts.add(SDecl(DDefn(macro)))
        if macros:
            stmts.add(SDecl(DDefn("")))  # Adding

        return stmts

    def __add_namespace(self, namespaces: List[str]) -> Statement:
        """
        Add file description
        """
        stmts = SBlock([])

        for namespace in namespaces:
            stmts.add(SDecl(DDefn(f"using namespace {namespace};")))
        if namespaces:
            stmts.add(SDecl(DDefn("")))  # Adding an empty line

        return stmts

    def __add_gpu_kernel(self) -> Statement:
        """
        Add GPU kernel
        """

        # Define kernel definition
        template_types = list(self.typenames)
        declarations = ["__global__"]
        return_type = "void"
        fn_name = "gpuloops_" + self.scheduler.get_scheduler()

        # Define kernel argumnets
        args = self.__construct_gpu_kernel_args()

        # Sample args
        '''args: List[Assignable] = [AVar_C("setup_t", "config"),
                                  AVar_C("std::size_t", "rows", True),
                                  AVar_C("std::size_t", "cols", True),
                                  AVar_C("std::size_t", "nnz", True),
                                  AVar_C("offset_t*", "offsets", True),
                                  AVar_C("index_t*", "indices", True),
                                  AVar_C("type_t*", "values", True),
                                  AVar_C("type_t*", "x", True),
                                  AVar_C("type_t*", "y")]'''

        # Define kernel body
        body = SBlock([])

        gpu_kernel = SFunc_C(
            return_type,
            fn_name,
            args,
            body,
            template_types,
            declarations)

        return gpu_kernel

    def __construct_gpu_kernel_args(self) -> List[Assignable]:
        """
        Construct GPU kernel arguments
        """
        args: List[Assignable] = []

        return args

    def __construct_gpu_kernel_body(self) -> Statement:
        """
        Construct GPU kernel body
        """
        body = SBlock([])

        return body

    def __add_main_fn(self) -> Statement:
        """
        Add main host function
        """

        # Define function definition
        return_type = "int"
        fn_name = "main"

        # Define function arguments
        args: List[Assignable] = [AVar_C("int", "argc"),
                                  AVar_C("char**", "argv")]

        # Define function body
        body = self.__construct_main_fn_body()

        main_fn = SFunc_C(
            return_type,
            fn_name,
            args,
            body)

        return main_fn

    def __construct_main_fn_body(self) -> Statement:
        """
        Construct main host body
        """
        body = SBlock([])

        # Step 1: Define typenames

        # Step 2: Create parameters obj

        # Step 3: Create tensors

        # Step 4: Create tiles

        # Step 5: Create a scheduler

        # Step 6: Define GPU kernel launch parameters

        # Step 7: Execute GPU kernel

        # Step 8: Validation

        # Step 9: Print output

        return body

    def get_files(self) -> dict[str, str]:
        """
        Return the string representation of this GPULoops program for both files
        """

        return {
            "loops_fname": self.loops_file.gen(0),
        }
