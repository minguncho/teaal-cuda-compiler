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
            problem_type: str,
            N: int,
            file_names: dict[str, str],
            arch: Optional[Architecture] = None,
            bindings: Optional[Bindings] = None,
            format_: Optional[Format] = None,
            block_size: Optional[str] = None,
            grid_size: Optional[str] = None) -> None:
        """
        Perform the Einsum to GPULoops translation
        """
        self.program = Program(einsum, mapping)
        self.scheduler = scheduler
        self.problem_type = problem_type

        # Check for problem_type
        req_problem_type = ["SpMV", "SpMM", "SpGEMM"]
        if problem_type not in req_problem_type:
            req_str = ", ".join([f"\"{t}\"" for t in req_problem_type])
            raise KeyError(
                f"Invalid problem type: '{problem_type}', must be {req_str}")

        self.N = N
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
        req_fnames = ["loops_fname"]
        for fname in req_fnames:
            if fname not in file_names:
                raise KeyError(
                    f"Missing required file name configuration: '{fname}'")
        self.file_names = file_names

        # Default kernel parameters
        self.block_size = "128"
        self.grid_size = "(A.rows + block_size - 1) / block_size"

        if block_size:
            self.block_size = block_size
        if grid_size:
            self.grid_size = grid_size

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
            stmts.add(SDecl_C(DDefn(comment)))
        stmts.add(SDecl_C(DDefn("")))  # Adding an empty line

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
            stmts.add(SDecl_C(DDefn(pragma)))
        if pragmas:
            stmts.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Add includes
        for include in includes:
            stmts.add(SDecl_C(DDefn(f"#include {include}")))
        stmts.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Add macros
        for macro in macros:
            stmts.add(SDecl_C(DDefn(macro)))
        if macros:
            stmts.add(SDecl_C(DDefn("")))  # Adding an empty line

        return stmts

    def __add_namespace(self, namespaces: List[str]) -> Statement:
        """
        Add file description
        """
        stmts = SBlock([])

        for namespace in namespaces:
            stmts.add(SDecl_C(DDefn(f"using namespace {namespace};")))
        if namespaces:
            stmts.add(SDecl_C(DDefn("")))  # Adding an empty line

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

        # Define kernel body
        body = self.__construct_gpu_kernel_body()

        gpu_kernel = SFunc_C(
            return_type,
            fn_name,
            args,
            body,
            template_types,
            declarations)

        return gpu_kernel

    def __construct_gpu_kernel_args(self) -> List[Argument]:
        """
        Construct GPU kernel arguments
        """
        args: List[Argument] = []

        if self.scheduler.get_scheduler() == "thread_mapped":
            # Requird args
            args.append(AParam_C("setup_t", "config"))
            args.append(AParam_C("type_t*", "B", "const"))
            args.append(AParam_C("type_t*", "Z"))

        elif self.scheduler.get_scheduler() == "group_mapped":
            pass
        else:  # work_oriented
            pass

        return args

    def __construct_gpu_kernel_body(self) -> Statement:
        """
        Construct GPU kernel body
        """
        body = SBlock([])

        if self.scheduler.get_scheduler() == "thread_mapped":
            # Construct the body for inner for-loop (traversing atoms)
            inner_body = SBlock([])
            inner_body.add(SDecl_C(DDefn("//TODO: Implement this\n")))

            # Construct the body for outer for-loop (traversing tiles)
            outer_body = SBlock([])
            outer_body.add(SAssign_C(AVar_C("type_t", "sum"), EVar("0")))
            outer_body.add(SDecl_C(DDefn("")))  # Add an empty line

            outer_body.add(SRangeFor_C(PVar_C("auto", "atom"),
                                       EMethod(EVar("config"),
                                               "atoms",
                                               [AJust(EVar("tile"))]),
                                       inner_body))
            outer_body.add(SDecl_C(DDefn("")))  # Add an empty line

            outer_body.add(SAssign_C(AAccess(EVar("Z"), EVar("tile")),
                                     EVar("sum")))

            # Build the outer for-loop
            body.add(SRangeFor_C(PVar_C("auto", "tile"),
                                 EMethod(EVar("config"),
                                         "tiles", []),
                                 outer_body))

        elif self.scheduler.get_scheduler() == "group_mapped":
            pass
        else:  # work_oriented
            pass

        return body

    def __add_main_fn(self) -> Statement:
        """
        Add main host function
        """

        # Define function definition
        return_type = "int"
        fn_name = "main"

        # Define function arguments
        args: List[Argument] = [AParam_C("int", "argc"),
                                AParam_C("char**", "argv")]

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
        for name, data_type in self.typenames.items():
            if name == "setup_t" or data_type is None:
                continue
            body.add(SAssignTypename_C(AVar(name), EVar(data_type)))
        body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 2: Create parameters obj
        body.add(
            SAssignObj_C(
                AVar_C("parameters_t", "parameters"),
                EFunc("",
                      [AJust(EVar("argc")),
                       AJust(EVar("argv"))])))
        body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 3: Create an input tensor A
        body.add(
            SAssignEmpty_C(
                AVar_C("matrix_market_t<index_t, offset_t, type_t>",
                       "mtx")))
        body.add(
            SAssignObj_C(
                AVar_C(
                    "csr_t<index_t, offset_t, type_t>", "A"), EParens(
                    EMethod(
                        EVar("mtx"), "load", [
                            AJust(
                                EField("parameters", "filename"))]))))
        body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 4: Based on problem type, create tensor B and Z
        if self.problem_type == "SpMV":
            # B and Z are vectors
            body.add(SAssignObj_C(
                AVar_C(
                    "vector_t<type_t>", "B"),
                EFunc("",
                      [AJust(EField("A", "cols"))])))

            body.add(SAssignObj_C(
                AVar_C(
                    "vector_t<type_t>", "Z"),
                EFunc("",
                      [AJust(EField("A", "rows"))])))
            body.add(SDecl_C(DDefn("")))  # Adding an empty line

            # Initialize vector B
            body.add(SExpr_C(EFunc("generate::random::uniform_distribution",
                                   [AJust(EMethod(EVar("B"), "begin", [])),
                                    AJust(EMethod(EVar("B"), "begin", [])),
                                       AJust(EVar(str(1))),
                                       AJust(EVar(str(self.N)))])))
            body.add(SDecl_C(DDefn("")))  # Adding an empty line
        else:  # SpMM, SpGEMM
            # B and Z are matrices
            body.add(SAssign_C(AVar_C("std::size_t", "n"), EVar(str(self.N))))

            if self.problem_type == "SpMM":
                body.add(SAssignObj_C(
                    AVar_C(
                        "matrix_t<type_t>", "B"),
                    EFunc("",
                          [AJust(EField("A", "cols")), AJust(EVar("n"))])))
            else:  # SpGEMM, create a sparse matrix B identical to A
                # TODO: Confirm this works in Loops
                body.add(
                    SAssignObj_C(
                        AVar_C(
                            "csr_t<index_t, offset_t, type_t>", "B"),
                        EParens(EMethod(
                                EVar("mtx"), "load", [
                                    AJust(
                                        EField("parameters", "filename"))]))))

            body.add(SAssignObj_C(
                AVar_C(
                    "matrix_t<type_t>", "Z"),
                EFunc("",
                      [AJust(EField("A", "rows")), AJust(EVar("n"))])))
            body.add(SDecl_C(DDefn("")))  # Adding an empty line

            # Initialize matrix B
            if self.problem_type == "SpMM":
                body.add(
                    SExpr_C(
                        EFunc(
                            "generate::random::uniform_distribution",
                            [AJust(
                                EMethod(
                                    EField("B", "m_data"), "begin", [])),
                             AJust(
                                EMethod(
                                    EField("B", "m_data"), "begin", [])),
                             AJust(
                                EVar(str(1))),
                             AJust(
                                EVar(str(self.N)))])))
                body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 5: Create tiles
        body.add(SDecl_C(DDefn("//TODO: Implement a tile pointer generator")))
        body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 6: Create a scheduler (only thread_mapped)
        if self.scheduler.get_scheduler() == "thread_mapped":
            body.add(
                SAssignTypename_C(AVar("setup_t"),
                                  EVar(self.scheduler.construct_expr(1, 1))))
            body.add(SAssignObj_C(
                AVar_C(
                    "setup_t", "config"),
                EFunc("",
                      [AJust(
                          EMethod(EMethod(
                              EField("A", "offsets"), "data", []),
                              "get", [])),
                       AJust(EField("A", "rows")),
                       AJust(EField("A", "nnz"))])))
            body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 7: Define GPU kernel launch parameters
        body.add(SAssign_C(AVar_C("std::size_t", "block_size", "constexpr"),
                           EVar(self.block_size)))
        body.add(SAssign_C(AVar_C("std::size_t", "grid_size"),
                           EVar(self.grid_size)))
        body.add(SAssign_C(AVar_C("cudaStream_t", "stream"),
                           EVar("0")))
        body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 8: Start timer
        body.add(
            SAssignEmpty_C(
                AVar_C("util::timer_t", "timer")))
        body.add(SExpr_C(EMethod(EVar("timer"), "start", [])))
        body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 9: Execute GPU kernel
        kernel_name = "gpuloops_" + self.scheduler.get_scheduler()
        kernel_template = "<" + \
            (", ".join(t for t in list(self.typenames))) + ">"
        body.add(
            SExpr_C(
                EFunc("launch::non_cooperative",
                      [AJust(EVar("stream")),
                       AJust(EVar(kernel_name + kernel_template)),
                       AJust(EVar("grid_size")), AJust(EVar("block_size")),
                       AJust(EVar("config")), AJust(EField("A", "rows")),
                       AJust(EField("A", "cols")), AJust(EField("A", "nnz")),
                       AJust(EMethod(EMethod(EField("A", "offsets"), "data", []),
                                     "get", [])),
                       AJust(EMethod(EMethod(EField("A", "indices"), "data", []),
                                     "get", [])),
                       AJust(EMethod(EMethod(EField("A", "values"), "data", []),
                                     "get", [])),
                       AJust(EMethod(EMethod(EVar("B"), "data", []),
                                     "get", [])),
                       AJust(EMethod(EMethod(EVar("Z"), "data", []),
                                     "get", []))])))
        body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 10: Synchronization and End timer
        body.add(
            SExpr_C(
                EFunc("cudaStreamSynchronize",
                      [AJust(EVar("stream"))])))
        body.add(SExpr_C(EMethod(EVar("timer"), "stop", [])))
        body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 11: Validation
        body.add(SDecl_C(DDefn("//TODO: Add a general validation")))
        body.add(SDecl_C(DDefn("")))  # Adding an empty line

        # Step 12: Print output
        body.add(SPrint_C([f"\"{self.file_names["loops_fname"]},\"",
                           "mtx.dataset", "\",\"", "A.rows", "\",\"",
                           "A.cols", "\",\"", "A.nnz", "\",\"",
                           "timer.milliseconds()"]))

        return body

    def get_files(self) -> dict[str, str]:
        """
        Return the string representation of this GPULoops program for both files
        """

        return {
            "loops_fname": self.loops_file.gen(0),
        }
