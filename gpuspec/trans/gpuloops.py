"""
INSERT LICENSE HERE

Translate an Einsum to the corresponding GPULoops code
"""

from typing import cast, List, Optional

from gpuspec.gpuloops import *
from gpuspec.parse import *

"""
    TODO:
    - Come back for arch, binding, and hardware stuff
    1. Develop a flow graph
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
            block_size: Optional[str] = None,
            grid_size: Optional[str] = None) -> None:
        """
        Perform the Einsum to GPULoops translation
        """
        self.einsum = einsum
        self.mapping = mapping
        self.scheduler = scheduler
        self.problem_type = problem_type
        self.N = N

        # Check for length of einsum expressions
        if len(einsum.get_expressions()) != 1:
            raise ValueError(
                "Current version only supports 1 Einsum expression!")

        # Typenames and default type
        self.typenames = {
            "setup_t": None,
            "index_t": "int",
            "offset_t": "int",
            "type_t": "float"
        }

        # Default kernel parameters
        self.block_size = "128"
        self.grid_size = "(A.rows + block_size - 1) / block_size"

        if block_size:
            self.block_size = block_size
        if grid_size:
            self.grid_size = grid_size

        self.gpuloops = self.__generate_loops_file()

    def __generate_loops_file(self) -> Statement:
        """
        Add statements for main loops file
        """

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
            stmts.add(SDecl(DDefn("")))  # Adding an empty line

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

        # Define kernel body
        body = self.__construct_gpu_kernel_body()

        gpu_kernel = SFunc(
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
            args.append(AParam("setup_t", "config"))
            args.append(AParam("type_t*", "B", "const"))
            args.append(AParam("type_t*", "Z"))

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
            inner_body.add(SDecl(DDefn("//TODO: Implement this\n")))

            # Construct the body for outer for-loop (traversing tiles)
            outer_body = SBlock([])
            outer_body.add(SAssign(ANewVar("type_t", "sum"), EVar("0")))
            outer_body.add(SDecl(DDefn("")))  # Add an empty line

            outer_body.add(SRangeFor(PVar("auto", "atom"),
                                     EMethod(EVar("config"),
                                             "atoms",
                                             [AJust(EVar("tile"))]),
                                     inner_body))
            outer_body.add(SDecl(DDefn("")))  # Add an empty line

            outer_body.add(SAssign(AAccess(EVar("Z"), EVar("tile")),
                                   EVar("sum")))

            # Build the outer for-loop
            body.add(SRangeFor(PVar("auto", "tile"),
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
        args: List[Argument] = [AParam("int", "argc"),
                                AParam("char**", "argv")]

        # Define function body
        body = self.__construct_main_fn_body()

        main_fn = SFunc(
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
            body.add(SAssignTypename(AVar(name), EVar(data_type)))
        body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 2: Create parameters obj
        body.add(
            SAssignObj(
                ANewVar("parameters_t", "parameters"),
                EFunc("",
                      [AJust(EVar("argc")),
                       AJust(EVar("argv"))])))
        body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 3: Create an input tensor A
        body.add(
            SAssignEmpty(
                ANewVar("matrix_market_t<index_t, offset_t, type_t>",
                        "mtx")))
        body.add(
            SAssignObj(
                ANewVar(
                    "csr_t<index_t, offset_t, type_t>", "A"), EParens(
                    EMethod(
                        EVar("mtx"), "load", [
                            AJust(
                                EField("parameters", "filename"))]))))
        body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 4: Based on problem type, create tensor B and Z
        if self.problem_type == "SpMV":
            # B and Z are vectors
            body.add(SAssignObj(
                ANewVar(
                    "vector_t<type_t>", "B"),
                EFunc("",
                      [AJust(EField("A", "cols"))])))

            body.add(SAssignObj(
                ANewVar(
                    "vector_t<type_t>", "Z"),
                EFunc("",
                      [AJust(EField("A", "rows"))])))
            body.add(SDecl(DDefn("")))  # Adding an empty line

            # Initialize vector B
            body.add(SExpr(EFunc("generate::random::uniform_distribution",
                                 [AJust(EMethod(EVar("B"), "begin", [])),
                                  AJust(EMethod(EVar("B"), "begin", [])),
                                  AJust(EVar(str(1))),
                                  AJust(EVar(str(self.N)))])))
            body.add(SDecl(DDefn("")))  # Adding an empty line
        else:  # SpMM, SpGEMM
            # B and Z are matrices
            body.add(SAssign(ANewVar("std::size_t", "n"), EVar(str(self.N))))

            if self.problem_type == "SpMM":
                body.add(SAssignObj(
                    ANewVar(
                        "matrix_t<type_t>", "B"),
                    EFunc("",
                          [AJust(EField("A", "cols")), AJust(EVar("n"))])))
            else:  # SpGEMM, create a sparse matrix B identical to A
                # TODO: Confirm this works in Loops
                body.add(
                    SAssignObj(
                        ANewVar(
                            "csr_t<index_t, offset_t, type_t>", "B"),
                        EParens(EMethod(
                                EVar("mtx"), "load", [
                                    AJust(
                                        EField("parameters", "filename"))]))))

            body.add(SAssignObj(
                ANewVar(
                    "matrix_t<type_t>", "Z"),
                EFunc("",
                      [AJust(EField("A", "rows")), AJust(EVar("n"))])))
            body.add(SDecl(DDefn("")))  # Adding an empty line

            # Initialize matrix B
            if self.problem_type == "SpMM":
                body.add(
                    SExpr(
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
                body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 5: Create tiles
        body.add(SDecl(DDefn("//TODO: Implement a tile pointer generator")))
        body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 6: Create a scheduler (only thread_mapped)
        if self.scheduler.get_scheduler() == "thread_mapped":
            body.add(
                SAssignTypename(AVar("setup_t"),
                                EVar(self.scheduler.construct_expr(1, 1))))
            body.add(SAssignObj(
                ANewVar(
                    "setup_t", "config"),
                EFunc("",
                      [AJust(
                          EMethod(EMethod(
                              EField("A", "offsets"), "data", []),
                              "get", [])),
                       AJust(EField("A", "rows")),
                       AJust(EField("A", "nnz"))])))
            body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 7: Define GPU kernel launch parameters
        body.add(SAssign(ANewVar("std::size_t", "block_size", "constexpr"),
                         EVar(self.block_size)))
        body.add(SAssign(ANewVar("std::size_t", "grid_size"),
                         EVar(self.grid_size)))
        body.add(SAssign(ANewVar("cudaStream_t", "stream"),
                         EVar("0")))
        body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 8: Start timer
        body.add(
            SAssignEmpty(
                ANewVar("util::timer_t", "timer")))
        body.add(SExpr(EMethod(EVar("timer"), "start", [])))
        body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 9: Execute GPU kernel
        kernel_name = "gpuloops_" + self.scheduler.get_scheduler()
        kernel_template = "<" + \
            (", ".join(t for t in list(self.typenames))) + ">"
        body.add(
            SExpr(
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
        body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 10: Synchronization and End timer
        body.add(
            SExpr(
                EFunc("cudaStreamSynchronize",
                      [AJust(EVar("stream"))])))
        body.add(SExpr(EMethod(EVar("timer"), "stop", [])))
        body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 11: Validation
        body.add(SDecl(DDefn("//TODO: Add a general validation")))
        body.add(SDecl(DDefn("")))  # Adding an empty line

        # Step 12: Print output
        body.add(SPrint([f"\"{self.problem_type},\"",
                         "mtx.dataset", "\",\"", "A.rows", "\",\"",
                         "A.cols", "\",\"", "A.nnz", "\",\"",
                         "timer.milliseconds()"]))

        return body

    def __str__(self) -> str:
        """
        Return the string representation of this GPULoops program
        """

        return self.gpuloops.gen(0)
