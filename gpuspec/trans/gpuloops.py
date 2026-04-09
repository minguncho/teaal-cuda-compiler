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
            tracker_enabled: Optional[bool] = False,
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
        self.tracker_enabled = tracker_enabled

        # Check for length of einsum expressions
        if len(einsum.get_expressions()) != 1:
            raise ValueError(
                "Current version only supports 1 Einsum expression!")

        # Typenames and default type
        self.typenames = {
            "setup_t": None,
            "index_t": "int",
            "offset_t": "int",
            "type_t": "float",
            "quarks_t": "std::size_t"
        }
        self.gpu_typenames = ["setup_t", "index_t", "type_t"]

        # Default kernel parameters
        self.block_size = "128"
        self.grid_size = "(partitioner.get_num_tiles() + block_size - 1) / block_size"
        self.kernel_name = self.scheduler.get_scheduler() + "_edge"

        if block_size:
            self.block_size = block_size
        if grid_size:
            self.grid_size = grid_size

        # TODO: Add implementation for this instead of hard coding
        self.partition_method = "coordinate"
        self.atoms_M0 = "1"
        self.atoms_K0 = "1"
        self.tiles_M1 = "1"
        self.tiles_K1 = "A.cols"
        self.atoms_nnz = self.tiles_num_atoms = "1"

        self.gpuloops = self.__generate_gpuloops()

    def __generate_gpuloops(self) -> Statement:
        """
        Main framework for loops file
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

    def __add_gpu_kernel(self) -> Statement:
        """
        Add GPU kernel
        """

        # Define kernel definition
        template_types = self.gpu_typenames
        declarations = ["__global__"]
        return_type = "void"
        fn_name = self.kernel_name

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
            args.append(AParam("index_t*", "row_indices", "const"))
            args.append(AParam("index_t*", "col_indices", "const"))
            args.append(AParam("type_t*", "values", "const"))
            args.append(AParam("type_t*", "B", "const"))
            args.append(AParam("type_t*", "Z"))

        elif self.scheduler.get_scheduler() == "group_mapped":
            pass
        else:  # work_oriented
            pass

        if self.tracker_enabled:
            args.append(AParam("size_t*", "nz_tid"))

        return args

    def __construct_gpu_kernel_body(self) -> Statement:
        """
        Construct GPU kernel body
        """
        body = SBlock([])

        if self.scheduler.get_scheduler() == "thread_mapped":
            # Construct the body of traversing quarks for-loop
            quark_body = SBlock([])
            quark_body.add(SExpr(EFunc("atomicAdd",
                                       [AJust(EFunc("&",
                                              [AJust(EAccess(EVar("Z"),
                                                             EAccess(EVar("row_indices"),
                                                                     EVar("*quark"))))])),
                                        AJust(EMult(EAccess(EVar("values"), EVar("*quark")),
                                                    EAccess(EVar("B"),
                                                            EAccess(EVar("col_indices"),
                                                                    EVar("*quark")))))])))

            if self.tracker_enabled:
                quark_body.add(SAssign(
                    AAccess(EVar("nz_tid"), EVar("*quark")),
                    EAdd(EMult(EVar("blockIdx.x"), EVar("blockDim.x"), True),
                         EVar("threadIdx.x"))))

            # Construct the body of traversing atoms for-loop
            atom_body = SBlock([])
            atom_body.add(SIf((EEqual(EPointerAccess(EVar("atom"),
                                                     EFunc("get_num_quarks", [])),
                                      EVar("0")),
                              SExpr(EVar("continue"))), [], None))
            atom_body.add(SNewEmptyLine())  # Add an empty line

            atom_body.add(SRangeFor(PVar("auto", "quark"),
                                    EMethod(EVar("config"),
                                            "quarks",
                                            [AJust(EVar("atom"))]),
                                    quark_body))

            # Construct the body of traversing tiles for-loop
            tile_body = SBlock([])
            tile_body.add(SRangeFor(PVar("auto", "atom"),
                                    EMethod(EVar("config"),
                                            "atoms",
                                            [AJust(EVar("tile_idx"))]),
                                    atom_body))

            # Build the outer for-loop
            body.add(SRangeFor(PVar("auto", "tile_idx"),
                               EMethod(EVar("config"),
                                       "tiles", []),
                               tile_body))

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
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 2: Create parameters obj
        body.add(
            SAssignObj(
                ANewVar("parameters_t", "parameters"),
                EFunc("",
                      [AJust(EVar("argc")),
                       AJust(EVar("argv"))])))
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 3: Create an input tensor A
        body.add(
            SAssignEmpty(
                ANewVar("matrix_market_t<index_t, offset_t, type_t>",
                        "mtx")))
        body.add(
            SAssign(
                ANewVar("coo_t<index_t, type_t, memory_space_t::host>", "A"),
                EMethod(
                    EVar("mtx"), "load", [
                        AJust(
                            EField("parameters", "filename"))])))
        body.add(SAssignObj(
            ANewVar(
                "coo_t<index_t, type_t>", "A_device"),
            EFunc("", [AJust(EVar("A"))])))

        body.add(SNewEmptyLine())  # Adding a new empty line

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
            body.add(SNewEmptyLine())  # Adding a new empty line

            # Initialize vector B
            body.add(SExpr(EFunc("generate::random::uniform_distribution",
                                 [AJust(EMethod(EVar("B"), "begin", [])),
                                  AJust(EMethod(EVar("B"), "begin", [])),
                                  AJust(EVar(str(1))),
                                  AJust(EVar(str(self.N)))])))
            body.add(SNewEmptyLine())  # Adding a new empty line
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
            body.add(SNewEmptyLine())  # Adding a new empty line

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
                body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 5: Create a partitioner
        body.add(SAssignObj(
            ANewVar(
                "Partitioner<index_t, type_t, quarks_t>", " partitioner"),
            EFunc("", [AJust(EVar("A"))])))
        if self.partition_method == "coordinate":
            body.add(SExpr(
                EMethod(EVar("partitioner"),
                        "partition_coordinate_space",
                        [AJust(EVar(self.atoms_M0)),
                         AJust(EVar(self.atoms_K0)),
                         AJust(EVar(self.tiles_M1)),
                         AJust(EVar(self.tiles_K1))])))
        elif self.partition_method == "position":
            body.add(SExpr(
                EMethod(EVar("partitioner"),
                        "partition_atoms_dynamic",
                        [AJust(EVar(self.atoms_nnz)),
                         AJust(EVar(self.tiles_num_atoms))])))
        else:
            raise ValueError(
                "Invalid partition method!, Must be 'coordinate' or 'position")
        body.add(SExpr(
            EMethod(EVar("partitioner"), "prepare_gpu", [])))
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 6: Create a scheduler (only thread_mapped)
        if self.scheduler.get_scheduler() == "thread_mapped":
            body.add(
                SAssignTypename(AVar("setup_t"),
                                EVar(self.scheduler.construct_expr(1, 1))))
            body.add(SAssignObj(
                ANewVar(
                    "setup_t", "config"),
                EFunc("",
                      [AJust(EMethod(EMethod(EMethod(EVar("partitioner"), "get_work_tiles", []),
                                             "data", []), "get", [])),
                       AJust(EMethod(EVar("partitioner"), "get_num_tiles", []))])))
            body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 7: Define GPU kernel launch parameters
        body.add(SAssign(ANewVar("std::size_t", "block_size", "constexpr"),
                         EVar(self.block_size)))
        body.add(SAssign(ANewVar("std::size_t", "grid_size"),
                         EVar(self.grid_size)))
        body.add(SAssign(ANewVar("cudaStream_t", "stream"),
                         EVar("0")))
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 8: Add a tracker if enabled
        if self.tracker_enabled:
            body.add(SAssignObj(
                ANewVar(
                    "Tracker", "tracker"),
                EFunc("", [AJust(EField("A", "nnzs")),
                           AJust(EMult(EVar("block_size"),
                                       EVar("grid_size")))])))
            body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 9: Start timer
        body.add(
            SAssignEmpty(
                ANewVar("util::timer_t", "timer")))
        body.add(SExpr(EMethod(EVar("timer"), "start", [])))
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 10: Execute GPU kernel
        kernel_template = "<" + \
            (", ".join(t for t in self.gpu_typenames)) + ">"
        launch_args = [
            AJust(EVar("stream")),
            AJust(EVar(self.kernel_name + kernel_template)),
            AJust(EVar("grid_size")), AJust(EVar("block_size")),
            AJust(EVar("config")),
            AJust(EMethod(EMethod(EField("A_device", "row_indices"),
                                  "data", []), "get", [])),
            AJust(EMethod(EMethod(EField("A_device", "col_indices"),
                                  "data", []), "get", [])),
            AJust(EMethod(EMethod(EField("A_device", "values"),
                                  "data", []), "get", [])),
            AJust(EMethod(EMethod(EVar("B"), "data", []),
                          "get", [])),
            AJust(EMethod(EMethod(EVar("Z"), "data", []),
                          "get", []))
        ]

        if self.tracker_enabled:
            launch_args.append(
                AJust(
                    EMethod(
                        EMethod(
                            EMethod(
                                EVar("tracker"),
                                "get_nz_tid",
                                []),
                            "data",
                            []),
                        "get",
                        [])))

        body.add(
            SExpr(
                EFunc("launch::non_cooperative", launch_args)))

        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 11: Synchronization and End timer
        body.add(
            SExpr(
                EFunc("cudaStreamSynchronize",
                      [AJust(EVar("stream"))])))
        body.add(SExpr(EMethod(EVar("timer"), "stop", [])))
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 12: Validation
        check_body = SBlock([])
        check_body.add(SAssignObj(
            ANewVar(
                "csr_t<index_t, offset_t, type_t>", "A_csr"),
            EFunc("", [AJust(EVar("A"))])))
        check_body.add(SExpr(EFunc("cpu::validate",
                                   [AJust(EVar("parameters")),
                                    AJust(EVar("A_csr")),
                                    AJust(EVar("B")),
                                    AJust(EVar("Z"))])))
        body.add(SIf((EField("parameters", "validate"), check_body), [], None))
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 13: Print output
        body.add(SPrint([f"\"{self.problem_type},\"",
                         f"\"{self.scheduler.get_scheduler() + "_edge"}\"",
                         "mtx.dataset", "\",\"", "A.rows", "\",\"",
                         "A.cols", "\",\"", "A.nnzs", "\",\"",
                         "timer.milliseconds()"]))
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 14: Add a tracker's generate output if enabled
        if self.tracker_enabled:
            body.add(
                SExpr(
                    EMethod(
                        EVar("tracker"), "generate_output", [
                            AJust(
                                EString(
                                    self.scheduler.get_scheduler() + "_edge"))])))
            body.add(SNewEmptyLine())  # Adding a new empty line

        return body

    def __str__(self) -> str:
        """
        Return the string representation of this GPULoops program
        """

        return self.gpuloops.gen(0)
