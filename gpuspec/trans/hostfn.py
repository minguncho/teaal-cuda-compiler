"""
INSERT LICENSE HERE

Translate the host main function information
"""

from gpuspec.gpuloops import *
from gpuspec.ir.program import Program
from gpuspec.trans.scheduler import Scheduler
from gpuspec.trans.partitioner import Partitioner


class HostFn:
    """
    Generate the GPUloops code for the host main function information
    """

    def __init__(self,
                 program: Program,
                 scheduler: Scheduler,
                 partitioner: Partitioner) -> None:

        self.program = program
        self.scheduler = scheduler
        self.partitioner = partitioner

    def add_main_fn(self) -> Statement:
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
        for name, data_type in self.program.get_typenames().items():
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
        if self.program.get_problem_type() == "SpMV":
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
                                  AJust(EVar(str(self.program.get_N())))])))
            body.add(SNewEmptyLine())  # Adding a new empty line
        else:  # SpMM, SpGEMM
            # B and Z are matrices
            body.add(SAssign(ANewVar("std::size_t", "n"),
                     EVar(str(self.program.get_N()))))

            if self.program.get_problem_type() == "SpMM":
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
            if self.program.get_problem_type() == "SpMM":
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
                                EVar(str(self.program.get_N())))])))
                body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 5: Create a partitioner
        body.add(self.partitioner.create_partitioner())
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 6: Create a scheduler (only thread_mapped)
        if self.program.get_scheduler_type() == "thread_mapped":
            body.add(self.scheduler.create_scheduler())
            body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 7: Define GPU kernel launch parameters
        body.add(SAssign(ANewVar("std::size_t", "block_size", "constexpr"),
                         EVar(self.program.get_block_size())))
        body.add(SAssign(ANewVar("std::size_t", "grid_size"),
                         EVar(self.program.get_grid_size())))
        body.add(SAssign(ANewVar("cudaStream_t", "stream"),
                         EVar("0")))
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 8: Add a tracker if enabled
        if self.program.get_tracker_enabled():
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
            (", ".join(t for t in self.program.get_gpu_typenames())) + ">"
        launch_args = [
            AJust(EVar("stream")),
            AJust(EVar(self.program.get_kernel_name() + kernel_template)),
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

        if self.program.get_tracker_enabled():
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
        body.add(SPrint([f"\"{self.program.get_problem_type()},\"",
                         f"\"{self.program.get_scheduler_type() + "_edge"}\"",
                         "mtx.dataset", "\",\"", "A.rows", "\",\"",
                         "A.cols", "\",\"", "A.nnzs", "\",\"",
                         "timer.milliseconds()"]))
        body.add(SNewEmptyLine())  # Adding a new empty line

        # Step 14: Add a tracker's generate output if enabled
        if self.program.get_tracker_enabled():
            body.add(
                SExpr(
                    EMethod(
                        EVar("tracker"), "generate_output", [
                            AJust(
                                EString(
                                    self.program.get_scheduler_type() + "_edge"))])))
            body.add(SNewEmptyLine())  # Adding a new empty line

        return body
