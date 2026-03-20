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

"""
    The goal is to generate 1 file:
        1. .cu file with main() function and GPU kernel
    File contents for (1):
        1. File description
        2. Preprocessor directives (#pragma once, #include, macros,...)
        3. GPU Kernel
            - template
            - function name with parameters
            - contents
        4. main() function
            - function name with parameters
            - define types (index_t, offset_t, type_t)
            - Creating input tensors
            - Based on defined tiles and atoms, generate a pointer for tiles
            - Creating a scheduler
            - Defining thread block and grid size
            - A call to launch kernel
            - Kernel synchronization
            - Print out (or write to an output file) the benchmark result
            - Validation for correctness
"""

"""
    Challenges:
    1. Multiple Einsum expression (HiFiber simply handles each expression separately)
    2. Understanding spacetime and applying the sequential and parallel steps.
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

        self.hardware: Optional[Hardware] = None
        self.format = format_
        if arch and bindings and arch.get_spec():
            self.hardware = Hardware(arch, bindings, self.program)
            self.fusion = Fusion(self.hardware)

        self.trans_utils = TransUtils(self.program)

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
            stmts.add(SExpr(EVar(comment)))
        stmts.add(SExpr(EVar("")))  # Adding an empty line

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
            stmts.add(SExpr(EVar(pragma)))
        if pragmas:
            stmts.add(SExpr(EVar("")))  # Adding an empty line

        # Add includes
        for include in includes:
            stmts.add(SExpr(EVar(f"#include {include}")))
        stmts.add(SExpr(EVar("")))  # Adding an empty line

        # Add macros
        for macro in macros:
            stmts.add(SExpr(EVar(macro)))
        if macros:
            stmts.add(SExpr(EVar("")))  # Adding

        return stmts

    def __add_namespace(self, namespaces: List[str]) -> Statement:
        """
        Add file description
        """
        stmts = SBlock([])

        for namespace in namespaces:
            stmts.add(SExpr(EVar(f"using namespace {namespace};")))
        if namespaces:
            stmts.add(SExpr(EVar("")))  # Adding an empty line

        return stmts

    def __add_gpu_kernel(self) -> Statement:
        """
        Add GPU kernel
        """
        stmts = SBlock([])

        return stmts

    def __add_main_fn(self) -> Statement:
        """
        Add main host function
        """
        stmts = SBlock([])

        return stmts

    def __translate(self, i: int) -> Statement:

        # Generate for the given einsum
        self.program.add_einsum(i)

        # Build metrics if there is hardware
        """self.metrics: Optional[Metrics] = None
        if self.hardware and self.format:
            self.metrics = Metrics(self.program, self.hardware, self.format)
            self.fusion.add_einsum(self.program)

        # Create the flow graph and get the relevant nodes
        flow_graph = FlowGraph(self.program, self.metrics, ["hoist"])
        nodes = flow_graph.get_sorted()

        # Create all relevant translator objects
        self.graphics = Graphics(self.program, self.metrics)
        self.partitioner = Partitioner(self.program, self.trans_utils)
        self.header = Header(self.program, self.metrics, self.partitioner)
        self.graph = IterationGraph(self.program)
        self.eqn = Equation(self.program, self.metrics)

        if self.metrics:
            self.collector = Collector(self.program, self.metrics, self.fusion)

        stmt = self.__trans_nodes(nodes)[1]"""

        stmt = SBlock([])

        self.program.reset()
        return stmt

    def __trans_nodes(self, nodes: List[Node]) -> Tuple[int, Statement]:
        """
        Recursive function to generate the actual GPULoops program
        """
        code = SBlock([])

        i = 0
        """while i < len(nodes):
            node = nodes[i]

            if isinstance(node, EagerInputNode):
                code.add(
                    self.eqn.make_eager_inputs(
                        node.get_rank(),
                        node.get_tensors()))

            elif isinstance(node, EndLoopNode):
                return i + 1, code

            elif isinstance(node, FromFiberNode):
                tensor = self.program.get_equation().get_tensor(node.get_tensor())
                code.add(Header.make_tensor_from_fiber(tensor))

            elif isinstance(node, GetPayloadNode):
                tensor = self.program.get_equation().get_tensor(node.get_tensor())
                code.add(
                    self.header.make_get_payload(
                        tensor, node.get_ranks()))

            elif isinstance(node, GetRootNode):
                tensor = self.program.get_equation().get_tensor(node.get_tensor())
                code.add(Header.make_get_root(tensor))

            elif isinstance(node, IntervalNode):
                code.add(self.eqn.make_interval(node.get_rank()))

            elif isinstance(node, LoopNode):
                # Generate the for loop
                rank, tensors = self.graph.peek_concord()
                expr = self.eqn.make_iter_expr(cast(str, rank), tensors)
                _, tensors = self.graph.pop_concord()
                payload = self.eqn.make_payload(cast(str, rank), tensors)

                # Recurse for the for loop body
                j, body = self.__trans_nodes(nodes[(i + 1):])
                code.add(SFor(payload, expr, body))
                i += j

            elif isinstance(node, MetricsNode):
                if node.get_type() == "Body":
                    code.add(self.collector.make_body())

                elif node.get_type() == "Dump":
                    code.add(self.collector.dump())

                elif node.get_type() == "End":
                    code.add(self.collector.end())

                elif node.get_type() == "Start":
                    code.add(self.collector.start())

                else:
                    raise ValueError(
                        "Unknown node: " +
                        repr(node))  # pragma: no cover

            elif isinstance(node, MetricsFooterNode):
                code.add(self.collector.make_loop_footer(node.get_rank()))

            elif isinstance(node, MetricsHeaderNode):
                code.add(self.collector.make_loop_header(node.get_rank()))

            elif isinstance(node, OtherNode):
                if node.get_type() == "Body":
                    code.add(self.eqn.make_update())
                    code.add(self.graphics.make_body())

                elif node.get_type() == "Footer":
                    code.add(
                        Footer.make_footer(
                            self.program,
                            self.graphics,
                            self.partitioner))

                elif node.get_type() == "Graphics":
                    code.add(self.graphics.make_header())

                elif node.get_type() == "Output":
                    code.add(self.header.make_output())

                else:
                    raise ValueError(
                        "Unknown node: " +
                        repr(node))  # pragma: no cover

            elif isinstance(node, PartNode):
                tensor = self.program.get_equation().get_tensor(node.get_tensor())
                ranks = node.get_ranks()

                tensor.from_fiber()
                code.add(self.partitioner.partition(tensor, ranks))

            elif isinstance(node, SwizzleNode):
                tensor = self.program.get_equation().get_tensor(node.get_tensor())
                code.add(
                    self.header.make_swizzle(
                        tensor,
                        node.get_ranks(),
                        node.get_type()))

            else:
                raise ValueError(
                    "Unknown node: " +
                    repr(node))  # pragma: no cover

            i += 1"""

        return i, code

    def get_files(self) -> dict[str, str]:
        """
        Return the string representation of this GPULoops program for both files
        """

        return {
            "loops_fname": self.loops_file.gen(0),
        }
