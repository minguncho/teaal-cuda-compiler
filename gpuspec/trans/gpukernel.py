"""
INSERT LICENSE HERE

Translate the GPUKernel information
"""

from gpuspec.gpuloops import *
from gpuspec.ir.program import Program


class GPUKernel:
    """
    Generate the GPUloops code for the GPUKernel information
    """

    def __init__(self,
                 program: Program) -> None:

        self.program = program

    def add_gpu_kernel(self) -> Statement:
        """
        Add GPU kernel
        """

        # Define kernel definition
        template_types = self.program.get_gpu_typenames()
        declarations = ["__global__"]
        return_type = "void"
        fn_name = self.program.get_kernel_name()

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

        if self.program.get_scheduler_type() == "thread_mapped":
            # Requird args
            args.append(AParam("setup_t", "config"))
            args.append(AParam("index_t*", "row_indices", "const"))
            args.append(AParam("index_t*", "col_indices", "const"))
            args.append(AParam("type_t*", "values", "const"))
            args.append(AParam("type_t*", "B", "const"))
            args.append(AParam("type_t*", "Z"))

        elif self.program.get_scheduler_type() == "group_mapped":
            pass
        else:  # work_oriented
            pass

        if self.program.get_tracker_enabled():
            args.append(AParam("size_t*", "nz_tid"))

        return args

    def __construct_gpu_kernel_body(self) -> Statement:
        """
        Construct GPU kernel body
        """
        body = SBlock([])

        if self.program.get_scheduler_type() == "thread_mapped":
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

            if self.program.get_tracker_enabled():
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

        elif self.program.get_scheduler_type() == "group_mapped":
            pass
        else:  # work_oriented
            pass

        return body
