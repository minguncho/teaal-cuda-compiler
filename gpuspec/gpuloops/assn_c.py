"""
INSERT LICENSE HERE

Additional implementation for assignables targetted for C/C++ code
"""

from gpuspec.gpuloops.base import Assignable


class AVar_C(Assignable):
    """
    An GPULoops Variable (C/C++)
    """

    def __init__(self, var_type: str, name: str,
                 is_const: bool = False) -> None:
        self.var_type = var_type
        self.name = name
        self.is_const = is_const

    def gen(self) -> str:
        """
        Generate the C/C++ code for an AVar_C
        """
        const = "const " if self.is_const else ""
        return f"{const}{self.var_type} {self.name}"
