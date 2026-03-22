"""
INSERT LICENSE HERE

Additional implementation for arguments targetted for C/C++ code
"""

from typing import Optional

from gpuspec.gpuloops.base import Argument


class AParam_C(Argument):
    """
    A GPULoops parameterized argument (C/C++)
    """

    def __init__(self, var_type: str, name: str,
                 const: Optional[str] = None) -> None:
        self.var_type = var_type
        self.name = name
        self.const = const

    def gen(self) -> str:
        """
        Generate the C/C++ code for an AParamC
        """
        c = f"{self.const} " if self.const else ""
        return f"{c}{self.var_type} {self.name}"
