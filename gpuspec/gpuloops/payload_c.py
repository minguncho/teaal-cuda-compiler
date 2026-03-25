"""
INSERT LICENSE HERE

Additional implementation for payload targetted for C/C++ code
"""

from gpuspec.gpuloops.base import Payload


class PVar_C(Payload):
    """
    A single variable payload
    """

    def __init__(self, var_type: str, name: str) -> None:
        self.var_type = var_type
        self.name = name

    def gen(self, parens: bool) -> str:
        """
        Generate the C/C++ output for a PVar_C
        """
        return f"{self.var_type} {self.name}"
