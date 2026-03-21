"""
INSERT LICENSE HERE

GPULoops AST and code generation for GPULoops declarations
"""

from gpuspec.gpuloops.base import Declaration


class DDefn(Declaration):
    """
    An GPULoops definition
    """

    def __init__(self, defn: str) -> None:
        self.defn = defn

    def gen(self) -> str:
        """
        Generate the GPULoops code for an DDefn
        """
        return self.defn
