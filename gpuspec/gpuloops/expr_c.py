"""
INSERT LICENSE HERE

Additional implementation for expressions targetted for C/C++ code
"""

import numpy as np

from typing import Sequence

from gpuspec.gpuloops.base import Argument, Expression


class EFloat32(Expression):
    """
    An GPULoops float (32-bit)

    #TODO: Implement this
    """

    def __init__(self, float_: np.float32) -> None:
        self.float = float_

    def gen(self) -> str:
        """
        Generate GPULoops code for an EFloat
        """
        if self.float == float("inf"):
            return "float(\"inf\")"
        elif self.float == -float("inf"):
            return "-float(\"inf\")"
        else:
            return str(self.float)


class EDouble(Expression):
    """
    An GPULoops double (64-bit)

    #TODO: Implement this
    """

    def __init__(self, double_: np.float64) -> None:
        self.double = double_

    def gen(self) -> str:
        """
        Generate GPULoops code for an EDouble
        """
        if self.double == float("inf"):
            return "float(\"inf\")"
        elif self.double == -float("inf"):
            return "-float(\"inf\")"
        else:
            return str(self.double)


class EVector(Expression):
    """
    An GPULoops Vector

    #TODO: Implement this
    """

    def __init__(self, vector_: Sequence[Expression]) -> None:
        self.vector = vector_

    def gen(self) -> str:
        """
        Generate the GPULoops code for an EVector
        """
        return "[" + ", ".join([e.gen() for e in self.vector]) + "]"


class EArray(Expression):
    """
    An GPULoops Array

    #TODO: Implement this
    """

    def __init__(self, array_: Sequence[Expression]) -> None:
        self.array = array_

    def gen(self) -> str:
        """
        Generate the HiFiber code for an EArray

        #TODO: Implement this
        """
        return "[" + ", ".join([e.gen() for e in self.array]) + "]"
