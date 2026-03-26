"""
MIT License

Copyright (c) 2021 University of Illinois

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

GPULoops AST and code generation for expressions in C/C++
"""

import numpy as np

from typing import Dict, Sequence

from gpuspec.gpuloops.base import Argument, Expression, Operator


class EAccess(Expression):
    """
    An access into a list or dictionary
    """

    def __init__(self, obj: Expression, ind: Expression) -> None:
        self.obj = obj
        self.ind = ind

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EAccess
        """
        return self.obj.gen() + "[" + self.ind.gen() + "]"


class EBinOp(Expression):
    """
    An GPULoops binary operation
    """

    def __init__(
            self,
            expr1: Expression,
            op: Operator,
            expr2: Expression) -> None:
        self.expr1 = expr1
        self.op = op
        self.expr2 = expr2

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EBinOp
        """
        return self.expr1.gen() + " " + self.op.gen() + " " + self.expr2.gen()


class EBool(Expression):
    """
    An GPULoops boolean variable
    """

    def __init__(self, bool_: bool) -> None:
        self.bool = bool_

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EBool
        """
        return str(self.bool)


class EField(Expression):
    """
    An GPULoops object field access
    """

    def __init__(self, obj: str, field: str):
        self.obj = obj
        self.field = field

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EField
        """
        return self.obj + "." + self.field


class EFloat(Expression):
    """
    An GPULoops float (32-bit)

    #TODO: Implement this
    """

    def __init__(self, float_: np.float32) -> None:
        self.float = float_

    def gen(self) -> str:
        """
        Generate C/C++ code for an EFloat
        """
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
        Generate C/C++ code for an EDouble
        """
        return str(self.double)


class EFunc(Expression):
    """
    An GPULoops function call
    """

    def __init__(self, name: str, args: Sequence[Argument]) -> None:
        self.name = name
        self.args = args

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EFunc
        """
        return self.name + \
            "(" + ", ".join([a.gen() for a in self.args]) + ")"


class EInt(Expression):
    """
    An GPULoops integer
    """

    def __init__(self, int_: int) -> None:
        self.int = int_

    def gen(self) -> str:
        """
        Generate C/C++ code for an EInt
        """
        return str(self.int)


class EVector(Expression):
    """
    An GPULoops Vector
    """

    def __init__(self, vector_: Sequence[Expression]) -> None:
        self.vector = vector_

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EVector
        """
        return "[" + ", ".join([e.gen() for e in self.vector]) + "]"


class EArray(Expression):
    """
    An GPULoops Array
    """

    def __init__(self, array_: Sequence[Expression]) -> None:
        self.array = array_

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EArray
        """
        return "[" + ", ".join([e.gen() for e in self.array]) + "]"


class EMethod(Expression):
    """
    An GPULoops method call
    """

    def __init__(self, obj: Expression, name: str,
                 args: Sequence[Argument]) -> None:
        self.obj = obj
        self.name = name
        self.args = args

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EMethod
        """
        return self.obj.gen() + "." + self.name + \
            "(" + ", ".join([a.gen() for a in self.args]) + ")"


class EParens(Expression):
    """
    An GPULoops expression surrounded by parentheses
    """

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EParens
        """
        return "(" + self.expr.gen() + ")"


class EString(Expression):
    """
    A string in GPULoops
    """

    def __init__(self, string: str) -> None:
        self.string = string

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EString
        """
        return "\"" + self.string + "\""


class EVar(Expression):
    """
    An GPULoops variable
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def gen(self) -> str:
        """
        Generate the C/C++ code for an EVar
        """
        return self.name
