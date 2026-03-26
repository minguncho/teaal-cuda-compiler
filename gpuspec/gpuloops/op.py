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

GPULoops AST and code generation for operators in C/C++
"""

from gpuspec.gpuloops.base import Operator


class OAdd(Operator):
    """
    The GPULoops addition operator
    """

    def gen(self) -> str:
        """
        Generate the C/C++ code for the OAdd operator
        """
        return "+"


class OAnd(Operator):
    """
    The GPULoops and operator
    """

    def gen(self) -> str:
        """
        Generate the C/C++ code for the OAnd operator
        """
        return "&"


class ODiv(Operator):
    """
    The GPULoops divide operator
    """

    def gen(self) -> str:
        """
        Generate the C/C++ code for the ODiv operator
        """
        return "/"


class OEqEq(Operator):
    """
    The GPULoops equal-equal operator
    """

    def gen(self) -> str:
        """
        Generate the C/C++ code for the OEqEq operator
        """
        return "=="


class OLt(Operator):
    """
    The GPULoops less-than less-than operator
    """

    def gen(self) -> str:
        """
        Generate the C/C++ code for the OLt operator
        """
        return "<"


class OMod(Operator):
    """
    The GPULoops modulo operator
    """

    def gen(self) -> str:
        """
        Generate the C/C++ code for the OMod operator
        """
        return "%"


class OMul(Operator):
    """
    The GPULoops multiplication operator
    """

    def gen(self) -> str:
        """
        Generate the C/C++ code for the OMul operator
        """
        return "*"


class OOr(Operator):
    """
    The GPULoops or operator
    """

    def gen(self) -> str:
        """
        Generate the C/C++ code for the OOr operator
        """
        return "|"


class OSub(Operator):
    """
    The GPULoops subtract operator
    """

    def gen(self) -> str:
        """
        Generate the C/C++ code for the OSub operator
        """
        return "-"
