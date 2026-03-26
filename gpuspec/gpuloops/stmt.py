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

GPULoops AST and code generation for statements in C/C++
"""

from typing import List, Optional, Tuple

from gpuspec.gpuloops.base import Argument, Assignable, Declaration, Expression, Operator, Payload, Statement
from gpuspec.gpuloops.expr import EVar


class SAssignEmpty(Statement):
    """
    An assignment with no initialization
    """

    def __init__(self, assn: Assignable) -> None:
        self.assn = assn

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SAssignEmpty
        """
        return "    " * depth + self.assn.gen() + ";"


class SAssign(Statement):
    """
    An assignment with an expression
    """

    def __init__(self, assn: Assignable, expr: Expression) -> None:
        self.assn = assn
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SAssign
        """
        return "    " * depth + self.assn.gen() + " = " + self.expr.gen() + ";"


class SAssignObj(Statement):
    """
    An assignment for object
    """

    def __init__(self, assn: Assignable, expr: Expression) -> None:
        self.assn = assn
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SAssignObj
        """
        return "    " * depth + self.assn.gen() + self.expr.gen() + ";"


class SAssignTypename(Statement):
    """
    An assignment for typename
    """

    def __init__(self, assn: Assignable, expr: Expression) -> None:
        self.assn = assn
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SAssignTypename
        """
        return "    " * depth + "using " + self.assn.gen() + " = " + \
            self.expr.gen() + ";"


class SBlock(Statement):
    """
    A block of statements
    """

    def __init__(self, stmts: List[Statement]) -> None:
        self.stmts = stmts

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SBlock
        """
        return "\n".join([s.gen(depth) for s in self.stmts])

    def add(self, stmt: Statement) -> None:
        """
        Add a statement onto the end of the SBlock, combine if the new
        statement is also an SBlock
        """
        if isinstance(stmt, SBlock):
            self.stmts.extend(stmt.stmts)
        else:
            self.stmts.append(stmt)


class SDecl(Statement):
    """
    A statement that is a declaration
    """

    def __init__(self, decl: Declaration) -> None:
        self.decl = decl

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SDecl
        """
        return "    " * depth + self.decl.gen()


class SExpr(Statement):
    """
    A statement that is an expression
    """

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SExpr
        """
        return "    " * depth + self.expr.gen() + ";"


class SFunc(Statement):
    """
    A function definition for C/C++
    """

    def __init__(
            self,
            return_type: str,
            name: str,
            args: List[Argument],
            body: Statement,
            templates: Optional[List[str]] = None,
            declaration: Optional[List[str]] = None) -> None:

        self.return_type = return_type
        self.name = name
        self.args = args
        self.body = body
        self.templates = templates
        self.declaration = declaration

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SFunc
        """

        # Construct string for templates
        templates_str = ""
        if self.templates:
            template_params = [f"typename {t}" for t in self.templates]
            templates_str = f"template <{(", ".join(template_params))}>\n"

        # Construct string for declaration
        declaration_str = ""
        if self.declaration:
            declaration_str = " ".join([d for d in self.declaration])
            declaration_str += " "

        # Construct string for args
        args_str = ""
        if self.args:
            args = [arg.gen() for arg in self.args]
            args_str = ", ".join([a for a in args])

        return templates_str + declaration_str + self.return_type + " " + \
            self.name + "(" + args_str + ") {\n" + self.body.gen(depth + 1) + "\n}\n"


class SPrint(Statement):
    """
    A print (std::cout) statement
    """

    def __init__(self, items: List[str]) -> None:
        self.items = items

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SPrint
        """

        return "    " * depth + "std::cout << " + \
            (" << ".join([i for i in self.items])) + " << std::endl;"


class SRangeFor(Statement):
    """
    A range-based for loop
    """

    def __init__(
            self,
            payload: Payload,
            expr: Expression,
            body: SBlock) -> None:
        self.payload = payload
        self.expr = expr
        self.body = body

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SRangeFor
        """
        return "    " * depth + "for (" + \
            self.payload.gen(False) + " : " + self.expr.gen() + \
            ") {\n" + self.body.gen(depth + 1) + "}\n"


class SReturn(Statement):
    """
    A return statement for the end of a function
    """

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the C/C++ output for an SReturn
        """
        return "    " * depth + "return " + self.expr.gen() + ";"
