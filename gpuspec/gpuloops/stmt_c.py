"""
INSERT LICENSE HERE

Additional implementation for statements targetted for C/C++ code
"""

from typing import List, Optional

from gpuspec.gpuloops.base import Argument, Assignable, Expression, Declaration, Statement


class SDecl(Statement):
    """
    A statement that is a declaration
    """

    def __init__(self, decl: Declaration) -> None:
        self.decl = decl

    def gen(self, depth: int) -> str:
        """
        Generate the GPULoops output for an SDecl
        """
        return "    " * depth + self.decl.gen()


class SPrint(Statement):
    """
    A print (std::cout) statement
    """

    def __init__(self, items: List[str]) -> None:
        self.items = items

    def gen(self, depth: int) -> str:
        """
        Generate the GPULoops output for an SPrint
        """

        return "    " * depth + "std::cout << " + \
            (" << ".join([i for i in self.items])) + " << std::endl;"


class SAssignEmpty_C(Statement):
    """
    An assignment with no initialization
    """

    def __init__(self, assn: Assignable) -> None:
        self.assn = assn

    def gen(self, depth: int) -> str:
        """
        Generate the GPULoops output for an SAssignEmpty_C
        """
        return "    " * depth + self.assn.gen() + ";"


class SAssign_C(Statement):
    """
    An assignment with an expression
    """

    def __init__(self, assn: Assignable, expr: Expression) -> None:
        self.assn = assn
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the GPULoops output for an SAssign
        """
        return "    " * depth + self.assn.gen() + " = " + self.expr.gen() + ";"


class SAssignObj_C(Statement):
    """
    An assignment for object
    """

    def __init__(self, assn: Assignable, expr: Expression) -> None:
        self.assn = assn
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the GPULoops output for an SAssignObj_C
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
        Generate the GPULoops output for an SAssign
        """
        return "    " * depth + "using " + self.assn.gen() + " = " + \
            self.expr.gen() + ";"


class SExpr_C(Statement):
    """
    A statement that is an expression (C/C++)
    """

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def gen(self, depth: int) -> str:
        """
        Generate the GPULoops output for an SExpr
        """
        return "    " * depth + self.expr.gen() + ";"


class SFunc_C(Statement):
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
