"""
INSERT LICENSE HERE

Additional implementation for statements targetted for C/C++ code
"""

from typing import List, Optional

from gpuspec.gpuloops.base import Assignable, Expression, Declaration, Statement


class SDecl(Statement):
    """
    A statement that is a declaration
    """

    def __init__(self, decl: Declaration) -> None:
        self.decl = decl

    def gen(self, depth: int) -> str:
        """
        Generate the HiFiber output for an SDecl
        """
        return "    " * depth + self.decl.gen()


class SFunc_C(Statement):
    """
    A function definition for C/C++
    """

    def __init__(
            self,
            return_type: str,
            name: str,
            args: List[Assignable],
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
            template_header = "template <"

            # Add 'typename' in front of each template
            templates = [f"typename {t}" for t in self.templates]

            # Add indentation space to each template
            templates = [templates[0]] + \
                [" " * len(template_header) + t for t in templates[1:]]

            template_contents = ",\n".join(
                [t for t in templates])
            templates_str = f"{template_header}{template_contents}>\n"

        # Construct string for declaration
        declaration = ""
        if self.declaration:
            declaration = " ".join([d for d in self.declaration])
            declaration += " "

        # Construct string for args
        args_str = ""
        if self.args:
            args = [arg.gen() for arg in self.args]
            # 1 for space, 1 for open parenthesis
            args_indent = len(declaration) + \
                len(self.return_type) + len(self.name) + 2

            # Add indentation space to each args
            args = [args[0]] + [" " * args_indent + a for a in args[1:]]
            args_str = ",\n".join([a for a in args])

        return templates_str + declaration + self.return_type + " " + self.name + \
            "(" + args_str + ") {\n" + self.body.gen(depth + 1) + "\n}\n"
