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

Parse the input YAML for the mapping
"""

from lark.tree import Tree
from typing import Dict, List, Optional

from gpuspec.parse.partitioning import PartitioningParser
from gpuspec.parse.yaml import YamlParser


class Mapping:
    """
    Parse the input YAML for the mapping
    """

    def __init__(self, yaml: Optional[dict]) -> None:
        """
        Read the YAML input
        """
        work_quark: List[str]
        work_atom: Dict[Tree, List[Tree]]
        work_tile: Dict[Tree, List[Tree]]

        if yaml is not None and "mapping" in yaml.keys() and \
                yaml["mapping"] is not None:
            mapping = yaml["mapping"]

            if "work_quark" in mapping.keys():
                work_quark = mapping["work_quark"]

            if "work_atom" in mapping.keys():
                work_atom = {}
                for ranks_str, parts in mapping["work_atom"].items():
                    if len(parts) > 1:
                        raise ValueError(
                            "Current version only supports 1 level of partition of work atom!")

                    ranks_tree = PartitioningParser.parse_ranks(ranks_str)
                    work_atom[ranks_tree] = []
                    for part in parts:
                        work_atom[ranks_tree].append(
                            PartitioningParser.parse_partitioning(part))

            if "work_tile" in mapping.keys():
                work_tile = {}
                for ranks_str, parts in mapping["work_tile"].items():
                    if len(parts) > 1:
                        raise ValueError(
                            "Current version only supports 1 level of partition of work tile!")

                    ranks_tree = PartitioningParser.parse_ranks(ranks_str)
                    work_tile[ranks_tree] = []
                    for part in parts:
                        work_tile[ranks_tree].append(
                            PartitioningParser.parse_partitioning(part))

        if work_quark is None:
            raise KeyError(f"Undefined work_quark!")
        else:
            self.work_quark = work_quark

        if work_atom is None:
            raise KeyError(f"Undefined work_atom!")
        else:
            self.work_atom = work_atom

        if work_tile is None:
            raise KeyError(f"Undefined work_tile!")
        else:
            self.work_tile = work_tile

    @classmethod
    def from_file(cls, filename: str) -> "Mapping":
        """
        Construct a new Mapping from a YAML file
        """
        return cls(YamlParser.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Mapping":
        """
        Construct a new Mapping from a string in the YAML format
        """
        return cls(YamlParser.parse_str(string))

    def get_work_quark(self) -> List[str]:
        """
        Get the work_quark information
        """
        return self.work_quark

    def get_work_atom(self) -> Dict[Tree, List[Tree]]:
        """
        Get the work_atom information
        """
        return self.work_atom

    def get_work_tile(self) -> Dict[Tree, List[Tree]]:
        """
        Get the work_tile information
        """
        return self.work_tile

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Mappings
        """
        if isinstance(other, type(self)):
            return self.work_quark == other.work_quark and \
                self.work_atom == other.work_atom and \
                self.work_tile == other.work_tile
        return False
