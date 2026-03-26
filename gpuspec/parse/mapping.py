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
        work_atom: List[str]
        work_unit: List[str] | Dict[str, Dict[Tree, List[Tree]]]
        work_tile: List[str] | Dict[str, Dict[Tree, List[Tree]]]

        if yaml is not None and "mapping" in yaml.keys() and \
                yaml["mapping"] is not None:
            mapping = yaml["mapping"]

            if "work_atom" in mapping.keys():
                work_atom = mapping["work_atom"]

            if "work_unit" in mapping.keys():
                if isinstance(mapping["work_unit"], list):
                    work_unit = mapping["work_unit"]
                elif isinstance(mapping["work_unit"], dict):
                    work_unit = {}

                    for tensor, ranks in mapping["work_unit"].items():
                        work_unit[tensor] = {}

                        if ranks is None:
                            continue

                        for ranks_str, parts in ranks.items():
                            ranks_tree = PartitioningParser.parse_ranks(
                                ranks_str)
                            work_unit[tensor][ranks_tree] = []
                            for part in parts:
                                work_unit[tensor][ranks_tree].append(
                                    PartitioningParser.parse_partitioning(part))
                else:
                    raise KeyError(
                        f"Invalid type of work_unit: '{mapping["work_unit"]}', "
                        "must be a \'list\' of ranks or a \'dict\' of rank and partitioning "
                        "mapping information")

            if "work_tile" in mapping.keys():

                if isinstance(mapping["work_tile"], list):
                    work_tile = mapping["work_tile"]
                elif isinstance(mapping["work_tile"], dict):
                    work_tile = {}

                    for tensor, ranks in mapping["work_tile"].items():
                        work_tile[tensor] = {}

                        if ranks is None:
                            continue

                        for ranks_str, parts in ranks.items():
                            ranks_tree = PartitioningParser.parse_ranks(
                                ranks_str)
                            work_tile[tensor][ranks_tree] = []
                            for part in parts:
                                work_tile[tensor][ranks_tree].append(
                                    PartitioningParser.parse_partitioning(part))
                else:
                    raise KeyError(
                        f"Invalid type of work_tile: '{mapping["work_tile"]}', "
                        "must be a \'list\' of ranks or a \'dict\' of rank and partitioning "
                        "mapping information")

        if work_atom is None:
            raise KeyError(f"Undefined work_atomt!")
        else:
            self.work_atom = work_atom

        if work_unit is None:
            raise KeyError(f"Undefined work_unit!")
        else:
            self.work_unit = work_unit

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

    def get_work_atom(self) -> List[str]:
        """
        Get the work_atom information
        """
        return self.work_atom

    def get_work_unit(self) -> List[str] | Dict[str, Dict[Tree, List[Tree]]]:
        """
        Get the work_unit information
        """
        return self.work_unit

    def get_work_tile(self) -> List[str] | Dict[str, Dict[Tree, List[Tree]]]:
        """
        Get the work_tile information
        """
        return self.work_tile

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Mappings
        """
        if isinstance(other, type(self)):
            return self.work_atom == other.work_atom and \
                self.work_unit == other.work_unit and \
                self.work_tile == other.work_tile
        return False
