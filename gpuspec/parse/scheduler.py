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

Parse the input YAML for the GPULoops scheduler
"""

from gpuspec.parse.yaml import YamlParser


class Scheduler:
    """
    Parse the input YAML for the GPULoops scheduler
    """

    def __init__(self, yaml: dict) -> None:
        """
        Read the YAML input
        """
        # Parse the scheduler
        self.scheduler = yaml["scheduler"]

        # Validate the scheduler name
        valid_schedulers = ["thread-mapped", "group-mapped", "work-oriented"]
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"Invalid scheduler '{self.scheduler}'. "
                f"Must be one of: {', '.join(valid_schedulers)}"
            )

    @classmethod
    def from_file(cls, filename: str) -> "Scheduler":
        """
        Construct a new GPULoops scheduler from a YAML file
        """
        return cls(YamlParser.parse_file(filename))

    @classmethod
    def from_str(cls, string: str) -> "Scheduler":
        """
        Construct a new GPULoops scheduler from a string in the YAML format
        """
        return cls(YamlParser.parse_str(string))

    def get_scheduler(self) -> str:
        """
        Get the scheduler
        """
        return self.scheduler

    def __eq__(self, other: object) -> bool:
        """
        The == operator for Scheduler
        """
        if isinstance(other, type(self)):
            return self.scheduler == other.scheduler
        return False
