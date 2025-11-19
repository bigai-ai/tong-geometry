r"""Basic Action class."""

from typing import TYPE_CHECKING, Optional, Tuple

import tonggeometry.constructor

if TYPE_CHECKING:
    from tonggeometry.constructor import Constructor
    from tonggeometry.diagram import Diagram


def compare(this: 'Action', that: 'Action', new_points: str):
    """Comparison two non-base actions at the same level."""
    index = tonggeometry.constructor.ConstructorIndex
    this_idx = index[this.constructor_name]
    that_idx = index[that.constructor_name]
    if this_idx != that_idx:
        cmp_bool = this_idx < that_idx
    else:
        cmp_bool = this.constructor.lt(this.from_names, that.from_names,
                                       new_points)
    if cmp_bool:
        return -1
    return 1


def action_to_string(action: 'Action', to_names: str):
    """Turn an action into Action string."""
    action_type = action.constructor_name
    from_names = action.from_names
    if to_names is None:
        to_names = ""
    return f"""Action({action_type}, "{from_names}", "{to_names}")"""


class Action:
    """The Action class.

    to_names is used for tracking dependencies.

    Action is considered the same if the operator is the same and the from_names
    the same, no matter whether the to_names the same.
    """

    def __init__(self,
                 constructor: 'Constructor',
                 from_names: str,
                 to_names: Optional[str] = None,
                 output: Optional[Tuple] = None,
                 s_check: bool = False,
                 gen_depth: int = -1,
                 rev_names: bool = False):
        self.constructor = constructor
        self.constructor_name = self.constructor.__name__
        self.from_names = from_names
        self.to_names = to_names
        self.output = output
        self.s_check = self.output is not None or s_check
        self.gen_depth = gen_depth
        self.rev_names = rev_names

    def __repr__(self) -> str:
        from_names_str = ",".join(self.from_names)
        return f"{self.constructor_name}({from_names_str})"

    def __eq__(self, other: 'Action') -> bool:
        return all([
            self.constructor_name == other.constructor_name,
            self.from_names == other.from_names
        ])

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def convert(self, diagram: 'Diagram') -> 'Action':
        """Order from_names and to_names."""
        self.from_names, self.to_names = self.constructor.convert(
            diagram, self.from_names, self.to_names)
