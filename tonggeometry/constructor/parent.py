r"""Constructors for every step in the creation of an geometry diagram.

The constructor module is for contructing graphical elements in the canvas.
Each constructor function, after called, will draw an geometrical element.
The validity of an action is defined as properly fed input and output, each of
which cannnot exist already.

Also we can use analytical methods for checking if an action can be performed,
basically checking if the action is valid in itself (static check). We also
will check if any output already exists in the diagram (runtime check). Those
checks can be performed with analytical methods, but facts cannot be created
via neumerical check.

This file only defines the parent class.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

import tonggeometry.action
from tonggeometry.inference_engine.predicate import Predicate

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram

__all__ = ['Constructor', 'ErrorMessages']

(DISTINCT, POINT, INVALID, USED, SAMELINE, EXIST, CANNOT, NOINTERSECT,
 SSUCCESS, RSUCCESS) = range(10)

ErrorMessages = {
    DISTINCT: "The points must be distinct and of the right length.",
    POINT: "All points must exist.",
    INVALID: "The output size is invalid.",
    USED: "The output is already used.",
    SAMELINE: "The points are on the same line.",
    EXIST: "The construct already exists.",
    CANNOT: "Cannot create the construct.",
    NOINTERSECT: "No intersection.",
    SSUCCESS: "Static check success.",
    RSUCCESS: "Runtime check success.",
}


class Constructor(ABC):
    """Abtract base class for all constructors.

    Naming convention: from_names should be listed from small to large
    except the anchor point. Note that to_names are set following the
    alphabetical order. And when new points are added, they should be named
    sequentially from characters in to_names. Note that to_names_len must be
    fixed for a constructor. Could not vary from case to case.

    Lines should be named from the latest point, meaning that all lines
    should be created with the latest point as the first, and going from
    the oldest point to the second to the latest.
    """
    from_names_len = 1
    to_names_len = 1

    @staticmethod
    @abstractmethod
    def compute(froms: List) -> Tuple:
        """Compute the new point."""
        return ()

    @staticmethod
    @abstractmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        """Update the diagram with the new construction."""

    @staticmethod
    @abstractmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        "Return the given facts induced from the diagram."
        return []

    @classmethod
    def static_check(cls,
                     diagram: 'Diagram',
                     from_names: str,
                     to_names: Optional[str] = None,
                     draw_only: bool = False,
                     pick_rep: bool = False) -> Tuple[bool, int]:
        """Return True if the construction is valid in static analysis.

        Consider both from_names and to_names validity. Also check for lines and
        circles, if the representative element is used, otherwise, pass. When
        checking if can be computed / drawn, can be done analytically.
        """
        # check if input and output are of valid forms
        g2i, message = cls.good_to_in(diagram, from_names, to_names)
        if not g2i:
            return False, message

        # check if points can be drawn
        g2d, message = cls.good_to_draw(diagram, from_names, draw_only)
        if not g2d:
            return False, message

        if pick_rep:
            # check if the from_names objects have been cached
            g2e, message = cls.good_to_rep(diagram, from_names)
            if not g2e:
                return False, message

        return True, SSUCCESS

    @classmethod
    def good_to_in(cls,
                   diagram: 'Diagram',
                   from_names: str,
                   to_names: Optional[str] = None) -> Tuple[bool, int]:
        """Check if the input and output are of valid forms."""
        if len(from_names) != cls.from_names_len or (cls.from_names_len > 1
                                                     and len(set(from_names))
                                                     != cls.from_names_len):
            return False, DISTINCT
        if not all(p_name in diagram.point_dict for p_name in from_names):
            return False, POINT
        if to_names:
            if cls.to_names_len == 0:
                return False, INVALID
            if len(to_names) != cls.to_names_len or (cls.to_names_len > 1
                                                     and len(set(to_names))
                                                     != cls.to_names_len):
                return False, INVALID
            if any(p_name in diagram.point_dict for p_name in to_names):
                return False, USED
        return True, SSUCCESS

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        """Check if the from_names objects are the representatives of eqclass.
        Duplicates will be removed and only lines and circles need rep check,
        not triangles."""
        return True, SSUCCESS

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        """Check if the objects can be drawn."""
        return True, SSUCCESS

    @staticmethod
    def runtime_check(diagram: 'Diagram', to: Tuple) -> Tuple[bool, int]:
        """Return True if the construction is valid during runtime. Generally,
        the construction only creates one new point. Note that OOB_terminal flag
        is set for checking if points / circles are out of bound. And the
        full_dup label is maintained for checking if every creation already
        exists in the diagram. Need to implement for new constructors if they
        have more than one new object.

        Check if points / lines / circles with the new property already exists.
        """
        p = to[0]
        if p is None:
            return False, CANNOT
        if p in diagram.point_dict.values():
            diagram.full_dup = True
            return False, EXIST
        if abs(p.x) >= 2000 or abs(p.y) >= 2000:
            diagram.OOB_terminal = True
            return False, CANNOT
        return True, RSUCCESS

    @classmethod
    def valid_actions(cls,
                      diagram: 'Diagram',
                      new_points: str = "",
                      pick_rep: bool = True,
                      from_circle: bool = False,
                      order: bool = True):
        """Return the valid actions that can be taken on the diagram.

        Note that order shall be preserved in new_points.

        If proper order is assumed, there is no need to check action duplicate,
        as it is impossible to happen. Acitons are always configured such that
        order is preserved in from_names.

        Note that returned actions shall be ordered, making the creation order
        consistent with lt, convert, and good_to_rep.
        """
        if len(diagram.all_names) < cls.to_names_len:
            return []
        valid = []
        if not from_circle:
            if order:
                new_points = "".join(sorted(new_points))
            new_from_names_list = cls.new_from_names(diagram, new_points)
        else:
            new_from_names_list = cls.new_from_circle(diagram, new_points)
        for new_from_names in new_from_names_list:
            from_names = "".join(new_from_names)
            sgo, _ = cls.static_check(diagram, from_names, pick_rep=pick_rep)
            if not sgo:
                continue
            valid.append(
                tonggeometry.action.Action(cls,
                                           from_names,
                                           s_check=True,
                                           gen_depth=diagram.depth))
        return valid

    @staticmethod
    def new_from_names(diagram: 'Diagram', new_points: str) -> List[Tuple]:
        """Create new from_names for construction based on the incremental new
        points. Names shall be ordered (representative).

        Also, as actions shall be ordered for search pruning, we need to check
        the action generation order here so that we know which action is created
        earlier.

        The idea is that in a normalized representation, action sequence is
        monotonically increasing. There will not be symmetry issue (invariance)
        anymore in this representation.
        """
        return [()]

    @staticmethod
    def new_from_circle(diagram: 'Diagram', circle: str) -> List[Tuple]:
        """Create new from_names for construction based on the new circle."""
        return [()]

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        """Convert the from_names and to_names to follow the minimum convention
        used in new from names."""
        return from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        """Return if A's from_names is earlier than B's from_names when they are
        created in the same batch."""
        num_new_A = sum(int(name in new_points) for name in from_names_A)
        num_new_B = sum(int(name in new_points) for name in from_names_B)
        if num_new_A != num_new_B:
            return num_new_A < num_new_B
        return from_names_A < from_names_B
