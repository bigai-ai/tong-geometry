r"""The database module. Database is a collection of facts."""

from typing import TYPE_CHECKING, List, Optional

from tonggeometry.inference_engine.handler import ALL_HANDLERS
from tonggeometry.inference_engine.primitives import (Angle, Circle, Ratio,
                                                      Segment, Triangle)

if TYPE_CHECKING:
    from tonggeometry.inference_engine.predicate import Fact, Predicate


class Database:
    """Database is a collection of facts.

    The goal of the database is to handle eqclasses. There are transitive
    relations, such as, congruent segments, parallel, equal angles,
    equal ratios, similar triangles, and congruent triangles. The transitive
    relations are represented with a representative and a set of equivalents,
    including the representative. midpoint and perpendicular are however not
    transitive, and is represented with an element and all others where the
    relation holds. Note that lines and circles are also in this case. An
    inverse index is maintained for all relations. For the transitive relation,
    the inverse is the element to its representative. For non-transitive, the
    inverse index is also built. Note that perpendicular is also reflexive but,
    non-transitive, hence its inverse index is itself. lines and circles are
    built with eqline and eqcircle; they are transitive as well.

    Auxiliary data structures are also maintained for fast search.

    Each index and its inverse shall be updated at the same time.

    'Fact' object order does not matter as fact key is ordered. Primitive key is
    also ordered. For printing and searching, absolute order is not as important
    as relative fixed order, hence only OrderedSet is used.

    Note a general issue: points are added in handler's intermediate data
    structures earlier than facts during reasoning, such that loop in tracing
    facts could potentially happen, where a fact depends on facts not created,
    consider using fact check rather than point check to resolve the issue.
    """

    def __init__(self):
        for handler in ALL_HANDLERS.values():
            handler.init(self)

    def predicate_to_fact(self, predicate: 'Predicate') -> 'Fact':
        """Turn raw primitive into structured fact."""
        return ALL_HANDLERS[predicate.type].predicate_to_fact(predicate)

    def add_fact(self, fact: 'Fact') -> List['Fact']:
        """Add a new fact into the database."""
        return ALL_HANDLERS[fact.type].add_fact(self, fact)

    def filter(self, fact: 'Fact') -> Optional['Fact']:
        """Filter trivial facts."""
        return ALL_HANDLERS[fact.type].filter(self, fact)

    def is_eqline(self, s1: Segment, s2: Segment) -> bool:
        """Check if two segments are on the same line."""
        return ALL_HANDLERS["eqline"].known(self, s1, s2)

    def is_eqcircle(self, c1: Circle, c2: Circle) -> bool:
        """Check if two circles are the same."""
        return ALL_HANDLERS["eqcircle"].known(self, c1, c2)

    def is_cong(self, s1: Segment, s2: Segment) -> bool:
        """Check if two segments are of the same length."""
        return ALL_HANDLERS["cong"].known(self, s1, s2)

    def is_midp(self, m: str, s: Segment) -> bool:
        """Check if a point is the midpoint of a segment."""
        return ALL_HANDLERS["midp"].known(self, m, s)

    def is_para(self, s1: Segment, s2: Segment) -> bool:
        """Check if two segments are parallel."""
        return ALL_HANDLERS["para"].known(self, s1, s2)

    def is_perp(self, a: Angle) -> bool:
        """Check if the angle formed by two segments is perpendicular."""
        return ALL_HANDLERS["perp"].known(self, a)

    def is_eqangle(self, a1: Angle, a2: Angle) -> bool:
        """Check if two angles are equal."""
        return ALL_HANDLERS["eqangle"].known(self, a1, a2)

    def is_eqratio(self, r1: Ratio, r2: Ratio) -> bool:
        """Check if two ratios are equal."""
        return ALL_HANDLERS["eqratio"].known(self, r1, r2)

    def is_simtri(self, t1: Triangle, t2: Triangle) -> bool:
        """Check if two triangles are similar."""
        return ALL_HANDLERS["simtri"].known(self, t1, t2)

    def is_contri(self, t1: Triangle, t2: Triangle) -> bool:
        """Check if two triangles are congruent."""
        return ALL_HANDLERS["contri"].known(self, t1, t2)

    def itsll(self, s1: Segment, s2: Segment) -> Optional[str]:
        """Return the intersection of two representative lines from the segments
         on the lines."""
        return ALL_HANDLERS["eqline"].itsll(self, s1, s2)

    def itscc(self, c1: Circle, c2: Circle) -> Optional[List]:
        """Return the intersection of two representative circles from the
        circles on the circles."""
        return ALL_HANDLERS["eqcircle"].itscc(self, c1, c2)

    def __repr__(self) -> str:
        s = "\nDatabase\n"

        for handler in ALL_HANDLERS.values():
            s += handler.stringify(self)

        return s
