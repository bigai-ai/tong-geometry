r"""Predicate class that holds raw predicate info."""

from functools import cached_property
from typing import TYPE_CHECKING, List

import tonggeometry.inference_engine.handler
from tonggeometry.inference_engine.primitives import (Angle, Circle, Ratio,
                                                      Segment, Triangle)
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def string_to_fact(fact_string: str):
    """Turn fact string into fact object."""
    l = fact_string.strip()
    left_bracket = l.index("(")
    right_bracket = l.index(")")
    fact_type = l[:left_bracket].strip()
    if fact_type == "eqline":
        s1, s2 = l[left_bracket + 1:right_bracket].split(", ")
        fact = Fact("eqline", [Segment(*s1), Segment(*s2)])
    elif fact_type == "eqcircle":
        ctr1, cir1, ctr2, cir2 = l[left_bracket + 1:right_bracket].split(", ")
        if ctr1 == "None":
            ctr1 = None
        if ctr2 == "None":
            ctr2 = None
        fact = Fact("eqcircle", [Circle(ctr1, [*cir1]), Circle(ctr2, [*cir2])])
    elif fact_type == "perp":
        a = l[left_bracket + 1:right_bracket]
        fact = Fact("perp", [Angle(*a)])
    elif fact_type == "cong":
        s1, s2 = l[left_bracket + 1:right_bracket].split(", ")
        fact = Fact("cong", [Segment(*s1), Segment(*s2)])
    elif fact_type == "para":
        s1, s2 = l[left_bracket + 1:right_bracket].split(", ")
        fact = Fact("para", [Segment(*s1), Segment(*s2)])
    elif fact_type == "eqangle":
        a1, a2 = l[left_bracket + 1:right_bracket].split(", ")
        fact = Fact("eqangle", [Angle(*a1), Angle(*a2)])
    elif fact_type == "eqratio":
        s1, s2, s3, s4 = l[left_bracket + 1:right_bracket].split(", ")
        fact = Fact("eqratio", [
            Ratio(Segment(*s1), Segment(*s2)),
            Ratio(Segment(*s3), Segment(*s4))
        ])
    elif fact_type == "midp":
        m, s = l[left_bracket + 1:right_bracket].split(", ")
        fact = Fact("midp", [m, Segment(*s)])
    elif fact_type == "simtri":
        t1, t2 = l[left_bracket + 1:right_bracket].split(", ")
        fact = Fact("simtri", [Triangle(*t1), Triangle(*t2)])
    elif fact_type == "contri":
        t1, t2 = l[left_bracket + 1:right_bracket].split(", ")
        fact = Fact("contri", [Triangle(*t1), Triangle(*t2)])
    return fact


def fact_transform(diagram: 'Diagram', fact: 'Fact'):
    """Transform fact to the representative."""
    if fact.type == "eqangle":
        a1, a2 = fact.objects
        A, B, C = a1.name
        D, E, F = a2.name
        l_AB = Segment(A, B)
        l_AB_rep = diagram.database.inverse_eqline[l_AB]
        it = iter(diagram.database.lines_points[l_AB_rep])
        pA = next(it)
        if pA == B:
            pA = next(it)
        l_BC = Segment(B, C)
        l_BC_rep = diagram.database.inverse_eqline[l_BC]
        it = iter(iter(diagram.database.lines_points[l_BC_rep]))
        pC = next(it)
        if pC == B:
            pC = next(it)
        l_DE = Segment(D, E)
        l_DE_rep = diagram.database.inverse_eqline[l_DE]
        it = iter(diagram.database.lines_points[l_DE_rep])
        pD = next(it)
        if pD == E:
            pD = next(it)
        l_EF = Segment(E, F)
        l_EF_rep = diagram.database.inverse_eqline[l_EF]
        it = iter(diagram.database.lines_points[l_EF_rep])
        pF = next(it)
        if pF == E:
            pF = next(it)
        return Fact("eqangle", [Angle(pA, B, pC), Angle(pD, E, pF)])
    if fact.type == "para":
        s1, s2 = fact.objects
        l_AB = s1
        l_AB_rep = diagram.database.inverse_eqline[l_AB]
        it = iter(diagram.database.lines_points[l_AB_rep])
        pA = next(it)
        pB = next(it)
        l_CD = s2
        l_CD_rep = diagram.database.inverse_eqline[l_CD]
        it = iter(diagram.database.lines_points[l_CD_rep])
        pC = next(it)
        pD = next(it)
        return Fact("para", [Segment(pA, pB), Segment(pC, pD)])
    return fact


def get_fact_dep(fact: 'Fact') -> str:
    """Get the dependency point for a fact"""
    out = ""
    for obj in fact.objects:
        if isinstance(obj, str):
            out += obj
        else:
            out += obj.dependency
    return out


class Predicate:
    """Predicate is simple fact based on points, such as para(A,B,C,D) where
    A, B, C, D are points. The parent attribute is for facts needed to
    reach this fact."""

    def __init__(
            self,
            type: str,  # pylint: disable=redefined-builtin
            objects: List,
            fn: str = ""):
        self.type = type
        self.objects = objects
        self.parents = []
        self.fn = fn
        self.dependency = ""
        self.edges = OrderedSet()
        self.depth = 0
        # self.nodes = {}

    @cached_property
    def key(self) -> str:
        """Key of Fact."""
        return tonggeometry.inference_engine.handler.ALL_HANDLERS[
            self.type].fact_key(self)

    def add_parent(self, fact: 'Predicate'):
        """Add a parent fact."""
        self.parents.append(fact)

    def add_parents(self, facts: List['Predicate']):
        """Add a list of parent facts."""
        self.parents += facts

    def __repr__(self) -> str:
        return self.key

    def __eq__(self, other: 'Predicate') -> bool:
        return isinstance(other, Predicate) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __lt__(self, other: 'Predicate') -> bool:
        """For sorting purposes."""
        if self.type != other.type:
            types = tonggeometry.inference_engine.handler.ORDER
            return types[self.type] < types[other.type]

        return str(self) < str(other)


# Fact uses primitives; Predicate uses points.
Fact = Predicate


class OrderedFact:
    """A collection of ordered facts."""

    def __init__(self):
        self.keys = tonggeometry.inference_engine.handler.ORDER.keys()
        self.order = OrderedSet(
            zip(self.keys, [OrderedSet() for _ in range(len(self.keys))]))
        self.num = 0

    def __contains__(self, fact: Fact) -> bool:
        """Check if the collection contains a fact."""
        return fact in self.order[fact.type]

    def __iter__(self):
        """Iterate over the collection."""
        yield from self.order

    def __getitem__(self, key: str):
        return self.order[key]

    def __repr__(self) -> str:
        s = ""
        for key in self.order:
            for fact in self.order[key]:
                s += f"{fact}\n"
        return s

    def __len__(self) -> int:
        return self.num

    def add(self, fact: Fact):
        """Add a fact to the collection. Val is must_keep (not filered) as it is
        from database adding and must be added to parents."""
        if fact not in self.order[fact.type]:
            self.order[fact.type][fact] = fact.fn == "add_fact"
            self.num += 1
        elif fact.fn == "add_fact":
            self.order[fact.type][fact] = True

    def first(self) -> Fact:
        """Get the first fact."""
        for key in self.order:
            for fact in self.order[key]:
                must_keep = self.order[key].pop(fact)
                self.num -= 1
                return fact, must_keep

    def is_empty(self) -> bool:
        """Check if the collection is empty."""
        return self.num == 0
