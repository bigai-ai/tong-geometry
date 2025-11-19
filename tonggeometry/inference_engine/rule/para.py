r"""Parallel-related rules."""

from typing import TYPE_CHECKING, List

from tonggeometry.constructor.primitives import on_same_line
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def para_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """para imply equal angle from a line."""
    facts = []
    s1, s2 = fact.objects
    if on_same_line(*[diagram.point_dict[p] for p in [s1.p1, s1.p2, s2.p1]]):
        return facts
    for p1, p2 in [(s1.p1, s1.p2), (s1.p2, s1.p1)]:
        for p3, p4 in [(s2.p1, s2.p2), (s2.p2, s2.p1)]:
            a1 = Angle(p3, p1, p2)
            a2 = Angle(p1, p3, p4)
            f = Fact("eqangle", [a1, a2], "para_to_eqangle")
            f.add_parent(fact)
            facts.append(f)
    return facts


def para_to_eqline(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """para and intersection imply collinear."""
    facts = []
    s1, s2 = fact.objects
    if not (s1.p1 in [s2.p1, s2.p2] or s1.p2 in [s2.p1, s2.p2]):
        return facts
    f = Fact("eqline", [s1, s2], "para_to_eqline")
    f.add_parent(fact)
    facts.append(f)
    return facts


def para_and_eqline_to_para(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """para and eqline imply para. One-stop link between any two segs"""
    facts = []
    s1, s2 = fact.objects
    if on_same_line(*[diagram.point_dict[p] for p in [s1.p1, s1.p2, s2.p1]]):
        return facts
    eqclass_rep = diagram.database.inverse_eqline[s1]
    eqclass = diagram.database.eqline[eqclass_rep]
    for s3 in eqclass:
        if s3 == s1:
            continue
        f = Fact("para", [s3, s2], "para_and_eqline_to_para")
        f.add_parent(fact)
        f.add_parent(Fact("eqline", [s1, s3]))
        facts.append(f)
    eqclass_rep = diagram.database.inverse_eqline[s2]
    eqclass = diagram.database.eqline[eqclass_rep]
    for s4 in eqclass:
        if s4 == s2:
            continue
        f = Fact("para", [s1, s4], "para_and_eqline_to_para")
        f.add_parent(fact)
        f.add_parent(Fact("eqline", [s2, s4]))
        facts.append(f)
    return facts


def eqline_and_para_to_para(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """para and eqline imply para. One-stop link between any two segs."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    if s1 in diagram.database.inverse_para:
        eqclass_rep = diagram.database.inverse_para[s1]
        eqclass = diagram.database.para[eqclass_rep]
        for s3 in eqclass:
            if not on_same_line(
                    *[diagram.point_dict[p] for p in [s1.p1, s1.p2, s3.p1]]):
                f = Fact("para", [s3, s2], "eqline_and_para_to_para")
                f.add_parent(fact)
                f.add_parent(Fact("para", [s1, s3]))
                facts.append(f)
    if s2 in diagram.database.inverse_para:
        eqclass_rep = diagram.database.inverse_para[s2]
        eqclass = diagram.database.para[eqclass_rep]
        for s4 in eqclass:
            if not on_same_line(
                    *[diagram.point_dict[p] for p in [s2.p1, s2.p2, s4.p1]]):
                f = Fact("para", [s1, s4], "eqline_and_para_to_para")
                f.add_parent(fact)
                f.add_parent(Fact("para", [s2, s4]))
                facts.append(f)
    return facts
