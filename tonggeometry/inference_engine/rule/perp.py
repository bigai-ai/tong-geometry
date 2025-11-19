r"""Perpendicular-related rules.
    The following theorem is implemented:
    1. Midp of hypotenuse of right triangle is the outer center of the triangle.
"""

from typing import TYPE_CHECKING, List

from tonggeometry.constructor.primitives import on_same_line
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Segment
from tonggeometry.inference_engine.util import four_joint

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def perp_and_perp_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Two perpendiculars are equal. And if shared, can derive eqangle."""
    facts = []
    a = fact.objects[0]
    A, B, C = a.name
    for aa in diagram.database.perp:
        if aa == a:
            continue
        f = Fact("eqangle", [a, aa], "perp_and_perp_to_eqangle")
        f.add_parent(fact)
        f.add_parent(Fact("perp", [aa]))
        facts.append(f)
        aa_p = Angle(*aa.name[::-1])
        f = Fact("eqangle", [a, aa_p], "perp_and_perp_to_eqangle")
        f.add_parent(fact)
        f.add_parent(Fact("perp", [aa]))
        facts.append(f)
        if Segment(aa.name[0], aa.name[2]) in [a.s1, a.s2]:
            D = aa.name[1]
            if Segment(aa.name[0], aa.name[2]) == a.s1:
                AA, CC = C, A
            else:
                AA, CC = A, C
            f = Fact("eqangle",
                     [Angle(AA, B, D), Angle(B, CC, D)],
                     "perp_and_perp_to_eqangle")
            f.add_parent(fact)
            f.add_parent(Fact("perp", [aa]))
            facts.append(f)
        if Segment(A, C) in [aa.s1, aa.s2]:
            D = aa.name[0] if aa.name[0] not in [A, C] else aa.name[2]
            if aa.name[1] == A:
                f = Fact("eqangle",
                         [Angle(B, C, A), Angle(B, A, D)],
                         "perp_and_perp_to_eqangle")
            else:
                f = Fact("eqangle",
                         [Angle(B, C, D), Angle(B, A, C)],
                         "perp_and_perp_to_eqangle")
            f.add_parent(fact)
            f.add_parent(Fact("perp", [aa]))
            facts.append(f)
    return facts


def eqline_and_perp_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """eqline and perp imply perp."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    B = s12_joint
    A, C = s12_p_set
    s_AB = Segment(A, B)
    s_BC = Segment(B, C)
    for s, p in [(s_AB, C), (s_BC, A)]:
        if s not in diagram.database.segments_perps:
            continue
        for a in diagram.database.segments_perps[s]:
            if a.name[1] != B:
                continue
            ss = a.s1 if a.s1 != s else a.s2
            f = Fact("perp", [Angle(p, ss.p1, ss.p2)],
                     "eqline_and_perp_to_perp")
            f.add_parent(fact)
            f.add_parent(Fact("perp", [a]))
            facts.append(f)
    return facts


def perp_and_eqline_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """eqline and perp imply perp."""
    facts = []
    a = fact.objects[0]
    A, B, C = a.name
    s_AB = Segment(A, B)
    s_BC = Segment(B, C)
    l_AB = diagram.database.inverse_eqline[s_AB]
    l_BC = diagram.database.inverse_eqline[s_BC]
    for p in diagram.database.lines_points[l_AB]:
        if p in [A, B]:
            continue
        f = Fact("perp", [Angle(p, B, C)], "perp_and_eqline_to_perp")
        f.add_parent(fact)
        f.add_parent(Fact("eqline", [s_AB, Segment(p, B)]))
        facts.append(f)
    for p in diagram.database.lines_points[l_BC]:
        if p in [B, C]:
            continue
        f = Fact("perp", [Angle(A, B, p)], "perp_and_eqline_to_perp")
        f.add_parent(fact)
        f.add_parent(Fact("eqline", [s_BC, Segment(B, p)]))
        facts.append(f)
    return facts


def perp_and_midp_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """From righttri and midp to congruent."""
    facts = []
    a = fact.objects[0]
    A, H, B = a.name
    s = Segment(A, B)
    if s in diagram.database.inverse_midp:
        M = diagram.database.inverse_midp[s]
        f = Fact("cong", [Segment(H, M), Segment(A, M)],
                 "perp_and_midp_to_cong")
        f.add_parent(fact)
        f.add_parent(Fact("midp", [M, s]))
        facts.append(f)
    return facts


def midp_and_perp_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """From righttri and midp to congruent."""
    facts = []
    M, s = fact.objects
    if s not in diagram.database.h_segments_perps:
        return facts
    A = s.p1
    for a in diagram.database.h_segments_perps[s]:
        H = a.name[1]
        f = Fact("cong", [Segment(H, M), Segment(A, M)],
                 "midp_and_perp_to_cong")
        f.add_parent(fact)
        f.add_parent(Fact("perp", [a]))
        facts.append(f)
    return facts


def perp_and_eqline_and_cong_to_cong(diagram: 'Diagram',
                                     fact: Fact) -> List[Fact]:
    """From righttri and cong to cong."""
    facts = []
    a = fact.objects[0]
    A, B, C = a.name
    l = Segment(A, C)
    l_rep = diagram.database.inverse_eqline[l]
    for p in diagram.database.lines_points[l_rep]:
        if p in [A, C]:
            continue
        if diagram.database.is_cong(Segment(p, B), Segment(p, A)):
            f = Fact("cong", [Segment(p, A), Segment(p, C)],
                     "perp_and_eqline_and_cong_to_cong")
            f.add_parent(fact)
            f.add_parent(Fact("cong", [Segment(p, B), Segment(p, A)]))
            f.add_parent(Fact("eqline", [Segment(A, C), Segment(p, C)]))
            facts.append(f)
        elif diagram.database.is_cong(Segment(p, B), Segment(p, C)):
            f = Fact("cong", [Segment(p, A), Segment(p, C)],
                     "perp_and_eqline_and_cong_to_cong")
            f.add_parent(fact)
            f.add_parent(Fact("cong", [Segment(p, B), Segment(p, C)]))
            f.add_parent(Fact("eqline", [Segment(A, C), Segment(p, C)]))
            facts.append(f)
    return facts


def cong_and_eqline_and_perp_to_cong(diagram: 'Diagram',
                                     fact: Fact) -> List[Fact]:
    """From righttri and cong to cong."""
    facts = []
    s1, s2 = fact.objects
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    D = s12_joint
    A, B = s12_p_set
    if on_same_line(*[diagram.point_dict[p] for p in [A, B, D]]):
        return facts
    s = Segment(A, B)
    if s not in diagram.database.segments_perps:
        return facts
    for a in diagram.database.segments_perps[s]:
        C = a.name[0] if a.name[0] not in [A, B] else a.name[2]
        if a.name[1] == B and diagram.database.is_eqline(
                Segment(C, D), Segment(D, A)):
            f = Fact("cong", [Segment(C, D), Segment(D, A)],
                     "cong_and_eqline_and_perp_to_cong")
            f.add_parent(fact)
            f.add_parent(Fact("eqline", [Segment(C, D), Segment(D, A)]))
            f.add_parent(Fact("perp", [a]))
            facts.append(f)
        if a.name[1] == A and diagram.database.is_eqline(
                Segment(C, D), Segment(D, B)):
            f = Fact("cong", [Segment(C, D), Segment(D, B)],
                     "cong_and_eqline_and_perp_to_cong")
            f.add_parent(fact)
            f.add_parent(Fact("eqline", [Segment(C, D), Segment(D, B)]))
            f.add_parent(Fact("perp", [a]))
            facts.append(f)
    return facts


def eqline_and_cong_and_perp_to_cong(diagram: 'Diagram',
                                     fact: Fact) -> List[Fact]:
    """From righttri and cong to cong."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    D = s12_joint
    A, C = s12_p_set
    s = Segment(A, C)
    if s not in diagram.database.h_segments_perps:
        return facts
    for a in diagram.database.h_segments_perps[s]:
        B = a.name[1]
        if diagram.database.is_cong(Segment(D, B), Segment(D, C)):
            f = Fact("cong", [Segment(D, A), Segment(D, C)],
                     "eqline_and_cong_and_perp_to_cong")
            f.add_parent(fact)
            f.add_parent(Fact("cong", [Segment(D, B), Segment(D, C)]))
            f.add_parent(Fact("perp", [a]))
            facts.append(f)
        if diagram.database.is_cong(Segment(D, B), Segment(D, A)):
            f = Fact("cong", [Segment(D, A), Segment(D, C)],
                     "eqline_and_cong_and_perp_to_cong")
            f.add_parent(fact)
            f.add_parent(Fact("cong", [Segment(D, B), Segment(D, A)]))
            f.add_parent(Fact("perp", [a]))
            facts.append(f)
    return facts
