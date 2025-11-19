r"""Concyclic-related rules."""

from itertools import combinations, product
from typing import TYPE_CHECKING, List, Tuple

from tonggeometry.constructor.primitives import on_same_line, same_dir
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import (Angle, Circle, Ratio,
                                                      Segment)
from tonggeometry.inference_engine.util import four_joint
from tonggeometry.util import OrderedSet, isclose, unique

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def eqcircle_to_eqcircle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """From eqcircles get all equivalents."""
    facts = []
    c1, c2 = fact.objects
    if c1.center and c2.center:
        c1_key = diagram.database.inverse_eqcircle[c1]
        if len(diagram.database.eqcircle[c1_key]) > 2 and all(
                c.center is not None
                for c in diagram.database.eqcircle[c1_key]):
            p1 = list(c1.points)[0]
            p2 = list(c2.points)[0]
            c = diagram.database.circles_circles[c1_key]
            for p in c.points:
                if p not in [p1, p2]:
                    break
            c_new = Circle(None, [p, p1, p2])
            f = Fact("eqcircle", [c1, c_new], "eqcircle_to_eqcircle")
            f.add_parent(fact)
            f.add_parent(Fact("eqcircle", [c1, Circle(c1.center, [p])]))
            facts.append(f)
    elif c1.center:
        for p in c2.points:
            c_new = Circle(c1.center, [p])
            if c_new != c1:
                f = Fact("eqcircle", [c1, c_new], "eqcircle_to_eqcircle")
                f.add_parent(fact)
                facts.append(f)
        p_c1 = list(c1.points)[0]
        if p_c1 not in c2.points:
            for p1, p2 in combinations(c2.points, 2):
                c_new = Circle(None, [p1, p2, p_c1])
                f = Fact("eqcircle", [c1, c_new], "eqcircle_to_eqcircle")
                f.add_parent(fact)
                facts.append(f)
    elif c2.center:
        for p in c1.points:
            c_new = Circle(c2.center, [p])
            if c_new != c2:
                f = Fact("eqcircle", [c2, c_new], "eqcircle_to_eqcircle")
                f.add_parent(fact)
                facts.append(f)
        p_c2 = list(c2.points)[0]
        if p_c2 not in c1.points:
            for p1, p2 in combinations(c1.points, 2):
                c_new = Circle(None, [p1, p2, p_c2])
                f = Fact("eqcircle", [c2, c_new], "eqcircle_to_eqcircle")
                f.add_parent(fact)
                facts.append(f)
    else:
        all_points = OrderedSet()
        all_points.update(c1.points)
        all_points.update(c2.points)
        for triple in combinations(all_points, 3):
            c_new = Circle(None, list(triple))
            if c_new not in [c1, c2]:
                f = Fact("eqcircle", [c1, c_new], "eqcircle_to_eqcircle")
                f.add_parent(fact)
                facts.append(f)
    return facts


def eqcircle_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Concyclic implies equal angle."""
    facts = []
    c1, c2 = fact.objects
    all_points = OrderedSet()
    all_points.update(c1.points)
    all_points.update(c2.points)
    for A, B, C, D in combinations(all_points, 4):
        for a1, a2 in [[Angle(B, A, C), Angle(B, D, C)],
                       [Angle(A, B, D), Angle(A, C, D)],
                       [Angle(C, A, D), Angle(C, B, D)],
                       [Angle(A, C, B), Angle(A, D, B)],
                       [Angle(A, B, C), Angle(A, D, C)],
                       [Angle(B, A, D), Angle(B, C, D)]]:
            f = Fact("eqangle", [a1, a2], "eqcircle_to_eqangle")
            f.add_parent(fact)
            facts.append(f)
    return facts


def eqcircle_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Radii are congruent."""
    facts = []
    c1, c2 = fact.objects
    if not c1.center and not c2.center:
        return facts
    center = c1.center if c1.center else c2.center
    all_points = OrderedSet()
    all_points.update(c1.points)
    all_points.update(c2.points)
    all_points_list = list(all_points)
    p_anchor = all_points_list[0]
    for p in all_points_list[1:]:
        f = Fact("cong", [Segment(center, p_anchor),
                          Segment(center, p)], "eqcircle_to_cong")
        f.add_parent(fact)
        facts.append(f)
    return facts


def cong_to_eqcircle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Congruent radii imply concyclic."""
    facts = []
    s1, s2 = fact.objects
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    B = s12_joint
    A, C = s12_p_set
    c1 = Circle(B, [A])
    c2 = Circle(B, [C])
    f = Fact("eqcircle", [c1, c2], "cong_to_eqcircle")
    f.add_parent(fact)
    facts.append(f)
    return facts


def eqangle_to_eqcircle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Equal angles imply concyclic."""
    facts = []
    a1, a2 = fact.objects
    if not (a1.p1 == a2.p1 and a1.p3 == a2.p3 and a1.p2 != a2.p2):
        return facts
    if on_same_line(*[diagram.point_dict[p] for p in a1.name]):
        return facts
    A, B, C = a1.name
    D = a2.p2
    c1 = Circle(None, [A, B, C])
    c2 = Circle(None, [A, D, C])
    f = Fact("eqcircle", [c1, c2], "eqangle_to_eqcircle")
    f.add_parent(fact)
    facts.append(f)
    return facts


def eqline_and_eqcircle_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Perpendicular for angle on circumference corresponding to diameter."""
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
    c1 = Circle(B, [A])
    c2 = Circle(B, [C])
    if not diagram.database.is_eqcircle(c1, c2):
        return facts
    c_rep = diagram.database.inverse_eqcircle[c1]
    c_all = diagram.database.circles_circles[c_rep]
    for p in c_all.points:
        if p in [
                A, C
        ] or not diagram.database.is_eqcircle(c1, Circle(None, [A, C, p])):
            continue
        f = Fact("perp", [Angle(A, p, C)], "eqline_and_eqcircle_to_perp")
        f.add_parent(fact)
        f.add_parent(Fact("eqcircle", [c1, Circle(None, [A, C, p])]))
        facts.append(f)
    return facts


def eqcircle_and_eqline_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Perpendicular for angle on circumference corresponding to diameter."""
    facts = []
    c1, c2 = fact.objects
    if not c1.center and not c2.center or c1.center and c2.center:
        return facts
    center = c1.center if c1.center else c2.center
    all_points = OrderedSet()
    all_points.update(c1.points)
    all_points.update(c2.points)
    for p1, p2 in combinations(all_points, 2):
        s1 = Segment(center, p1)
        s2 = Segment(center, p2)
        if not diagram.database.is_eqline(s1, s2):
            continue
        for p in all_points:
            if p in [p1, p2]:
                continue
            f = Fact("perp", [Angle(p1, p, p2)], "eqcircle_and_eqline_to_perp")
            f.add_parent(fact)
            f.add_parent(Fact("eqline", [s1, s2]))
            facts.append(f)
    return facts


def eqangle_and_eqcircle_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Chord angle = Circum angle."""
    facts = []
    a1, a2 = fact.objects
    if a1.p3 == a2.p3 and a1.p1 == a2.p1:
        return facts
    if a1.p3 == a2.p3:
        a1 = Angle(*a1.name[::-1])
        a2 = Angle(*a2.name[::-1])
    if a1.p1 == a2.p1 and a1.p2 == a2.p3:
        a1, a2 = a2, a1
    if not (a1.p1 == a2.p1 and a1.p3 == a2.p2 and a1.p2 != a2.p3):
        return facts
    A, B, C = a1.name
    D = a2.p3
    c = Circle(None, [A, B, C])
    if c not in diagram.database.inverse_eqcircle:
        return facts
    c_rep = diagram.database.inverse_eqcircle[c]
    c_all = diagram.database.circles_circles[c_rep]
    O = c_all.center
    if not O or not diagram.database.is_eqcircle(Circle(O, [A]),
                                                 Circle(None, [A, B, C])):
        return facts
    f = Fact("perp", [Angle(O, C, D)], "eqangle_and_eqcircle_to_perp")
    f.add_parent(fact)
    f.add_parent(Fact("eqcircle", [Circle(O, [A]), Circle(None, [A, B, C])]))
    facts.append(f)
    return facts


def eqcircle_and_eqangle_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Chord angle = Circum angle."""
    facts = []
    c1, c2 = fact.objects
    if not c1.center and not c2.center or c1.center and c2.center:
        return facts
    all_points = OrderedSet()
    all_points.update(c1.points)
    all_points.update(c2.points)
    if len(all_points) != 3:
        return facts
    O = c1.center if c1.center else c2.center
    A, B, C = all_points
    for AA, BB, CC in [(A, B, C), (B, C, A), (C, A, B)]:
        s = Segment(AA, CC)
        if s not in diagram.database.segments_eqangles:
            continue
        for a in diagram.database.segments_eqangles[s]:
            if a.p2 == CC:
                D = a.p1 if a.p1 != AA else a.p3
                if not diagram.database.is_eqangle(Angle(AA, BB, CC),
                                                   Angle(AA, CC, D)):
                    continue
                f = Fact("perp", [Angle(O, CC, D)],
                         "eqcircle_and_eqangle_to_perp")
                f.add_parent(fact)
                f.add_parent(
                    Fact("eqangle", [Angle(AA, BB, CC),
                                     Angle(AA, CC, D)]))
                facts.append(f)
            elif a.p2 == AA:
                D = a.p1 if a.p1 != CC else a.p3
                if not diagram.database.is_eqangle(Angle(CC, BB, AA),
                                                   Angle(CC, AA, D)):
                    continue
                f = Fact("perp", [Angle(O, AA, D)],
                         "eqcircle_and_eqangle_to_perp")
                f.add_parent(fact)
                f.add_parent(
                    Fact("eqangle", [Angle(CC, BB, AA),
                                     Angle(CC, AA, D)]))
                facts.append(f)
    return facts


def eqcircle_and_perp_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Chord angle = Circum angle."""
    facts = []
    c1, c2 = fact.objects
    if not c1.center and not c2.center or c1.center and c2.center:
        return facts
    all_points = OrderedSet()
    all_points.update(c1.points)
    all_points.update(c2.points)
    if len(all_points) != 3:
        return facts
    O = c1.center if c1.center else c2.center
    A, B, C = all_points
    for p, pp, ppp in [(A, B, C), (B, C, A), (C, A, B)]:
        s = Segment(O, p)
        if s in diagram.database.segments_perps:
            D = None
            for a in diagram.database.segments_perps[s]:
                if a.p2 == p:
                    D = a.p1 if a.p1 != O else a.p3
                    break
            if D:
                f = Fact(
                    "eqangle",
                    [Angle(pp, ppp, p), Angle(pp, p, D)],
                    "eqcircle_and_perp_to_eqangle")
                f.add_parent(fact)
                f.add_parent(Fact("perp", [Angle(O, p, D)]))
                facts.append(f)
                f = Fact(
                    "eqangle",
                    [Angle(ppp, pp, p), Angle(ppp, p, D)],
                    "eqcircle_and_perp_to_eqangle")
                f.add_parent(fact)
                f.add_parent(Fact("perp", [Angle(O, p, D)]))
                facts.append(f)
    return facts


def perp_and_eqcircle_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Chord angle = Circum angle."""
    facts = []
    a = fact.objects[0]
    p = a.name[1]
    for O, D in [(a.p1, a.p3), (a.p3, a.p1)]:
        c = Circle(O, [p])
        if c not in diagram.database.inverse_eqcircle:
            continue
        c_rep = diagram.database.inverse_eqcircle[c]
        c_all = diagram.database.circles_circles[c_rep]
        for pp, ppp in combinations(c_all.points, 2):
            if p in [pp, ppp] or not diagram.database.is_eqcircle(
                    c, Circle(None, [p, pp, ppp])):
                continue
            f = Fact("eqangle",
                     [Angle(pp, p, D), Angle(pp, ppp, p)],
                     "perp_and_eqcircle_to_eqangle")
            f.add_parent(fact)
            f.add_parent(Fact("eqcircle", [c, Circle(None, [p, pp, ppp])]))
            facts.append(f)
            f = Fact("eqangle",
                     [Angle(ppp, p, D), Angle(ppp, pp, p)],
                     "perp_and_eqcircle_to_eqangle")
            f.add_parent(fact)
            f.add_parent(Fact("eqcircle", [c, Circle(None, [p, pp, ppp])]))
            facts.append(f)
    return facts


def eqcircle_and_perp_to_eqline(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Perp in a circle is against diameter."""
    facts = []
    c1, c2 = fact.objects
    if not c1.center and not c2.center or c1.center and c2.center:
        return facts
    all_points = OrderedSet()
    all_points.update(c1.points)
    all_points.update(c2.points)
    if len(all_points) != 3:
        return facts
    O = c1.center if c1.center else c2.center
    A, B, C = all_points
    for p, pp, ppp in [(A, B, C), (B, C, A), (C, A, B)]:
        if not diagram.database.is_perp(Angle(p, pp, ppp)):
            continue
        f = Fact("eqline", [Segment(O, p), Segment(O, ppp)],
                 "eqcircle_and_perp_to_eqline")
        f.add_parent(fact)
        f.add_parent(Fact("perp", [Angle(p, pp, ppp)]))
        facts.append(f)
        break
    return facts


def perp_and_eqcircle_to_eqline(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Perp in a circle is against diameter."""
    facts = []
    a = fact.objects[0]
    p, pp, ppp = a.name
    c = Circle(None, [p, pp, ppp])
    if c not in diagram.database.inverse_eqcircle:
        return facts
    c_rep = diagram.database.inverse_eqcircle[c]
    c_all = diagram.database.circles_circles[c_rep]
    O = c_all.center
    if not O or not diagram.database.is_eqcircle(c, Circle(O, [p])):
        return facts
    f = Fact("eqline", [Segment(O, p), Segment(O, ppp)],
             "perp_and_eqcircle_to_eqline")
    f.add_parent(fact)
    f.add_parent(Fact("eqcircle", [c, Circle(O, [p])]))
    facts.append(f)
    return facts


def eqangle_and_eqcircle_to_eqangle_cen(diagram: 'Diagram',
                                        fact: Fact) -> List[Fact]:
    """Central angle equal implies circum angle equal."""
    facts = []
    a1, a2 = fact.objects
    dir_val = same_dir(*[diagram.point_dict[p] for p in a1.name + a2.name])
    if dir_val != 1:
        return facts
    A1, O1, C1 = a1.name
    A2, O2, C2 = a2.name
    c1A = Circle(O1, [A1])
    c1C = Circle(O1, [C1])
    c2A = Circle(O2, [A2])
    c2C = Circle(O2, [C2])
    if not (diagram.database.is_eqcircle(c1A, c1C)
            and diagram.database.is_eqcircle(c2A, c2C)):
        return facts
    c1_rep = diagram.database.inverse_eqcircle[c1A]
    c1_all = diagram.database.circles_circles[c1_rep]
    c2_rep = diagram.database.inverse_eqcircle[c2A]
    c2_all = diagram.database.circles_circles[c2_rep]
    if len(c1_all.points) < 3 or len(c2_all.points) < 3:
        return facts
    for D1 in c1_all.points:
        if D1 in [A1, C1]:
            continue
        break
    for D2 in c2_all.points:
        if D2 in [A2, C2]:
            continue
        break
    if not (diagram.database.is_eqcircle(c1A, Circle(None, [A1, C1, D1]))
            and diagram.database.is_eqcircle(c2A, Circle(None, [A2, C2, D2]))):
        return facts
    f = Fact("eqangle",
             [Angle(A1, D1, C1), Angle(A2, D2, C2)],
             "eqangle_and_eqcircle_to_eqangle_cen")
    f.add_parent(fact)
    f.add_parent(Fact("eqcircle", [c1A, Circle(None, [A1, C1, D1])]))
    f.add_parent(Fact("eqcircle", [c2A, Circle(None, [A2, C2, D2])]))
    facts.append(f)
    return facts


def eqcircle_and_eqangle_to_eqangle_cen(diagram: 'Diagram',
                                        fact: Fact) -> List[Fact]:
    """Central angle equal implies circum angle equal."""
    facts = []
    c1, c2 = fact.objects
    if not c1.center and not c2.center or c1.center and c2.center:
        return facts
    all_points = OrderedSet()
    all_points.update(c1.points)
    all_points.update(c2.points)
    if len(all_points) != 3:
        return facts
    A1, B1, C1 = all_points
    O1 = c1.center if c1.center else c2.center
    c_rep = diagram.database.inverse_eqcircle[c1]
    c_all = diagram.database.circles_circles[c_rep]
    for p, pp, ppp in [(A1, B1, C1), (B1, C1, A1), (C1, A1, B1)]:
        a = Angle(p, O1, ppp)
        if a not in diagram.database.inverse_eqangle:
            continue
        eqclass_rep = diagram.database.inverse_eqangle[a]
        eqclass = diagram.database.eqangle[eqclass_rep]
        a = eqclass[a]
        is_perp = diagram.database.is_perp(a)
        for a_p in eqclass:
            if a_p == a:
                continue
            dir_val = same_dir(
                *[diagram.point_dict[p] for p in a.name + a_p.name])
            if dir_val != 1:
                if not is_perp:
                    continue
                a_p = Angle(*a_p.name[::-1])
            c1 = Circle(a_p.p2, [a_p.p1])
            c2 = Circle(a_p.p2, [a_p.p3])
            if not diagram.database.is_eqcircle(c1, c2):
                continue
            c_rep = diagram.database.inverse_eqcircle[c1]
            c_all = diagram.database.circles_circles[c_rep]
            if len(c_all.points) < 3:
                continue
            for B2 in c_all.points:
                if B2 in [a_p.p1, a_p.p3]:
                    continue
                break
            f = Fact("eqangle",
                     [Angle(a.p1, pp, a.p3),
                      Angle(a_p.p1, B2, a_p.p3)],
                     "eqcircle_and_eqangle_to_eqangle_cen")
            f.add_parent(fact)
            f.add_parent(
                Fact("eqcircle", [c1, Circle(None, [a_p.p1, B2, a_p.p3])]))
            f.add_parent(Fact("eqangle", [a, a_p]))
            facts.append(f)
    return facts


def eqangle_and_eqcircle_to_eqangle_cir(diagram: 'Diagram',
                                        fact: Fact) -> List[Fact]:
    """Central angle equal implies circum angle equal."""
    facts = []
    a1, a2 = fact.objects
    dir_val = same_dir(*[diagram.point_dict[p] for p in a1.name + a2.name])
    if dir_val != 1:
        return facts
    A1, B1, C1 = a1.name
    A2, B2, C2 = a2.name
    c1 = Circle(None, [A1, B1, C1])
    c2 = Circle(None, [A2, B2, C2])
    if not (c1 in diagram.database.inverse_eqcircle
            and c2 in diagram.database.inverse_eqcircle):
        return facts
    c1_rep = diagram.database.inverse_eqcircle[c1]
    c1_all = diagram.database.circles_circles[c1_rep]
    c2_rep = diagram.database.inverse_eqcircle[c2]
    c2_all = diagram.database.circles_circles[c2_rep]
    O1 = c1_all.center
    O2 = c2_all.center
    if not (O1 and O2):
        return facts
    if not (diagram.database.is_eqcircle(c1, Circle(O1, [A1]))
            and diagram.database.is_eqcircle(c2, Circle(O2, [A2]))):
        return facts
    f = Fact("eqangle",
             [Angle(A1, O1, C1), Angle(A2, O2, C2)],
             "eqangle_and_eqcircle_to_eqangle_cir")
    f.add_parent(fact)
    f.add_parent(Fact("eqcircle", [c1, Circle(O1, [A1])]))
    f.add_parent(Fact("eqcircle", [c2, Circle(O2, [A2])]))
    facts.append(f)
    return facts


def eqcircle_and_eqangle_to_eqangle_cir(diagram: 'Diagram',
                                        fact: Fact) -> List[Fact]:
    """Central angle equal implies circum angle equal."""
    facts = []
    c1, c2 = fact.objects
    if not c1.center and not c2.center or c1.center and c2.center:
        return facts
    all_points = OrderedSet()
    all_points.update(c1.points)
    all_points.update(c2.points)
    if len(all_points) != 3:
        return facts
    A1, B1, C1 = all_points
    O1 = c1.center if c1.center else c2.center
    c_rep = diagram.database.inverse_eqcircle[c1]
    c_all = diagram.database.circles_circles[c_rep]
    for p, pp, ppp in [(A1, B1, C1), (B1, C1, A1), (C1, A1, B1)]:
        a = Angle(p, pp, ppp)
        if a not in diagram.database.inverse_eqangle:
            continue
        eqclass_rep = diagram.database.inverse_eqangle[a]
        eqclass = diagram.database.eqangle[eqclass_rep]
        a = eqclass[a]
        for a_p in eqclass:
            if a_p == a:
                continue
            dir_val = same_dir(
                *[diagram.point_dict[p] for p in a.name + a_p.name])
            if dir_val != 1:
                continue
            c = Circle(None, [*a_p.name])
            if c not in diagram.database.inverse_eqcircle:
                continue
            c_rep = diagram.database.inverse_eqcircle[c]
            c_all = diagram.database.circles_circles[c_rep]
            if not c_all.center:
                continue
            O2 = c_all.center
            f = Fact("eqangle",
                     [Angle(a.p1, O1, a.p3),
                      Angle(a_p.p1, O2, a_p.p3)],
                     "eqcircle_and_eqangle_to_eqangle_cir")
            f.add_parent(fact)
            f.add_parent(Fact("eqcircle", [c, Circle(O2, [a_p.p1])]))
            f.add_parent(Fact("eqangle", [a, a_p]))
            facts.append(f)
    return facts


def eqangle_and_eqcircle_to_eqangle_half(diagram: 'Diagram',
                                         fact: Fact) -> List[Fact]:
    """Half of central angle equal circum angle."""
    facts = []
    a1, a2 = fact.objects
    dir_val = same_dir(*[diagram.point_dict[p] for p in a1.name + a2.name])
    if dir_val != 1:
        return facts
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if s1 == s4:
        s1, s2, s3, s4 = s2, s1, s4, s3
    if not (s2 == s3 and len(set([s1, s2, s3, s4])) == 3):
        return facts
    if not a1.p2 == a2.p2:
        return facts
    A = s1.p2
    O = s1.p1
    C = s4.p2
    D = s2.p2
    cA = Circle(O, [A])
    cC = Circle(O, [C])
    if not diagram.database.is_eqcircle(cA, cC):
        return facts
    c_rep = diagram.database.inverse_eqcircle[cA]
    c_all = diagram.database.circles_circles[c_rep]
    if len(c_all.points) < 3:
        return facts
    for B in c_all.points:
        if B in [A, C]:
            continue
        break
    if not diagram.database.is_eqcircle(cA, Circle(None, [A, B, C])):
        return facts
    f = Fact("eqangle", [Angle(A, O, D), Angle(A, B, C)],
             "eqangle_and_eqcircle_to_eqangle_half")
    f.add_parent(fact)
    f.add_parent(Fact("eqcircle", [cA, Circle(None, [A, B, C])]))
    facts.append(f)
    return facts


def eqcircle_and_eqangle_to_eqangle_half(diagram: 'Diagram',
                                         fact: Fact) -> List[Fact]:
    """Half of central angle equal circum angle."""
    facts = []
    c1, c2 = fact.objects
    if not c1.center and not c2.center or c1.center and c2.center:
        return facts
    all_points = OrderedSet()
    all_points.update(c1.points)
    all_points.update(c2.points)
    if len(all_points) != 3:
        return facts
    O = c1.center if c1.center else c2.center
    A, B, C = all_points
    for p, pp, ppp in [(A, B, C), (B, C, A), (C, A, B)]:
        s = Segment(p, O)
        if s not in diagram.database.segments_eqangles:
            continue
        for a in diagram.database.segments_eqangles[s]:
            if a.name[1] != O or a == Angle(p, O, ppp):
                continue
            ss = a.s1 if a.s1 != s else a.s2
            D = ss.p2
            a1 = Angle(p, O, D)
            a2 = Angle(D, O, ppp)
            dir_val = same_dir(
                *[diagram.point_dict[p] for p in a1.name + a2.name])
            if dir_val != 1:
                continue
            if not diagram.database.is_eqangle(a1, a2):
                continue
            f = Fact("eqangle",
                     [Angle(p, pp, ppp), Angle(p, O, D)],
                     "eqcircle_and_eqangle_to_eqangle_half")
            f.add_parent(fact)
            f.add_parent(Fact("eqangle", [a1, a2]))
            facts.append(f)
            break
    return facts


# Definition of radax


def add_ax(diagram: 'Diagram', c: Circle, ax: Segment, p: str, center: str,
           in_between: bool) -> bool:
    """Log ax. c must be rep."""
    if c not in diagram.database.axes:
        diagram.database.axes[c] = {}
    ax_p = (ax, p)
    if ax_p in diagram.database.axes[c]:
        return False
    diagram.database.axes[c][ax_p] = None
    if p not in diagram.database.points_axes:
        diagram.database.points_axes[p] = {}
    if ax not in diagram.database.points_axes[p]:
        diagram.database.points_axes[p][ax] = (in_between, [])
    diagram.database.points_axes[p][ax][1].append(center)
    return True


def add_radax(diagram: 'Diagram', c1: Circle, c2: Circle, p: str,
              val: Tuple) -> bool:
    """Log radax. c1, c2 must be rep."""
    parents, c1_orig, c2_orig = val
    if c1 not in diagram.database.radaxes:
        diagram.database.radaxes[c1] = {}
    if c2 not in diagram.database.radaxes[c1]:
        diagram.database.radaxes[c1][c2] = OrderedSet()
    if c2 not in diagram.database.radaxes:
        diagram.database.radaxes[c2] = {}
    if c1 not in diagram.database.radaxes[c2]:
        diagram.database.radaxes[c2][c1] = OrderedSet()
    if p in diagram.database.radaxes[c1][c2]:
        return False
    diagram.database.radaxes[c1][c2][p] = (parents, c1_orig, c2_orig)
    diagram.database.radaxes[c2][c1][p] = (parents, c2_orig, c1_orig)
    if p not in diagram.database.points_radaxes:
        diagram.database.points_radaxes[p] = OrderedSet()
    if c1 not in diagram.database.points_radaxes[p]:
        diagram.database.points_radaxes[p][c1] = OrderedSet()
    if c2 not in diagram.database.points_radaxes[p]:
        diagram.database.points_radaxes[p][c2] = OrderedSet()
    diagram.database.points_radaxes[p][c1][c2] = None
    diagram.database.points_radaxes[p][c2][c1] = None
    return True


def cong_and_ax_to_radax(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """cong perp axes imply radax."""
    facts = []
    s1, s2 = fact.objects
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    p = s12_joint
    p_ax, p_axx = s12_p_set
    ax, axx = s1, s2
    if not (p in diagram.database.points_axes
            and ax in diagram.database.points_axes[p]
            and axx in diagram.database.points_axes[p]):
        return facts
    for center1, center2 in product(diagram.database.points_axes[p][ax][1],
                                    diagram.database.points_axes[p][axx][1]):
        c_r1 = (diagram.point_dict[center1] - diagram.point_dict[p_ax]).norm()
        c_r2 = (diagram.point_dict[center2] - diagram.point_dict[p_axx]).norm()
        if center1 == center2 and isclose(c_r1, c_r2):
            continue
        c1 = Circle(center1, [p_ax])
        c1_rep = diagram.database.inverse_eqcircle[c1]
        c2 = Circle(center2, [p_axx])
        c2_rep = diagram.database.inverse_eqcircle[c2]
        a1 = Angle(center1, p_ax, p)
        a2 = Angle(center2, p_axx, p)
        parents = [fact]
        parents.append(Fact("perp", [a1]))
        parents.append(Fact("perp", [a2]))
        status = add_radax(diagram, c1_rep, c2_rep, p, (parents, c1, c2))
        if status:
            facts += radax(diagram, c1_rep, c2_rep, p)
    return facts


def ax_and_cong_to_radax(diagram: 'Diagram', ax: Segment,
                         p: str) -> List[Fact]:
    """cong perp axes imply radax."""
    facts = []
    if ax not in diagram.database.inverse_cong:
        return facts
    ax_rep = diagram.database.inverse_cong[ax]
    ax_eqclass = diagram.database.cong[ax_rep]
    p_ax = ax.p2
    for axx in ax_eqclass:
        if axx == ax or p not in [
                axx.p1, axx.p2
        ] or axx not in diagram.database.points_axes[p]:
            continue
        p_axx = axx.p1 if axx.p1 != p else axx.p2
        for center1, center2 in product(
                diagram.database.points_axes[p][ax][1],
                diagram.database.points_axes[p][axx][1]):
            c_r1 = (diagram.point_dict[center1] -
                    diagram.point_dict[p_ax]).norm()
            c_r2 = (diagram.point_dict[center2] -
                    diagram.point_dict[p_axx]).norm()
            if center1 == center2 and isclose(c_r1, c_r2):
                continue
            c1 = Circle(center1, [p_ax])
            c1_rep = diagram.database.inverse_eqcircle[c1]
            c2 = Circle(center2, [p_axx])
            c2_rep = diagram.database.inverse_eqcircle[c2]
            a1 = Angle(center1, p_ax, p)
            a2 = Angle(center2, p_axx, p)
            parents = [Fact("cong", [ax, axx])]
            parents.append(Fact("perp", [a1]))
            parents.append(Fact("perp", [a2]))
            status = add_radax(diagram, c1_rep, c2_rep, p, (parents, c1, c2))
            if status:
                facts += radax(diagram, c1_rep, c2_rep, p)
    return facts


def perp_and_eqcircle_to_ax(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Log a possible radical axis point."""
    facts = []
    a = fact.objects[0]
    A, H, B = a.name
    for p, center in [(A, B), (B, A)]:
        c = Circle(center, [H])
        ax = Segment(p, H)
        if c not in diagram.database.inverse_eqcircle:
            continue
        c_rep = diagram.database.inverse_eqcircle[c]
        status = add_ax(diagram, c_rep, ax, p, center, False)
        if status:
            facts += ax_and_cong_to_radax(diagram, ax, p)
    return facts


def eqcircle_and_perp_to_ax(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Log a possible radical axis point."""
    facts = []
    c1, c2 = fact.objects
    if not (c1.center and c2.center):
        return facts
    A = list(c1.points)[0]
    B = list(c2.points)[0]
    center = c1.center
    c_rep = diagram.database.inverse_eqcircle[c1]
    for H in [A, B]:
        s = Segment(center, H)
        if s not in diagram.database.segments_perps:
            continue
        for a in diagram.database.segments_perps[s]:
            if a.name[1] != H:
                continue
            p = a.name[0] if a.name[0] != center else a.name[2]
            ax = Segment(p, H)
            status = add_ax(diagram, c_rep, ax, p, center, False)
            if status:
                facts += ax_and_cong_to_radax(diagram, ax, p)
    return facts


def eqratio_and_ax_to_radax(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Same power axes imply radax."""
    facts = []
    r1, r2 = fact.objects
    if ((diagram.database.is_cong(r1.s1, r2.s2)
         and diagram.database.is_cong(r1.s2, r2.s1))
            or diagram.database.is_cong(r1.s1, r1.s2)
            or diagram.database.is_cong(r2.s1, r2.s2)):
        return facts
    eqclass_rep = diagram.database.inverse_eqratio[r1]
    eqclass = diagram.database.eqratio[eqclass_rep]
    dir_r1 = eqclass[r1].s1 != r1.s1
    dir_r2 = eqclass[r2].s1 != r2.s1
    if dir_r1 != dir_r2:
        return facts
    s1, s2 = r1.s1, r1.s2
    s3, s4 = r2.s1, r2.s2
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    s34_joint, s34_p_set = four_joint(s3.p1, s3.p2, s4.p1, s4.p2)
    if not (s12_joint and s34_joint and s12_joint == s34_joint):
        return facts
    s12_p_set.pop(s12_joint)
    s34_p_set.pop(s34_joint)
    p = s12_joint
    A, C = s12_p_set
    D, B = s34_p_set
    if A != B:
        ax = Segment(A, B)
    else:
        ax = Segment(p, A)
    if C != D:
        axx = Segment(C, D)
    else:
        axx = Segment(p, C)
    if A == B:
        ax, axx = axx, ax
    if not (p in diagram.database.points_axes
            and ax in diagram.database.points_axes[p]
            and axx in diagram.database.points_axes[p]
            and diagram.database.points_axes[p][ax][0]
            == diagram.database.points_axes[p][axx][0]):
        return facts
    # s_pA = Segment(p, ax.p1)
    # s_pB = Segment(p, ax.p2)
    p_in = p == axx.p1
    # r1 = Ratio(s_pA, axx)
    # r2 = Ratio(axx, s_pB)
    # r3 = Ratio(s_pA, Segment(p, axx.p1))
    # r4 = Ratio(Segment(p, axx.p2), s_pB)
    # if not (p_in and diagram.database.is_eqratio(r1, r2)
    #         or not p_in and diagram.database.is_eqratio(r3, r4)):
    #     return facts
    p_ax = ax.p2
    p_axx = axx.p2
    for center1, center2 in product(diagram.database.points_axes[p][ax][1],
                                    diagram.database.points_axes[p][axx][1]):
        c_r1 = (diagram.point_dict[center1] - diagram.point_dict[p_ax]).norm()
        c_r2 = (diagram.point_dict[center2] - diagram.point_dict[p_axx]).norm()
        if center1 == center2 and isclose(c_r1, c_r2):
            continue
        c1 = Circle(center1, [p_ax])
        c1_rep = diagram.database.inverse_eqcircle[c1]
        c2 = Circle(center2, [p_axx])
        c2_rep = diagram.database.inverse_eqcircle[c2]
        parents = [
            Fact("eqline",
                 [Segment(p, ax.p1), Segment(p, ax.p2)]),
            Fact("eqcircle",
                 [Circle(center1, [ax.p1]),
                  Circle(center1, [ax.p2])])
        ]
        if p_in:
            # if diagram.database.is_cong(
            #         r1.s1, r1.s2) and diagram.database.is_cong(r2.s1, r2.s2):
            #     parents.append(Fact("cong", [r1.s1, r1.s2]))
            #     parents.append(Fact("cong", [r2.s1, r2.s2]))
            # else:
            #     parents.append(Fact("eqratio", [r1, r2]))
            parents.append(fact)
            parents.append(Fact("perp", [Angle(center2, p_axx, p)]))
        else:
            # if diagram.database.is_cong(
            #         r3.s1, r3.s2) and diagram.database.is_cong(r4.s1, r4.s2):
            #     parents.append(Fact("cong", [r3.s1, r3.s2]))
            #     parents.append(Fact("cong", [r4.s1, r4.s2]))
            # else:
            #     parents.append(Fact("eqratio", [r3, r4]))
            parents.append(fact)
            parents.append(
                Fact("eqline", [Segment(p, axx.p1),
                                Segment(p, axx.p2)]))
            parents.append(
                Fact("eqcircle",
                     [Circle(center2, [axx.p1]),
                      Circle(center2, [axx.p2])]))
        status = add_radax(diagram, c1_rep, c2_rep, p, (parents, c1, c2))
        if status:
            facts += radax(diagram, c1_rep, c2_rep, p)
    return facts


def cong_and_cong_and_ax_to_radax(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """Same power axes imply radax."""
    facts = []
    s1, s2 = fact.objects
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    p = s12_joint
    A, C = s12_p_set
    if not p in diagram.database.points_axes:
        return facts
    if on_same_line(*[diagram.point_dict[pt] for pt in [p, A, C]]):
        return facts
    partition = {True: {}, False: {}}
    for ax in diagram.database.points_axes[p]:
        if p in [ax.p1, ax.p2]:
            continue
        if A in [ax.p1, ax.p2]:
            B = ax.p1 if ax.p1 != A else ax.p2
            s_pB = Segment(p, B)
            if s_pB not in diagram.database.inverse_cong:
                continue
            rep = diagram.database.inverse_cong[s_pB]
            in_between = diagram.database.points_axes[p][ax][0]
            if rep not in partition[in_between]:
                partition[in_between][rep] = [[ax], []]
            else:
                partition[in_between][rep][0].append(ax)
    for ax in diagram.database.points_axes[p]:
        if p in [ax.p1, ax.p2]:
            continue
        if C in [ax.p1, ax.p2]:
            D = ax.p1 if ax.p1 != C else ax.p2
            s_pD = Segment(p, D)
            if s_pD not in diagram.database.inverse_cong:
                continue
            rep = diagram.database.inverse_cong[s_pD]
            in_between = diagram.database.points_axes[p][ax][0]
            if rep in partition[in_between]:
                partition[in_between][rep][1].append(ax)
    for in_between_group in partition.values():
        for rep_group in in_between_group.values():
            for ax, axx in product(rep_group[0], rep_group[1]):
                for center1, center2 in product(
                        diagram.database.points_axes[p][ax][1],
                        diagram.database.points_axes[p][axx][1]):
                    c_r1 = (diagram.point_dict[center1] -
                            diagram.point_dict[A]).norm()
                    c_r2 = (diagram.point_dict[center2] -
                            diagram.point_dict[C]).norm()
                    if center1 == center2 and isclose(c_r1, c_r2):
                        continue
                    c1 = Circle(center1, [A])
                    c1_rep = diagram.database.inverse_eqcircle[c1]
                    c2 = Circle(center2, [C])
                    c2_rep = diagram.database.inverse_eqcircle[c2]
                    B = ax.p1 if ax.p1 != A else ax.p2
                    D = axx.p1 if axx.p1 != C else axx.p2
                    parents = [
                        Fact("eqline", [Segment(p, ax.p1),
                                        Segment(p, ax.p2)]),
                        Fact("eqcircle", [
                            Circle(center1, [ax.p1]),
                            Circle(center1, [ax.p2])
                        ])
                    ]
                    parents.append(fact)
                    parents.append(Fact(
                        "cong", [Segment(p, B), Segment(p, D)]))
                    parents.append(
                        Fact("eqline",
                             [Segment(p, axx.p1),
                              Segment(p, axx.p2)]))
                    parents.append(
                        Fact("eqcircle", [
                            Circle(center2, [axx.p1]),
                            Circle(center2, [axx.p2])
                        ]))
                    status = add_radax(diagram, c1_rep, c2_rep, p,
                                       (parents, c1, c2))
                    if status:
                        facts += radax(diagram, c1_rep, c2_rep, p)
    return facts


def ax_and_eqratio_to_radax(diagram: 'Diagram', ax: Segment,
                            p: str) -> List[Fact]:
    """Same power axes imply radax. Handles both cong and cong and eqraito."""
    facts = []
    A, B = ax.p1, ax.p2
    s_pA = Segment(p, A)
    s_pB = Segment(p, B)
    if not (s_pA in diagram.database.segments_eqratios
            and s_pB in diagram.database.segments_eqratios):
        return facts
    for axx in diagram.database.points_axes[p]:
        if axx == ax or diagram.database.points_axes[p][axx][
                0] != diagram.database.points_axes[p][ax][0]:
            continue
        p_in = p == axx.p1
        r1 = Ratio(s_pA, axx)
        r2 = Ratio(axx, s_pB)
        r3 = Ratio(s_pA, Segment(p, axx.p1))
        r4 = Ratio(Segment(p, axx.p2), s_pB)
        if not (p_in and diagram.database.is_eqratio(r1, r2)
                or not p_in and diagram.database.is_eqratio(r3, r4)):
            continue
        p_ax = B
        p_axx = axx.p2
        for center1, center2 in product(
                diagram.database.points_axes[p][ax][1],
                diagram.database.points_axes[p][axx][1]):
            c_r1 = (diagram.point_dict[center1] -
                    diagram.point_dict[p_ax]).norm()
            c_r2 = (diagram.point_dict[center2] -
                    diagram.point_dict[p_axx]).norm()
            if center1 == center2 and isclose(c_r1, c_r2):
                continue
            c1 = Circle(center1, [p_ax])
            c1_rep = diagram.database.inverse_eqcircle[c1]
            c2 = Circle(center2, [p_axx])
            c2_rep = diagram.database.inverse_eqcircle[c2]
            parents = [
                Fact("eqline",
                     [Segment(p, ax.p1), Segment(p, ax.p2)]),
                Fact("eqcircle",
                     [Circle(center1, [ax.p1]),
                      Circle(center1, [ax.p2])])
            ]
            if p_in:
                parents.append(Fact("eqratio", [r1, r2]))
                parents.append(Fact("perp", [Angle(center2, p_axx, p)]))
            else:
                if diagram.database.is_cong(
                        r3.s1, r3.s2) and diagram.database.is_cong(
                            r4.s1, r4.s2):
                    parents.append(Fact("cong", [r3.s1, r3.s2]))
                    parents.append(Fact("cong", [r4.s1, r4.s2]))
                else:
                    parents.append(Fact("eqratio", [r3, r4]))
                parents.append(
                    Fact("eqline", [Segment(p, axx.p1),
                                    Segment(p, axx.p2)]))
                parents.append(
                    Fact(
                        "eqcircle",
                        [Circle(center2, [axx.p1]),
                         Circle(center2, [axx.p2])]))
            status = add_radax(diagram, c1_rep, c2_rep, p, (parents, c1, c2))
            if status:
                facts += radax(diagram, c1_rep, c2_rep, p)
    return facts


def eqline_and_eqcircle_to_ax(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Log a possible radical axis point."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    p = s12_joint
    A, B = s12_p_set
    if not (A in diagram.database.points_circles
            and B in diagram.database.points_circles):
        return facts
    all_c_AB = [
        c for c in diagram.database.points_circles[A]
        if diagram.database.circles_circles[c].center
        and c in diagram.database.points_circles[B]
    ]
    ax = Segment(A, B)
    p_p, p_A, p_B = [diagram.point_dict[pt] for pt in [p, A, B]]
    in_between = p_A < p_p < p_B or p_B < p_p < p_A
    for c_rep in all_c_AB:
        center = diagram.database.circles_circles[c_rep].center
        c_A = Circle(center, [A])
        c_B = Circle(center, [B])
        if (c_A not in diagram.database.inverse_eqcircle
                or c_B not in diagram.database.inverse_eqcircle
                or not diagram.database.is_eqcircle(c_A, c_B)):
            continue
        status = add_ax(diagram, c_rep, ax, p, center, in_between)
        if status:
            facts += ax_and_eqratio_to_radax(diagram, ax, p)
    return facts


def eqcircle_and_eqline_to_ax(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Log a possible radical axis point."""
    facts = []
    c1, c2 = fact.objects
    if not (c1.center and c2.center):
        return facts
    A = list(c1.points)[0]
    B = list(c2.points)[0]
    center = c1.center
    c_rep = diagram.database.inverse_eqcircle[c1]
    ax = Segment(A, B)
    l = diagram.database.inverse_eqline[ax]
    if len(diagram.database.lines_points[l]) < 3:
        return facts
    p_A, p_B = [diagram.point_dict[pt] for pt in [A, B]]
    for p in diagram.database.lines_points[l]:
        if p in [A, B]:
            continue
        p_p = diagram.point_dict[p]
        in_between = p_A < p_p < p_B or p_B < p_p < p_A
        status = add_ax(diagram, c_rep, ax, p, center, in_between)
        if status:
            facts += ax_and_eqratio_to_radax(diagram, ax, p)
    return facts


def eqcircle_and_eqcircle_to_radax(diagram: 'Diagram',
                                   fact: Fact) -> List[Fact]:
    """Trivial radax."""
    facts = []
    c1, c2 = fact.objects
    if not (c1.center and c2.center):
        return facts
    A = list(c1.points)[0]
    B = list(c2.points)[0]
    center = c1.center
    c_rep = diagram.database.inverse_eqcircle[c1]
    for p, c in [(A, c1), (B, c2)]:
        for cc_rep in diagram.database.points_circles[p]:
            if cc_rep == c_rep:
                continue
            cc_center = diagram.database.circles_circles[cc_rep].center
            if not cc_center or cc_center == center:
                continue
            cc = Circle(cc_center, [p])
            parents = []
            if not (diagram.database.is_eqcircle(c, c_rep)
                    and diagram.database.is_eqcircle(cc, cc_rep)):
                continue
            if p not in c_rep.points:
                parents.append(Fact("eqcircle", [c, c_rep]))
            if p not in cc_rep.points:
                parents.append(Fact("eqcircle", [cc, cc_rep]))
            status = add_radax(diagram, c_rep, cc_rep, p, (parents, c, cc))
            if status:
                facts += radax(diagram, c_rep, cc_rep, p)
    return facts


def eqline_and_radax_to_radax(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Radax from lines on radax."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    p = s12_joint
    A, B = s12_p_set
    if not (A in diagram.database.points_radaxes
            and B in diagram.database.points_radaxes):
        return facts
    circles = []
    for c in diagram.database.points_radaxes[A]:
        if c in diagram.database.points_radaxes[B]:
            circles.append(c)
    for c1, c2 in combinations(circles, 2):
        if c2 not in diagram.database.points_radaxes[A][
                c1] or c2 not in diagram.database.points_radaxes[B][c1]:
            continue
        parents = [fact]
        A_parents, A_c1, A_c2 = diagram.database.radaxes[c1][c2][A]
        B_parents, B_c1, B_c2 = diagram.database.radaxes[c1][c2][B]
        parents += unique(A_parents + B_parents)
        if A_c1 != B_c1:
            parents.append(Fact("eqcircle", [A_c1, B_c1]))
        if A_c2 != B_c2:
            parents.append(Fact("eqcircle", [A_c2, B_c2]))
        status = add_radax(diagram, c1, c2, p, (parents, A_c1, A_c2))
        if status:
            facts += radax(diagram, c1, c2, p)
    return facts


def eqline_and_radax_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """on center line implies perp."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    p = s12_joint
    A, B = s12_p_set
    if not p in diagram.database.points_radaxes:
        return facts
    cAs = []
    cBs = []
    for c in diagram.database.points_radaxes[p]:
        center = diagram.database.circles_circles[c].center
        if center == A:
            cAs.append(c)
        elif center == B:
            cBs.append(c)
    for cA, cB in product(cAs, cBs):
        if cB not in diagram.database.points_radaxes[p][cA] or len(
                diagram.database.radaxes[cA][cB]) < 2:
            continue
        for q in diagram.database.radaxes[cA][cB]:
            if q != p:
                break
        parents = [fact]
        p_parents, p_c1, p_c2 = diagram.database.radaxes[cA][cB][p]
        q_parents, q_c1, q_c2 = diagram.database.radaxes[cA][cB][q]
        parents += p_parents + q_parents
        if p_c1 != q_c1:
            parents.append(Fact("eqcircle", [p_c1, q_c1]))
        if p_c2 != q_c2:
            parents.append(Fact("eqcircle", [p_c2, q_c2]))
        f = Fact("perp", [Angle(A, p, q)])
        f.add_parents(parents)
        facts.append(f)
    return facts


def eqangle_and_radax_to_radax(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Radax from two perps."""
    facts = []
    a1, a2 = fact.objects
    if a1 == a2:
        return facts
    if a1.name[1] != a2.name[1]:
        return facts
    if not (diagram.database.is_perp(a1) or diagram.database.is_perp(a2)):
        return facts
    p_joint, p_set = four_joint(a1.name[0], a1.name[2], a2.name[0], a2.name[2])
    if not p_joint:
        return facts
    p_set.pop(p_joint)
    H = a1.name[1]
    A = p_joint
    O1, O2 = p_set
    for p_radax, p_new in [(H, A), (A, H)]:
        if p_radax not in diagram.database.points_radaxes:
            continue
        c_O1 = []
        c_O2 = []
        for c in diagram.database.points_radaxes[p_radax]:
            center = diagram.database.circles_circles[c].center
            if center == O1:
                c_O1.append(c)
            elif center == O2:
                c_O2.append(c)
        for c1, c2 in product(c_O1, c_O2):
            if c2 not in diagram.database.points_radaxes[p_radax][c1]:
                continue
            parents = []
            if diagram.database.is_perp(a1):
                parents.append(Fact("perp", [a1]))
            if diagram.database.is_perp(a2):
                parents.append(Fact("perp", [a2]))
            if len(parents) != 2:
                parents.append(fact)
            r_parents, r_c1, r_c2 = diagram.database.radaxes[c1][c2][p_radax]
            parents += r_parents
            status = add_radax(diagram, c1, c2, p_new, (parents, r_c1, r_c2))
            if status:
                facts += radax(diagram, c1, c2, p_new)
    return facts


def eqangle_and_radax_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """perp from two radaxes and circle centers."""
    facts = []
    a1, a2 = fact.objects
    if a1 == a2:
        return facts
    if a1.name[1] != a2.name[1]:
        return facts
    if not (diagram.database.is_perp(a1) or diagram.database.is_perp(a2)):
        return facts
    p_joint, p_set = four_joint(a1.name[0], a1.name[2], a2.name[0], a2.name[2])
    if not p_joint:
        return facts
    p_set.pop(p_joint)
    H = a1.name[1]
    center = p_joint
    A, B = p_set
    if not (A in diagram.database.points_radaxes
            and B in diagram.database.points_radaxes):
        return facts
    cs = []
    ccs = []
    for c_rep in diagram.database.points_radaxes[A]:
        if c_rep not in diagram.database.points_radaxes[B]:
            continue
        c_all = diagram.database.circles_circles[c_rep]
        c_center = c_all.center
        if not c_center or c_center == H:
            continue
        if c_center == center:
            cs.append(c_rep)
        else:
            ccs.append(c_rep)
    for c_rep, cc_rep in product(cs, ccs):
        if not (cc_rep in diagram.database.points_radaxes[A][c_rep]
                and cc_rep in diagram.database.points_radaxes[B][c_rep]):
            continue
        cc_center = diagram.database.circles_circles[cc_rep].center
        parents = []
        if diagram.database.is_perp(a1):
            parents.append(Fact("perp", [a1]))
        if diagram.database.is_perp(a2):
            parents.append(Fact("perp", [a2]))
        if len(parents) != 2:
            parents.append(fact)
        A_parents, A_c1, A_c2 = diagram.database.radaxes[c_rep][cc_rep][A]
        B_parents, B_c1, B_c2 = diagram.database.radaxes[c_rep][cc_rep][B]
        parents += A_parents + B_parents
        if A_c1 != B_c1:
            parents.append(Fact("eqcircle", [A_c1, B_c1]))
        if A_c2 != B_c2:
            parents.append(Fact("eqcircle", [A_c2, B_c2]))
        f = Fact("perp", [Angle(cc_center, H, A)], "eqangle_and_radax_to_perp")
        f.add_parents(parents)
        facts.append(f)
        f = Fact("perp", [Angle(cc_center, H, B)], "eqangle_and_radax_to_perp")
        f.add_parents(parents)
        facts.append(f)
    return facts


# Property of radax


def radax(diagram: 'Diagram', c1: Circle, c2: Circle, p: str) -> List[Fact]:
    """A new point on the radical axis. c1, c2 shall be rep circles.
    0. same power of point
    1. on one circle implies on another circle
    2. on center line implies perp
    3. perp implies on center line
    4. another point on center line implies perp
    5. another point perp implies on center line
    6. eqline for points on radax
    7. eqline points are on radax
    8. eqangle and radax implies radax
    9. Monge Theorem (radical center)
    """
    facts = []
    c1_center = diagram.database.circles_circles[c1].center
    c2_center = diagram.database.circles_circles[c2].center
    c1_p = Circle(c1_center, [p])
    c2_p = Circle(c2_center, [p])
    p_parents, p_c1, p_c2 = diagram.database.radaxes[c1][c2][p]
    # same power of point
    if c1 in diagram.database.axes and c2 in diagram.database.axes:
        partition = {True: [[], []], False: [[], []]}
        for ax_p in diagram.database.axes[c1]:
            if ax_p[1] != p:
                continue
            ax = ax_p[0]
            ax_p_in_between = diagram.database.points_axes[p][ax][0]
            partition[ax_p_in_between][0].append(ax)
        for ax_p in diagram.database.axes[c2]:
            if ax_p[1] != p:
                continue
            ax = ax_p[0]
            ax_p_in_between = diagram.database.points_axes[p][ax][0]
            partition[ax_p_in_between][1].append(ax)
        for axs in partition.values():
            for c1_ax, c2_ax in product(axs[0], axs[1]):
                if c1_ax.p1 == p and c2_ax.p1 == p:
                    f = Fact("cong", [c1_ax, c2_ax], "radax")
                    f.add_parents(p_parents)
                    f.add_parent(
                        Fact("perp", [Angle(c1_ax.p1, c1_ax.p2, c1_center)]))
                    f.add_parent(
                        Fact("perp", [Angle(c2_ax.p1, c2_ax.p2, c2_center)]))
                    facts.append(f)
                elif c1_ax.p1 == p and c2_ax.p1 != p:
                    f = Fact("eqratio", [
                        Ratio(c1_ax, Segment(p, c2_ax.p1)),
                        Ratio(Segment(p, c2_ax.p2), c1_ax)
                    ], "radax")
                    f.add_parents(p_parents)
                    f.add_parent(
                        Fact("perp", [Angle(c1_ax.p1, c1_ax.p2, c1_center)]))
                    f.add_parent(
                        Fact("cong", [
                            Segment(c2_center, c2_ax.p1),
                            Segment(c2_center, c2_ax.p2)
                        ]))
                    f.add_parent(
                        Fact("eqline",
                             [Segment(p, c2_ax.p1),
                              Segment(p, c2_ax.p2)]))
                    facts.append(f)
                elif c1_ax.p1 != p and c2_ax.p1 == p:
                    f = Fact("eqratio", [
                        Ratio(c2_ax, Segment(p, c1_ax.p1)),
                        Ratio(Segment(p, c1_ax.p2), c2_ax)
                    ], "radax")
                    f.add_parents(p_parents)
                    f.add_parent(
                        Fact("perp", [Angle(c2_ax.p1, c2_ax.p2, c2_center)]))
                    f.add_parent(
                        Fact("cong", [
                            Segment(c1_center, c1_ax.p1),
                            Segment(c1_center, c1_ax.p2)
                        ]))
                    f.add_parent(
                        Fact("eqline",
                             [Segment(p, c1_ax.p1),
                              Segment(p, c1_ax.p2)]))
                    facts.append(f)
                else:
                    f = Fact("eqratio", [
                        Ratio(Segment(p, c1_ax.p1), Segment(p, c2_ax.p1)),
                        Ratio(Segment(p, c2_ax.p2), Segment(p, c1_ax.p2))
                    ], "radax")
                    f.add_parents(p_parents)
                    f.add_parent(
                        Fact("cong", [
                            Segment(c1_center, c1_ax.p1),
                            Segment(c1_center, c1_ax.p2)
                        ]))
                    f.add_parent(
                        Fact("cong", [
                            Segment(c2_center, c2_ax.p1),
                            Segment(c2_center, c2_ax.p2)
                        ]))
                    f.add_parent(
                        Fact("eqline",
                             [Segment(p, c1_ax.p1),
                              Segment(p, c1_ax.p2)]))
                    f.add_parent(
                        Fact("eqline",
                             [Segment(p, c2_ax.p1),
                              Segment(p, c2_ax.p2)]))
                    facts.append(f)
    # on one circle implies on another circle
    if diagram.database.is_eqcircle(c1_p, c1):
        f = Fact("eqcircle", [p_c2, c2_p], "radax")
        if p_c1 != c1_p:
            f.add_parent(Fact("eqcircle", [p_c1, c1_p]))
        f.add_parents(p_parents)
        facts.append(f)
    elif diagram.database.is_eqcircle(c2_p, c2):
        f = Fact("eqcircle", [p_c1, c1_p], "radax")
        if p_c2 != c2_p:
            f.add_parent(Fact("eqcircle", [p_c2, c2_p]))
        f.add_parents(p_parents)
        facts.append(f)
    if len(diagram.database.radaxes[c1][c2]) > 1:
        # on center line implies perp
        s1_p = Segment(c1_center, p)
        s2_p = Segment(c2_center, p)
        if diagram.database.is_eqline(s1_p, s2_p):
            for q in diagram.database.radaxes[c1][c2]:
                if q != p:
                    break
            q_parents, q_c1, q_c2 = diagram.database.radaxes[c1][c2][q]
            f = Fact("perp", [Angle(q, p, c1_center)], "radax")
            f.add_parent(Fact("eqline", [s1_p, s2_p]))
            if p_c1 != q_c1:
                f.add_parent(Fact("eqcircle", [p_c1, q_c1]))
            if p_c2 != q_c2:
                f.add_parent(Fact("eqcircle", [p_c2, q_c2]))
            f.add_parents(p_parents + q_parents)
            facts.append(f)
        # perp implies on center line
        for q in diagram.database.radaxes[c1][c2]:
            if q == p:
                continue
            a1 = Angle(c1_center, p, q)
            a2 = Angle(c2_center, p, q)
            if diagram.database.is_perp(a1) or diagram.database.is_perp(a2):
                q_parents, q_c1, q_c2 = diagram.database.radaxes[c1][c2][q]
                f = Fact("eqline",
                         [Segment(p, c1_center),
                          Segment(p, c2_center)], "radax")
                if diagram.database.is_perp(a1):
                    f.add_parent(Fact("perp", [a1]))
                else:
                    f.add_parent(Fact("perp", [a2]))
                if p_c1 != q_c1:
                    f.add_parent(Fact("eqcircle", [p_c1, q_c1]))
                if p_c2 != q_c2:
                    f.add_parent(Fact("eqcircle", [p_c2, q_c2]))
                f.add_parents(p_parents + q_parents)
                facts.append(f)
                break
        # another point on center line implies perp
        for z in diagram.database.radaxes[c1][c2]:
            if z == p:
                continue
            if diagram.database.is_eqline(Segment(c1_center, z),
                                          Segment(c2_center, z)):
                z_parents, z_c1, z_c2 = diagram.database.radaxes[c1][c2][z]
                f = Fact("perp", [Angle(c1_center, z, p)], "radax")
                f.add_parent(
                    Fact("eqline",
                         [Segment(c1_center, z),
                          Segment(c2_center, z)]))
                if p_c1 != z_c1:
                    f.add_parent(Fact("eqcircle", [p_c1, z_c1]))
                if p_c2 != z_c2:
                    f.add_parent(Fact("eqcircle", [p_c2, z_c2]))
                f.add_parents(p_parents + z_parents)
                facts.append(f)
                break
        # another point perp implies on center line
        for z in diagram.database.radaxes[c1][c2]:
            if z == p:
                continue
            a1 = Angle(c1_center, z, p)
            a2 = Angle(c2_center, z, p)
            if diagram.database.is_perp(a1) or diagram.database.is_perp(a2):
                z_parents, z_c1, z_c2 = diagram.database.radaxes[c1][c2][z]
                f = Fact("eqline",
                         [Segment(z, c1_center),
                          Segment(z, c2_center)], "radax")
                if diagram.database.is_perp(a1):
                    f.add_parent(Fact("perp", [a1]))
                else:
                    f.add_parent(Fact("perp", [a2]))
                if p_c1 != z_c1:
                    f.add_parent(Fact("eqcircle", [p_c1, z_c1]))
                if p_c2 != z_c2:
                    f.add_parent(Fact("eqcircle", [p_c2, z_c2]))
                f.add_parents(p_parents + z_parents)
                facts.append(f)
                break
    if len(diagram.database.radaxes[c1][c2]) > 2:
        q_k = []
        for pp in diagram.database.radaxes[c1][c2]:
            if pp != p:
                q_k.append(pp)
            if len(q_k) == 2:
                break
        q, k = q_k[0], q_k[1]
        if not (diagram.database.is_eqline(Segment(p, q), Segment(p, k))
                or diagram.database.is_eqline(Segment(p, q), Segment(q, k))
                or diagram.database.is_eqline(Segment(p, k), Segment(
                    q, k))):  # avoid recursion from eqline points are on radax
            # eqline for points on radax
            f = Fact("eqline", [Segment(p, q), Segment(p, k)], "radax")
            q_parents, q_c1, q_c2 = diagram.database.radaxes[c1][c2][q]
            k_parents, k_c1, k_c2 = diagram.database.radaxes[c1][c2][k]
            if not (f in p_parents or f in q_parents or f in k_parents):
                if p_c1 != q_c1:
                    f.add_parent(Fact("eqcircle", [p_c1, q_c1]))
                if p_c2 != q_c2:
                    f.add_parent(Fact("eqcircle", [p_c2, q_c2]))
                if p_c1 != k_c1:
                    f.add_parent(Fact("eqcircle", [p_c1, k_c1]))
                if p_c2 != k_c2:
                    f.add_parent(Fact("eqcircle", [p_c2, k_c2]))
                f.add_parents(p_parents + q_parents + k_parents)
                facts.append(f)
    # eqline points are on radax
    to_add_eqline = []
    for q in diagram.database.radaxes[c1][c2]:
        if q != p:
            break
    if q != p:
        l_pq = Segment(p, q)
        l_rep = diagram.database.inverse_eqline[l_pq]
        pts = diagram.database.lines_points[l_rep]
        if len(pts) >= 3:
            for pp in pts:
                if pp in [p, q] or not diagram.database.is_eqline(
                        Segment(p, pp), Segment(q, pp)):
                    continue
                to_add_eqline.append((pp, q))
    # eqangle and radax implies radax
    p_on_line = on_same_line(
        *[diagram.point_dict[pt] for pt in [c1_center, c2_center, p]])
    to_add_foot = []
    s = Segment(c1_center, p)
    if not p_on_line and s in diagram.database.h_segments_perps:
        for a in diagram.database.h_segments_perps[s]:
            q = a.name[1]
            if q not in diagram.database.radaxes[c1][
                    c2] and diagram.database.is_perp(Angle(c2_center, q, p)):
                to_add_foot.append((c1, c2, q))
    to_add_side = []
    if p_on_line and s in diagram.database.segments_perps:
        for a in diagram.database.segments_perps[s]:
            if a.name[1] != p:
                continue
            q = a.name[0] if a.name[0] != c1_center else a.name[2]
            if q not in diagram.database.radaxes[c1][
                    c2] and diagram.database.is_perp(Angle(c2_center, p, q)):
                to_add_side.append((c1, c2, q))
    # Monge Theorem (radical center)
    to_add_monge = []
    for c, cc in [(c1, c2), (c2, c1)]:
        cc_all = diagram.database.circles_circles[cc]
        cc_center = cc_all.center
        cc_p = next(iter(cc_all.points))
        cc_r = (diagram.point_dict[cc_center] -
                diagram.point_dict[cc_p]).norm()
        for ccc in diagram.database.points_radaxes[p][c]:
            if ccc in [c1, c2
                       ] or ccc in diagram.database.points_radaxes[p][cc]:
                continue
            ccc_all = diagram.database.circles_circles[ccc]
            ccc_center = ccc_all.center
            ccc_p = next(iter(ccc_all.points))
            ccc_r = (diagram.point_dict[ccc_center] -
                     diagram.point_dict[ccc_p]).norm()
            if cc_center == ccc_center and isclose(cc_r, ccc_r):
                continue
            to_add_monge.append((c, cc, ccc))
    # final recursion
    for pp, q in to_add_eqline:
        q_parents = diagram.database.radaxes[c1][c2][q][0]
        parents = [Fact("eqline", [Segment(p, pp), Segment(q, pp)])]
        parents += unique(p_parents + q_parents)
        status = add_radax(diagram, c1, c2, pp, (parents, p_c1, p_c2))
        if status:
            facts += radax(diagram, c1, c2, pp)
    for c, cc, q in to_add_foot:
        parents = [
            Fact("perp", [Angle(c1_center, q, p)]),
            Fact("perp", [Angle(c2_center, q, p)])
        ]
        parents += p_parents
        status = add_radax(diagram, c, cc, q, (parents, p_c1, p_c2))
        if status:
            facts += radax(diagram, c, cc, q)
    for c, cc, q in to_add_side:
        parents = [
            Fact("perp", [Angle(c1_center, p, q)]),
            Fact("perp", [Angle(c2_center, p, q)])
        ]
        parents += p_parents
        status = add_radax(diagram, c, cc, q, (parents, p_c1, p_c2))
        if status:
            facts += radax(diagram, c, cc, q)
    for c, cc, ccc in to_add_monge:
        cc_parents, cc_c1, cc_c2 = diagram.database.radaxes[c][cc][p]
        ccc_parents, ccc_c1, ccc_c2 = diagram.database.radaxes[c][ccc][p]
        parents = unique(cc_parents + ccc_parents)
        if cc_c1 != ccc_c1:
            parents.append(Fact("eqcircle", [cc_c1, ccc_c1]))
        status = add_radax(diagram, cc, ccc, p, (parents, cc_c2, ccc_c2))
        if status:
            facts += radax(diagram, cc, ccc, p)
    return facts
