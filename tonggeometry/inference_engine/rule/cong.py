r"""Congruent-segment-related rules."""

from itertools import product
from typing import TYPE_CHECKING, List

from tonggeometry.constructor.primitives import on_same_line
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Ratio, Segment
from tonggeometry.inference_engine.util import OrderedSet, four_joint
from tonggeometry.util import isclose

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def cong_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Isosceles triangle, equal sides, equal angles."""
    facts = []
    s1, s2 = fact.objects
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    if on_same_line(*[diagram.point_dict[p] for p in s12_p_set]):
        return facts
    s12_p_set.pop(s12_joint)
    B = s12_joint
    A, C = s12_p_set
    f = Fact("eqangle", [Angle(B, A, C), Angle(A, C, B)], "cong_to_eqangle")
    f.add_parent(fact)
    facts.append(f)
    return facts


def eqangle_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Isosceles triangle, equal sides, equal angles."""
    facts = []
    a1, a2 = fact.objects
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if s1 == s4:
        s1, s2, s3, s4 = s2, s1, s4, s3
    if s2 != s3 or len(set([s1, s2, s3, s4])) != 3 or s1.p2 != s4.p2:
        return facts
    if on_same_line(*[diagram.point_dict[p] for p in a1.name]):
        return facts
    f = Fact("cong", [s1, s4], "eqangle_to_cong")
    f.add_parent(fact)
    facts.append(f)
    return facts


def cong_and_eqline_to_midp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Congruent segments on the same line imply midp."""
    facts = []
    s1, s2 = fact.objects
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    if not diagram.database.is_eqline(s1, s2):
        return facts
    s12_p_set.pop(s12_joint)
    B = s12_joint
    A, C = s12_p_set
    f = Fact("midp", [B, Segment(A, C)], "cong_and_eqline_to_midp")
    f.add_parent(fact)
    f.add_parent(Fact("eqline", [s1, s2]))
    facts.append(f)
    return facts


def eqline_and_cong_to_midp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Congruent segments on the same line imply midp."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    if not diagram.database.is_cong(s1, s2):
        return facts
    s12_p_set.pop(s12_joint)
    B = s12_joint
    A, C = s12_p_set
    f = Fact("midp", [B, Segment(A, C)], "eqline_and_cong_to_midp")
    f.add_parent(fact)
    f.add_parent(Fact("cong", [s1, s2]))
    facts.append(f)
    return facts


def cong_to_eqratio(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Congruent segments imply eqratio. One-stop link between any two ratios."""
    facts = []
    s1, s2 = fact.objects
    norm = (diagram.point_dict[s1.p1] - diagram.point_dict[s1.p2]).norm()
    for rep in diagram.database.cong:
        if rep == diagram.database.inverse_cong[s1]:
            continue
        norm_rep = (diagram.point_dict[rep.p1] -
                    diagram.point_dict[rep.p2]).norm()
        if isclose(norm, norm_rep):
            continue
        f = Fact("eqratio", [Ratio(s1, rep), Ratio(s2, rep)],
                 "cong_to_eqratio")
        f.add_parent(fact)
        facts.append(f)
    return facts


def cong_and_eqline_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Congruent segments arithmetics."""
    facts = []
    s1, s2 = fact.objects
    s12_joint, _ = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if s12_joint:
        return facts
    if not diagram.database.is_eqline(s1, s2):
        return facts
    A, B = s1.p1, s1.p2
    C, D = s2.p1, s2.p2
    p_A = diagram.point_dict[A]
    p_B = diagram.point_dict[B]
    p_C = diagram.point_dict[C]
    p_D = diagram.point_dict[D]
    if p_B < p_A:
        A, B = B, A
    if p_D < p_C:
        C, D = D, C
    f = Fact("cong", [Segment(A, C), Segment(B, D)], "cong_and_eqline_to_cong")
    f.add_parent(fact)
    f.add_parent(Fact("eqline", [s1, s2]))
    facts.append(f)
    return facts


def eqline_and_cong_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Congruent segments arithmetics."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, _ = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if s12_joint:
        return facts
    if not diagram.database.is_cong(s1, s2):
        return facts
    A, B = s1.p1, s1.p2
    C, D = s2.p1, s2.p2
    p_A = diagram.point_dict[A]
    p_B = diagram.point_dict[B]
    p_C = diagram.point_dict[C]
    p_D = diagram.point_dict[D]
    if p_B < p_A:
        A, B = B, A
    if p_D < p_C:
        C, D = D, C
    f = Fact("cong", [Segment(A, C), Segment(B, D)], "eqline_and_cong_to_cong")
    f.add_parent(fact)
    f.add_parent(Fact("cong", [s1, s2]))
    facts.append(f)
    return facts


def cong_and_eqline_to_l(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Building an L shape for isosceles triangle."""
    facts = []
    s1, s2 = fact.objects
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    O = s12_joint
    A, B = s12_p_set
    if B < A:
        A, B = B, A
    s = Segment(A, B)
    l = diagram.database.inverse_eqline[s]
    for p in diagram.database.lines_points[l]:
        if p in [O, A, B]:
            continue
        key = O + p + A + B
        if key in diagram.database.l:
            continue
        p_p = diagram.point_dict[p]
        p_A = diagram.point_dict[A]
        p_B = diagram.point_dict[B]
        if p_A < p_p < p_B or p_B < p_p < p_A:
            side = "in"
        else:
            side = "out"
        facts += l_and_cong_to_ll_stick(diagram, key, side)
        facts += l_and_cong_to_ll_radius(diagram, key, side)
        diagram.database.l[key] = side
        stick_key = O + p
        radius_key = O + A
        ratio_key = p + A + B
        if stick_key not in diagram.database.l_stick:
            diagram.database.l_stick[stick_key] = OrderedSet()
        diagram.database.l_stick[stick_key][A + B] = None
        if radius_key not in diagram.database.l_radius:
            diagram.database.l_radius[radius_key] = OrderedSet()
        diagram.database.l_radius[radius_key][p + B] = None
        if ratio_key not in diagram.database.l_ratio:
            diagram.database.l_ratio[ratio_key] = OrderedSet()
        diagram.database.l_ratio[ratio_key][O] = None
    return facts


def eqline_and_cong_to_l(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Building an L shape for isosceles triangle."""
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
    if B < A:
        A, B = B, A
    if not (A in diagram.database.points_congs
            and B in diagram.database.points_congs):
        return facts
    for s in diagram.database.points_congs[A]:
        O = s.p1 if s.p1 != A else s.p2
        if O in [B, p]:
            continue
        if not diagram.database.is_cong(Segment(O, A), Segment(O, B)):
            continue
        key = O + p + A + B
        if key in diagram.database.l:
            continue
        p_p = diagram.point_dict[p]
        p_A = diagram.point_dict[A]
        p_B = diagram.point_dict[B]
        if p_A < p_p < p_B or p_B < p_p < p_A:
            side = "in"
        else:
            side = "out"
        facts += l_and_cong_to_ll_stick(diagram, key, side)
        facts += l_and_cong_to_ll_radius(diagram, key, side)
        diagram.database.l[key] = side
        stick_key = O + p
        radius_key = O + A
        ratio_key = p + A + B
        if stick_key not in diagram.database.l_stick:
            diagram.database.l_stick[stick_key] = OrderedSet()
        diagram.database.l_stick[stick_key][A + B] = None
        if radius_key not in diagram.database.l_radius:
            diagram.database.l_radius[radius_key] = OrderedSet()
        diagram.database.l_radius[radius_key][p + B] = None
        if ratio_key not in diagram.database.l_ratio:
            diagram.database.l_ratio[ratio_key] = OrderedSet()
        diagram.database.l_ratio[ratio_key][O] = None
    return facts


def l_and_cong_to_ll_stick(diagram: 'Diagram', l: str,
                           side: str) -> List[Fact]:
    """On the L shape, PA*PB = PO**2-OA**2."""
    facts = []
    O, p, A, B = l
    s = Segment(O, p)
    if s not in diagram.database.inverse_cong:
        eqclass = OrderedSet.fromkeys([s])
    else:
        rep = diagram.database.inverse_cong[s]
        eqclass = diagram.database.cong[rep]
    for ss in eqclass:
        # including the same where ss == s
        for O_p, p_p in [(ss.p1, ss.p2), (ss.p2, ss.p1)]:
            if O_p + p_p not in diagram.database.l_stick:
                continue
            for sss in diagram.database.l_stick[O_p + p_p]:
                A_p, B_p = sss
                if O_p + p_p + A_p + B_p == l:
                    continue
                if side != diagram.database.l[O_p + p_p + A_p + B_p]:
                    continue
                s1 = Segment(O, A)
                s2 = Segment(O_p, A_p)
                r1 = Ratio(Segment(p, A), Segment(p_p, A_p))
                r2 = Ratio(Segment(p_p, B_p), Segment(p, B))
                if diagram.database.is_cong(s1, s2):
                    f = Fact("eqratio", [r1, r2], "l_and_cong_to_ll_stick")
                    if s1 != s2:
                        f.add_parent(Fact("cong", [s1, s2]))
                    if s != ss:
                        f.add_parent(Fact("cong", [s, ss]))
                    f.add_parent(Fact("cong", [Segment(O, A), Segment(O, B)]))
                    f.add_parent(Fact(
                        "eqline", [Segment(p, A), Segment(p, B)]))
                    f.add_parent(
                        Fact("cong", [Segment(O_p, A_p),
                                      Segment(O_p, B_p)]))
                    f.add_parent(
                        Fact("eqline", [Segment(p_p, A_p),
                                        Segment(p_p, B_p)]))
                    facts.append(f)
                if not diagram.database.is_cong(
                        r1.s1, r1.s2) and not diagram.database.is_cong(
                            r2.s1, r2.s2) and diagram.database.is_eqratio(
                                r1, r2):
                    f = Fact("cong", [s1, s2], "l_and_cong_to_ll_stick")
                    f.add_parent(Fact("eqratio", [r1, r2]))
                    if s != ss:
                        f.add_parent(Fact("cong", [s, ss]))
                    f.add_parent(Fact("cong", [Segment(O, A), Segment(O, B)]))
                    f.add_parent(Fact(
                        "eqline", [Segment(p, A), Segment(p, B)]))
                    f.add_parent(
                        Fact("cong", [Segment(O_p, A_p),
                                      Segment(O_p, B_p)]))
                    f.add_parent(
                        Fact("eqline", [Segment(p_p, A_p),
                                        Segment(p_p, B_p)]))
                    facts.append(f)
    return facts


def l_and_cong_to_ll_radius(diagram: 'Diagram', l: str,
                            side: str) -> List[Fact]:
    """On the L shape, PA*PB = PO**2-OA**2."""
    facts = []
    O, p, A, B = l
    s = Segment(O, A)
    # s must be in inverse_cong
    rep = diagram.database.inverse_cong[s]
    eqclass = diagram.database.cong[rep]
    for ss in eqclass:
        # including the same where ss == s
        for O_p, A_p in [(ss.p1, ss.p2), (ss.p2, ss.p1)]:
            if O_p + A_p not in diagram.database.l_radius:
                continue
            for sss in diagram.database.l_radius[O_p + A_p]:
                p_p, B_p = sss
                if O_p + p_p + A_p + B_p == l:
                    continue
                if side != diagram.database.l[O_p + p_p + A_p + B_p]:
                    continue
                s1 = Segment(O, p)
                s2 = Segment(O_p, p_p)
                r1 = Ratio(Segment(p, A), Segment(p_p, A_p))
                r2 = Ratio(Segment(p_p, B_p), Segment(p, B))
                if diagram.database.is_cong(s1, s2):
                    f = Fact("eqratio", [r1, r2], "l_and_cong_to_ll_radius")
                    if s1 != s2:
                        f.add_parent(Fact("cong", [s1, s2]))
                    if s != ss:
                        f.add_parent(Fact("cong", [s, ss]))
                    f.add_parent(Fact("cong", [Segment(O, A), Segment(O, B)]))
                    f.add_parent(Fact(
                        "eqline", [Segment(p, A), Segment(p, B)]))
                    f.add_parent(
                        Fact("cong", [Segment(O_p, A_p),
                                      Segment(O_p, B_p)]))
                    f.add_parent(
                        Fact("eqline", [Segment(p_p, A_p),
                                        Segment(p_p, B_p)]))
                    facts.append(f)
                if not diagram.database.is_cong(
                        r1.s1, r1.s2) and not diagram.database.is_cong(
                            r2.s1, r2.s2) and diagram.database.is_eqratio(
                                r1, r2):
                    f = Fact("cong", [s1, s2], "l_and_cong_to_ll_radius")
                    f.add_parent(Fact("eqratio", [r1, r2]))
                    if s != ss:
                        f.add_parent(Fact("cong", [s, ss]))
                    f.add_parent(Fact("cong", [Segment(O, A), Segment(O, B)]))
                    f.add_parent(Fact(
                        "eqline", [Segment(p, A), Segment(p, B)]))
                    f.add_parent(
                        Fact("cong", [Segment(O_p, A_p),
                                      Segment(O_p, B_p)]))
                    f.add_parent(
                        Fact("eqline", [Segment(p_p, A_p),
                                        Segment(p_p, B_p)]))
                    facts.append(f)
    return facts


# l_and_eqratio unnecessary as has been handled with triggers in cong


def cong_to_ll_stick(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """On the L shape, PA*PB = PO**2-OA**2."""
    facts = []
    s1, s2 = fact.objects
    OpAB = []
    OpAB_p = []
    for Op in [s1.p1 + s1.p2, s1.p2 + s1.p1]:
        if Op not in diagram.database.l_stick:
            continue
        O, p = Op
        for s in diagram.database.l_stick[Op]:
            A, B = s
            OpAB.append(O + p + A + B)
    for O_p_p_p in [s2.p1 + s2.p2, s2.p2 + s2.p1]:
        if O_p_p_p not in diagram.database.l_stick:
            continue
        O_p, p_p = O_p_p_p
        for ss in diagram.database.l_stick[O_p_p_p]:
            A_p, B_p = ss
            OpAB_p.append(O_p + p_p + A_p + B_p)
    for l, ll in product(OpAB, OpAB_p):
        if diagram.database.l[l] != diagram.database.l[ll]:
            continue
        O, p, A, B = l
        O_p, p_p, A_p, B_p = ll
        ss1 = Segment(O, A)
        ss2 = Segment(O_p, A_p)
        r1 = Ratio(Segment(p, A), Segment(p_p, A_p))
        r2 = Ratio(Segment(p_p, B_p), Segment(p, B))
        if diagram.database.is_cong(ss1, ss2):
            f = Fact("eqratio", [r1, r2], "cong_to_ll_stick")
            f.add_parent(fact)
            if ss1 != ss2:
                f.add_parent(Fact("cong", [ss1, ss2]))
            f.add_parent(Fact("cong", [Segment(O, A), Segment(O, B)]))
            f.add_parent(Fact("eqline", [Segment(p, A), Segment(p, B)]))
            f.add_parent(Fact("cong", [Segment(O_p, A_p), Segment(O_p, B_p)]))
            f.add_parent(
                Fact("eqline",
                     [Segment(p_p, A_p), Segment(p_p, B_p)]))
            facts.append(f)
        if not diagram.database.is_cong(
                r1.s1, r1.s2) and not diagram.database.is_cong(
                    r2.s1, r2.s2) and diagram.database.is_eqratio(r1, r2):
            f = Fact("cong", [ss1, ss2], "cong_to_ll_stick")
            f.add_parent(fact)
            f.add_parent(Fact("eqratio", [r1, r2]))
            f.add_parent(Fact("cong", [Segment(O, A), Segment(O, B)]))
            f.add_parent(Fact("eqline", [Segment(p, A), Segment(p, B)]))
            f.add_parent(Fact("cong", [Segment(O_p, A_p), Segment(O_p, B_p)]))
            f.add_parent(
                Fact("eqline",
                     [Segment(p_p, A_p), Segment(p_p, B_p)]))
            facts.append(f)
    return facts


def cong_to_ll_radius(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """On the L shape, PA*PB = PO**2-OA**2."""
    facts = []
    s1, s2 = fact.objects
    OpAB = []
    OpAB_p = []
    for O_A in [s1.p1 + s1.p2, s1.p2 + s1.p1]:
        if O_A not in diagram.database.l_radius:
            continue
        O, A = O_A
        for s in diagram.database.l_radius[O_A]:
            p, B = s
            OpAB.append(O + p + A + B)
    for O_p_A_p in [s2.p1 + s2.p2, s2.p2 + s2.p1]:
        if O_p_A_p not in diagram.database.l_radius:
            continue
        O_p, A_p = O_p_A_p
        for ss in diagram.database.l_radius[O_p_A_p]:
            p_p, B_p = ss
            OpAB_p.append(O_p + p_p + A_p + B_p)
    for l, ll in product(OpAB, OpAB_p):
        if diagram.database.l[l] != diagram.database.l[ll]:
            continue
        O, p, A, B = l
        O_p, p_p, A_p, B_p = ll
        ss1 = Segment(O, p)
        ss2 = Segment(O_p, p_p)
        r1 = Ratio(Segment(p, A), Segment(p_p, A_p))
        r2 = Ratio(Segment(p_p, B_p), Segment(p, B))
        if diagram.database.is_cong(ss1, ss2):
            f = Fact("eqratio", [r1, r2], "cong_to_ll_radius")
            f.add_parent(fact)
            if ss1 != ss2:
                f.add_parent(Fact("cong", [ss1, ss2]))
            f.add_parent(Fact("cong", [Segment(O, A), Segment(O, B)]))
            f.add_parent(Fact("eqline", [Segment(p, A), Segment(p, B)]))
            f.add_parent(Fact("cong", [Segment(O_p, A_p), Segment(O_p, B_p)]))
            f.add_parent(
                Fact("eqline",
                     [Segment(p_p, A_p), Segment(p_p, B_p)]))
            facts.append(f)
        if not diagram.database.is_cong(
                r1.s1, r1.s2) and not diagram.database.is_cong(
                    r2.s1, r2.s2) and diagram.database.is_eqratio(r1, r2):
            f = Fact("cong", [ss1, ss2], "cong_to_ll_radius")
            f.add_parent(fact)
            f.add_parent(Fact("eqratio", [r1, r2]))
            f.add_parent(Fact("cong", [Segment(O, A), Segment(O, B)]))
            f.add_parent(Fact("eqline", [Segment(p, A), Segment(p, B)]))
            f.add_parent(Fact("cong", [Segment(O_p, A_p), Segment(O_p, B_p)]))
            f.add_parent(
                Fact("eqline",
                     [Segment(p_p, A_p), Segment(p_p, B_p)]))
            facts.append(f)
    return facts


def eqratio_to_ll(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """On the L shape, PA*PB = PO**2-OA**2."""
    facts = []
    r1, r2 = fact.objects
    if ((diagram.database.is_cong(r1.s1, r2.s2)
         and diagram.database.is_cong(r1.s2, r2.s1))
            or diagram.database.is_cong(r1.s1, r1.s2)
            or diagram.database.is_cong(r2.s1, r2.s2)):
        return facts
    s1, s3 = r1.s1, r1.s2
    s4, s2 = r2.s1, r2.s2
    if s1 == s2 or s3 == s4:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    s34_joint, s34_p_set = four_joint(s3.p1, s3.p2, s4.p1, s4.p2)
    if not (s12_joint and s34_joint):
        return facts
    s12_p_set.pop(s12_joint)
    s34_p_set.pop(s34_joint)
    p = s12_joint
    A, B = s12_p_set
    if B < A:
        A, B = B, A
    p_p = s34_joint
    A_p, B_p = s34_p_set
    if B_p < A_p:
        A_p, B_p = B_p, A_p
    p_A_B = p + A + B
    p_p_A_p_B_p = p_p + A_p + B_p
    if not (p_A_B in diagram.database.l_ratio
            and p_p_A_p_B_p in diagram.database.l_ratio):
        return facts
    for O in diagram.database.l_ratio[p_A_B]:
        for O_p in diagram.database.l_ratio[p_p_A_p_B_p]:
            if diagram.database.l[O + p + A +
                                  B] != diagram.database.l[O_p + p_p + A_p +
                                                           B_p]:
                continue
            s1_stick = Segment(O, p)
            s2_stick = Segment(O_p, p_p)
            s1_radius = Segment(O, A)
            s2_radius = Segment(O_p, A_p)
            if diagram.database.is_cong(s1_stick, s2_stick):
                f = Fact("cong", [s1_radius, s2_radius], "eqratio_to_ll")
                f.add_parent(fact)
                if s1_stick != s2_stick:
                    f.add_parent(Fact("cong", [s1_stick, s2_stick]))
                f.add_parent(Fact("cong", [Segment(O, A), Segment(O, B)]))
                f.add_parent(Fact("eqline", [Segment(p, A), Segment(p, B)]))
                f.add_parent(
                    Fact("cong", [Segment(O_p, A_p),
                                  Segment(O_p, B_p)]))
                f.add_parent(
                    Fact("eqline", [Segment(p_p, A_p),
                                    Segment(p_p, B_p)]))
                facts.append(f)
            if diagram.database.is_cong(s1_radius, s2_radius):
                f = Fact("cong", [s1_stick, s2_stick], "eqratio_to_ll")
                f.add_parent(fact)
                if s1_radius != s2_radius:
                    f.add_parent(Fact("cong", [s1_radius, s2_radius]))
                f.add_parent(Fact("cong", [Segment(O, A), Segment(O, B)]))
                f.add_parent(Fact("eqline", [Segment(p, A), Segment(p, B)]))
                f.add_parent(
                    Fact("cong", [Segment(O_p, A_p),
                                  Segment(O_p, B_p)]))
                f.add_parent(
                    Fact("eqline", [Segment(p_p, A_p),
                                    Segment(p_p, B_p)]))
                facts.append(f)
    return facts
