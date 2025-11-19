r"""Eqangle-related rules."""

from itertools import product
from typing import TYPE_CHECKING, List

from tonggeometry.constructor.primitives import intersect, on_same_line
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Ratio, Segment
from tonggeometry.inference_engine.util import four_joint

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def eqangle_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """eqangle(ABC,DBF) => eqangle(ABD,CBF)."""
    facts = []
    a1, a2 = fact.objects
    if a1 == a2:
        return facts
    if a1.name[1] != a2.name[1]:
        return facts
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if len(set([s1, s2, s3, s4])) != 4:
        return facts
    a1_p = Angle(a1.name[0], a1.name[1], a2.name[0])
    a2_p = Angle(a1.name[2], a2.name[1], a2.name[2])
    f = Fact("eqangle", [a1_p, a2_p], "eqangle_to_eqangle")
    f.add_parent(fact)
    facts.append(f)
    return facts


def eqangle_and_eqline_to_eqline(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """One pair in eqangle eqline, the other pair eqline."""
    facts = []
    a1, a2 = fact.objects
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if diagram.database.is_eqline(s1, s2):
        f = Fact("eqline", [s3, s4], "eqangle_and_eqline_to_eqline")
        f.add_parent(fact)
        f.add_parent(Fact("eqline", [s1, s2]))
        facts.append(f)
    elif diagram.database.is_eqline(s3, s4):
        f = Fact("eqline", [s1, s2], "eqangle_and_eqline_to_eqline")
        f.add_parent(fact)
        f.add_parent(Fact("eqline", [s3, s4]))
        facts.append(f)
    return facts


def eqline_and_eqangle_to_eqline(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """One pair in eqangle eqline, the other pair eqline."""
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
    a = Angle(A, B, C)
    if a in diagram.database.inverse_eqangle:
        eqclass_rep = diagram.database.inverse_eqangle[a]
        eqclass = diagram.database.eqangle[eqclass_rep]
        a = eqclass[a]
        for aa in eqclass:
            if aa != a:
                s3, s4 = aa.s1, aa.s2
                f = Fact("eqline", [s3, s4], "eqline_and_eqangle_to_eqline")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [a, aa]))
                facts.append(f)
    return facts


def eqangle_and_perp_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """One pair in eqangle perp, the other pair perp as well."""
    facts = []
    a1, a2 = fact.objects
    if diagram.database.is_perp(a1):
        f = Fact("perp", [a2], "eqangle_and_perp_to_perp")
        f.add_parent(fact)
        f.add_parent(Fact("perp", [a1]))
        facts.append(f)
    elif diagram.database.is_perp(a2):
        f = Fact("perp", [a1], "eqangle_and_perp_to_perp")
        f.add_parent(fact)
        f.add_parent(Fact("perp", [a2]))
        facts.append(f)
    return facts


def perp_and_eqangle_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """One pair in eqangle perp, the other pair perp as well."""
    facts = []
    a = fact.objects[0]
    if a not in diagram.database.inverse_eqangle:
        return facts
    eqclass_rep = diagram.database.inverse_eqangle[a]
    eqclass = diagram.database.eqangle[eqclass_rep]
    a = eqclass[a]
    for aa in eqclass:
        if aa == a:
            continue
        f = Fact("perp", [aa], "perp_and_eqangle_to_perp")
        f.add_parent(fact)
        f.add_parent(Fact("eqangle", [a, aa]))
        facts.append(f)
    return facts


def eqangle_and_para_to_para(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """One pair in eqangle para, the other pair para as well."""
    facts = []
    a1, a2 = fact.objects
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if diagram.database.is_para(s1, s3):
        f = Fact("para", [s2, s4], "eqangle_and_para_to_para")
        f.add_parent(fact)
        f.add_parent(Fact("para", [s1, s3]))
        facts.append(f)
    elif diagram.database.is_para(s2, s4):
        f = Fact("para", [s1, s3], "eqangle_and_para_to_para")
        f.add_parent(fact)
        f.add_parent(Fact("para", [s2, s4]))
        facts.append(f)
    return facts


def para_and_eqangle_to_para(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """One pair in eqangle para, the other pair para as well."""
    facts = []
    s1, s2 = fact.objects
    if on_same_line(*[diagram.point_dict[p] for p in [s1.p1, s1.p2, s2.p1]]):
        return facts
    if not (s1 in diagram.database.segments_eqangles
            and s2 in diagram.database.segments_eqangles):
        return facts
    partition = {}
    for a in diagram.database.segments_eqangles[s1]:
        rep = diagram.database.inverse_eqangle[a]
        if a.s1 == s1:
            idx = 0
            ss = a.s2
        else:
            idx = 1
            ss = a.s1
        if rep in partition:
            partition[rep][0].append((a, idx, ss))
        else:
            partition[rep] = [[(a, idx, ss)], []]
    for a in diagram.database.segments_eqangles[s2]:
        rep = diagram.database.inverse_eqangle[a]
        if a.s1 == s2:
            idx = 0
            ss = a.s2
        else:
            idx = 1
            ss = a.s1
        if rep in partition:
            partition[rep][1].append((a, idx, ss))
    for pairs in partition.values():
        for a1_idx1_s3, a2_idx2_s4 in product(pairs[0], pairs[1]):
            a1, idx1, s3 = a1_idx1_s3
            a2, idx2, s4 = a2_idx2_s4
            if idx1 == idx2:
                f = Fact("para", [s3, s4], "para_and_eqangle_to_para")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [a1, a2]))
                facts.append(f)
    return facts


def eqangle_and_eqline_to_para(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """eqangle(l1,l2,l3,l2) => para(l1,l3)."""
    facts = []
    a1, a2 = fact.objects
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if diagram.database.is_eqline(
            s1, s3) and not diagram.database.is_eqline(s2, s4):
        f = Fact("para", [s2, s4], "eqangle_and_eqline_to_para")
        f.add_parent(fact)
        if s1 != s3:
            f.add_parent(Fact("eqline", [s1, s3]))
        facts.append(f)
    if diagram.database.is_eqline(
            s2, s4) and not diagram.database.is_eqline(s1, s3):
        f = Fact("para", [s1, s3], "eqangle_and_eqline_to_para")
        f.add_parent(fact)
        if s2 != s4:
            f.add_parent(Fact("eqline", [s2, s4]))
        facts.append(f)
    return facts


def eqline_and_eqangle_to_para(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """eqangle(l1,l2,l3,l2) => para(l1,l3)."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    if not (s1 in diagram.database.segments_eqangles
            and s2 in diagram.database.segments_eqangles):
        return facts
    partition = {}
    for a in diagram.database.segments_eqangles[s1]:
        rep = diagram.database.inverse_eqangle[a]
        if a.s1 == s1:
            idx = 0
            ss = a.s2
        else:
            idx = 1
            ss = a.s1
        if rep in partition:
            partition[rep][0].append((a, idx, ss))
        else:
            partition[rep] = [[(a, idx, ss)], []]
    for a in diagram.database.segments_eqangles[s2]:
        rep = diagram.database.inverse_eqangle[a]
        if a.s1 == s2:
            idx = 0
            ss = a.s2
        else:
            idx = 1
            ss = a.s1
        if rep in partition:
            partition[rep][1].append((a, idx, ss))
    for pairs in partition.values():
        for a1_idx1_s3, a2_idx2_s4 in product(pairs[0], pairs[1]):
            a1, idx1, s3 = a1_idx1_s3
            a2, idx2, s4 = a2_idx2_s4
            if idx1 == idx2:
                f = Fact("para", [s3, s4], "eqline_and_eqangle_to_para")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [a1, a2]))
                facts.append(f)
    return facts


def eqangle_to_eqline_or_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """eqangle(l1,l2,l2,l1) => para(l1,l2) / perp(l1,l2)."""
    facts = []
    a1, a2 = fact.objects
    if a1 != a2:
        return facts
    is_perp = not on_same_line(*[diagram.point_dict[pt] for pt in a1.name])
    if is_perp:
        f = Fact("perp", [a1], "eqangle_to_eqline_or_perp")
    else:
        f = Fact("eqline", [a1.s1, a1.s2], "eqangle_to_eqline_or_perp")
    f.add_parent(fact)
    facts.append(f)
    return facts


def eqangle_and_eqline_to_eqangle(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """Angles' trivial extensions. One-stop link between any two angles."""
    facts = []
    a1, a2 = fact.objects
    for a, aa in [(a1, a2), (a2, a1)]:
        s = a.s1
        l_s = diagram.database.inverse_eqline[s]
        for p in diagram.database.lines_points[l_s]:
            if p in [
                    s.p1, s.p2
            ] or not diagram.database.is_eqline(s, Segment(p, a.name[1])):
                continue
            f = Fact("eqangle", [a, Angle(p, a.name[1], a.name[2])],
                     "eqangle_and_eqline_to_eqangle")
            f.add_parent(Fact("eqline", [s, Segment(p, a.name[1])]))
            f.add_parent(fact)
            facts.append(f)
            f = Fact("eqangle", [aa, Angle(p, a.name[1], a.name[2])],
                     "eqangle_and_eqline_to_eqangle")
            f.add_parent(Fact("eqline", [s, Segment(p, a.name[1])]))
            f.add_parent(fact)
            facts.append(f)
        ss = a.s2
        l_ss = diagram.database.inverse_eqline[ss]
        for pp in diagram.database.lines_points[l_ss]:
            if pp in [
                    ss.p1, ss.p2
            ] or not diagram.database.is_eqline(ss, Segment(a.name[1], pp)):
                continue
            f = Fact("eqangle", [a, Angle(a.name[0], a.name[1], pp)],
                     "eqangle_and_eqline_to_eqangle")
            f.add_parent(Fact("eqline", [ss, Segment(a.name[1], pp)]))
            f.add_parent(fact)
            facts.append(f)
            f = Fact("eqangle", [aa, Angle(a.name[0], a.name[1], pp)],
                     "eqangle_and_eqline_to_eqangle")
            f.add_parent(Fact("eqline", [ss, Segment(a.name[1], pp)]))
            f.add_parent(fact)
            facts.append(f)
    return facts


def eqline_and_eqangle_to_eqangle(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """Angles' trivial extensions."""
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
    for s, P, PP in [(s1, A, C), (s2, C, A)]:
        if s not in diagram.database.segments_eqangles:
            continue
        for a in diagram.database.segments_eqangles[s]:
            if a.name[1] != B:
                continue
            aa = Angle(*a.name.replace(P, PP))
            f = Fact("eqangle", [a, aa], "eqline_and_eqangle_to_eqangle")
            f.add_parent(fact)
            facts.append(f)
    return facts


def eqangle_and_eqangle_to_eqangle(diagram: 'Diagram',
                                   fact: Fact) -> List[Fact]:
    """eqangle(a1,a2) & eqangle(a3,a4) => eqangle(a1+a3,a2+a4)"""
    facts = []
    a1, a2 = fact.objects
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if len(set([s1, s2, s3, s4])) < 3:
        return facts
    if on_same_line(*[diagram.point_dict[p] for p in a1.name]):
        return facts
    partition = {"prod": {}, "div": {}}
    for a1_p in diagram.database.segments_eqangles[s1]:
        if a1_p == a1:
            continue
        s2_p = a1_p.s1 if a1_p.s1 != s1 else a1_p.s2
        if s2_p.p1 == s2.p1:
            a = Angle(s2_p.p2, s2_p.p1, s2.p2)
        elif s2_p.p2 == s2.p2:
            a = Angle(s2_p.p1, s2_p.p2, s2.p1)
        else:
            continue
        if a1_p.s1 == s1:
            key = "div"
        else:
            key = "prod"
        rep = diagram.database.inverse_eqangle[a1_p]
        if rep in partition[key]:
            partition[key][rep][0].append((a1_p, a))
        else:
            partition[key][rep] = [[(a1_p, a)], []]
    for a1_p in diagram.database.segments_eqangles[s2]:
        if a1_p == a1:
            continue
        s1_p = a1_p.s1 if a1_p.s1 != s2 else a1_p.s2
        if s1_p.p1 == s1.p1:
            a = Angle(s1.p2, s1_p.p1, s1_p.p2)
        elif s1_p.p2 == s1.p2:
            a = Angle(s1.p1, s1_p.p2, s1_p.p1)
        else:
            continue
        if a1_p.s1 == s2:
            key = "prod"
        else:
            key = "div"
        rep = diagram.database.inverse_eqangle[a1_p]
        if rep in partition[key]:
            partition[key][rep][0].append((a1_p, a))
        else:
            partition[key][rep] = [[(a1_p, a)], []]
    for a2_pp in diagram.database.segments_eqangles[s3]:
        if a2_pp == a2:
            continue
        if diagram.database.is_perp(a2_pp):
            a_s = [a2_pp, Angle(*a2_pp.name[::-1])]
        else:
            a_s = [a2_pp]
        for a2_p in a_s:
            s4_p = a2_p.s1 if a2_p.s1 != s3 else a2_p.s2
            if s4_p.p1 == s4.p1:
                a = Angle(s4_p.p2, s4_p.p1, s4.p2)
            elif s4_p.p2 == s4.p2:
                a = Angle(s4_p.p1, s4_p.p2, s4.p1)
            else:
                continue
            if a2_p.s1 == s3:
                key = "div"
            else:
                key = "prod"
            rep = diagram.database.inverse_eqangle[a2_p]
            is_perp = a2_p.name != a2_pp.name
            if rep in partition[key]:
                partition[key][rep][1].append((a2_pp, a, is_perp))
    for a2_pp in diagram.database.segments_eqangles[s4]:
        if a2_pp == a2:
            continue
        if diagram.database.is_perp(a2_pp):
            a_s = [a2_pp, Angle(*a2_pp.name[::-1])]
        else:
            a_s = [a2_pp]
        for a2_p in a_s:
            s3_p = a2_p.s1 if a2_p.s1 != s4 else a2_p.s2
            if s3_p.p1 == s3.p1:
                a = Angle(s3.p2, s3_p.p1, s3_p.p2)
            elif s3_p.p2 == s3.p2:
                a = Angle(s3.p1, s3_p.p2, s3_p.p1)
            else:
                continue
            if a2_p.s1 == s4:
                key = "prod"
            else:
                key = "div"
            rep = diagram.database.inverse_eqangle[a2_p]
            is_perp = a2_p.name != a2_pp.name
            if rep in partition[key]:
                partition[key][rep][1].append((a2_pp, a, is_perp))
    for mode in partition.values():
        for pairs in mode.values():
            for a3_a, a4_aa in product(pairs[0], pairs[1]):
                a3, a = a3_a
                a4, aa, is_perp = a4_aa
                if (a3 == a2 and a4 == a1) or a3 == a4:
                    continue
                f = Fact("eqangle", [a, aa], "eqangle_and_eqangle_to_eqangle")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [a3, a4]))
                if is_perp:
                    f.add_parent(Fact("perp", [a4]))
                facts.append(f)
    return facts


def eqangle_and_eqangle_to_eqangle_bisector(diagram: 'Diagram',
                                            fact: Fact) -> List[Fact]:
    """The bisector theorem."""
    facts = []
    a1, a2 = fact.objects
    if a1.name[1] != a2.name[1]:
        return facts
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if s1 == s4:
        s1, s2, s3, s4 = s2, s1, s4, s3
    if s2 != s3 or len(set([s1, s2, s3, s4])) != 3:
        return facts
    if on_same_line(*[diagram.point_dict[p] for p in a1.name]):
        return facts
    A = a1.name[1]
    I = s2.p2
    for B, C in [(s1.p2, s4.p2), (s4.p2, s1.p2)]:
        a3 = Angle(B, C, I)
        a4 = Angle(I, C, A)
        if diagram.database.is_eqangle(a3, a4):
            f = Fact("eqangle",
                     [Angle(A, B, I), Angle(I, B, C)],
                     "eqangle_and_eqangle_to_eqangle_bisector")
            f.add_parent(fact)
            f.add_parent(Fact("eqangle", [a3, a4]))
            facts.append(f)
            break
    return facts


def eqangle_and_eqline_to_eqratio(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """Bisector and eqline implies eqratio from area method."""
    facts = []
    a1, a2 = fact.objects
    if a1.name[1] != a2.name[1]:
        return facts
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if s1 == s4:
        s1, s2, s3, s4 = s2, s1, s4, s3
    if s2 != s3 or len(set([s1, s2, s3, s4])) != 3:
        return facts
    if on_same_line(*[diagram.point_dict[p] for p in a1.name]):
        return facts
    A, B, D, C = s1.p2, s1.p1, s2.p2, s4.p2
    s_AD = Segment(A, D)
    s_DC = Segment(D, C)
    s_AB = Segment(A, B)
    s_BC = Segment(B, C)
    if not diagram.database.is_eqline(s_AD, s_DC):
        return facts
    f = Fact("eqratio",
             [Ratio(s_AD, s_AB), Ratio(s_DC, s_BC)],
             "eqangle_and_eqline_to_eqratio")
    f.add_parent(fact)
    f.add_parent(Fact("eqline", [s_AD, s_DC]))
    facts.append(f)
    return facts


def eqline_and_eqangle_to_eqratio(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """Bisector and eqline implies eqratio from area method."""
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
    s_AD = Segment(A, D)
    s_DC = Segment(D, C)
    if A not in diagram.database.points_eqangles:
        return facts
    for s in diagram.database.points_eqangles[A]:
        B = s.p1 if s.p1 != A else s.p2
        if on_same_line(*[diagram.point_dict[p]
                          for p in [A, B, C]]) or on_same_line(
                              *[diagram.point_dict[p] for p in [A, B, D]]):
            continue
        if not diagram.database.is_eqangle(Angle(A, B, D), Angle(D, B, C)):
            continue
        s_AB = Segment(A, B)
        s_BC = Segment(B, C)
        f = Fact("eqratio", [
            Ratio(s_AD, s_AB),
            Ratio(s_DC, s_BC),
        ], "eqline_and_eqangle_to_eqratio")
        f.add_parent(fact)
        f.add_parent(Fact("eqangle", [Angle(A, B, D), Angle(D, B, C)]))
        facts.append(f)
    return facts


def eqratio_and_eqline_to_eqangle(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """Bisector and eqline implies eqratio from area method."""
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
    s1, s3 = r1.s1, r1.s2
    s2, s4 = r2.s1, r2.s2
    if s1 == s2 or s3 == s4:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    s34_joint, s34_p_set = four_joint(s3.p1, s3.p2, s4.p1, s4.p2)
    if not (s12_joint and s34_joint and s12_joint != s34_joint):
        return facts
    s12_p_set.pop(s12_joint)
    A, C = s12_p_set
    B = s12_joint
    s34_p_set.pop(s34_joint)
    D, F = s34_p_set
    E = s34_joint
    if A != D or C != F:
        return facts
    if on_same_line(*[diagram.point_dict[p]
                      for p in [A, B, C]]) == on_same_line(
                          *[diagram.point_dict[p] for p in [D, E, F]]):
        return facts
    if diagram.database.is_eqline(s1, s2):
        f = Fact("eqangle", [Angle(A, E, B), Angle(B, E, C)],
                 "eqratio_and_eqline_to_eqangle")
        f.add_parent(fact)
        f.add_parent(Fact("eqline", [s1, s2]))
        facts.append(f)
    elif diagram.database.is_eqline(s3, s4):
        f = Fact("eqangle", [Angle(A, B, E), Angle(E, B, C)],
                 "eqratio_and_eqline_to_eqangle")
        f.add_parent(fact)
        f.add_parent(Fact("eqline", [s3, s4]))
        facts.append(f)
    return facts


def eqline_and_eqratio_to_eqangle(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """Bisector and eqline implies eqratio from area method."""
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
    s_AD = Segment(A, D)
    s_DC = Segment(D, C)
    if s_AD not in diagram.database.segments_eqratios:
        return facts
    for r in diagram.database.segments_eqratios[s_AD]:
        s = r.s1 if r.s1 != s_AD else r.s2
        if s.p1 == A:
            B = s.p2
        elif s.p2 == A:
            B = s.p1
        else:
            continue
        if on_same_line(*[diagram.point_dict[p] for p in [A, B, C]]):
            continue
        s_AB = Segment(A, B)
        s_BC = Segment(B, C)
        if diagram.database.is_cong(s_AB, s_AD) or diagram.database.is_cong(
                s_BC, s_DC) or not diagram.database.is_eqratio(
                    Ratio(s_AB, s_AD), Ratio(s_BC, s_DC)):
            continue
        f = Fact("eqangle", [Angle(A, B, D), Angle(D, B, C)],
                 "eqline_and_eqratio_to_eqangle")
        f.add_parent(fact)
        f.add_parent(Fact("eqratio", [Ratio(s_AB, s_AD), Ratio(s_BC, s_DC)]))
        facts.append(f)
    return facts


def eqangle_and_perp_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Bisector and perp implies eqangle."""
    facts = []
    a1, a2 = fact.objects
    if a1.name[1] != a2.name[1]:
        return facts
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if s1 == s4:
        s1, s2, s3, s4 = s2, s1, s4, s3
    if s2 != s3 or len(set([s1, s2, s3, s4])) != 3:
        return facts
    if diagram.database.is_perp(a1) or diagram.database.is_perp(
            a2) or on_same_line(*[diagram.point_dict[p] for p in a1.name]):
        return facts
    A, B, D, C = s1.p2, s1.p1, s2.p2, s4.p2
    s_BD = Segment(B, D)
    if s_BD not in diagram.database.segments_perps:
        return facts
    for a in diagram.database.segments_perps[s_BD]:
        if a.name[1] != B:
            continue
        E = a.name[0] if a.name[0] != D else a.name[2]
        f = Fact("eqangle", [Angle(A, B, E), Angle(E, B, C)],
                 "eqangle_and_perp_to_eqangle")
        f.add_parent(fact)
        f.add_parent(Fact("perp", [a]))
        facts.append(f)
        break
    return facts


def perp_and_eqangle_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Bisector and perp implies eqangle."""
    facts = []
    a = fact.objects[0]
    H = a.name[1]
    s1, s2 = a.s1, a.s2
    for s, ss in [(s1, s2), (s2, s1)]:
        partition = {}
        if s not in diagram.database.segments_eqangles:
            continue
        for aa in diagram.database.segments_eqangles[s]:
            if aa.name[1] != H or diagram.database.is_perp(aa) or on_same_line(
                    *[diagram.point_dict[p] for p in aa.name]):
                continue
            rep = diagram.database.inverse_eqangle[aa]
            if rep in partition and partition[rep][-1]:
                continue
            if aa.s1 == s:
                side = 0
                p = aa.s2.p2
            else:
                side = 1
                p = aa.s1.p2
            if rep in partition:
                existing = partition[rep]
                if existing[1] != side:
                    pp = existing[2]
                    f = Fact("eqangle",
                             [Angle(p, H, ss.p2),
                              Angle(ss.p2, H, pp)],
                             "perp_and_eqangle_to_eqangle")
                    f.add_parent(fact)
                    f.add_parent(Fact("eqangle", [aa, existing[0]]))
                    facts.append(f)
                    existing[-1] = True
            else:
                partition[rep] = [aa, side, p, False]
    return facts


def eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad(
        diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Isogonal conjugate for quadrilater (from tri to quad)."""
    facts = []
    a1, a2 = fact.objects
    if a1.name[1] != a2.name[1]:
        return facts
    if len(set([a1.name[0], a1.name[1], a1.name[2], a2.name[0],
                a2.name[2]])) != 5:
        return facts
    if on_same_line(*[diagram.point_dict[p]
                      for p in a1.name]):  # A, P, D not collinear
        return facts
    A = a1.name[1]
    p_A = diagram.point_dict[A]
    for D, P, Q, B in [(a1.name[0], a1.name[2], a2.name[0], a2.name[2]),
                       (a1.name[2], a1.name[0], a2.name[2], a2.name[0])]:
        p_B, p_D, p_P = [diagram.point_dict[p] for p in [B, D, P]]
        if on_same_line(p_A, p_B, p_D):  # A, B, D not collinear
            continue
        if on_same_line(p_A, p_B, p_P):  # A, B, P not collinear
            continue
        if not on_same_line(p_B, p_D, p_P):
            # B, D, P not collinear
            if diagram.database.is_eqangle(Angle(A, B, P), Angle(Q, B, D)):
                f = Fact(
                    "eqangle", [Angle(B, D, P), Angle(Q, D, A)],
                    "eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [Angle(A, B, P), Angle(Q, B, D)]))
                facts.append(f)
            elif diagram.database.is_eqangle(Angle(B, D, P), Angle(Q, D, A)):
                f = Fact(
                    "eqangle", [Angle(A, B, P), Angle(Q, B, D)],
                    "eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [Angle(B, D, P), Angle(Q, D, A)]))
                facts.append(f)
        a = Angle(A, P, B)
        if a not in diagram.database.inverse_eqangle:
            continue
        rep = diagram.database.inverse_eqangle[a]
        rep_eqclass = diagram.database.eqangle[rep]
        rev = rep_eqclass[a].name != a.name
        for aa in rep_eqclass:
            if aa == a:
                continue
            if aa.name[1] != P or (not rev and aa.name[0]
                                   != D) or (rev and aa.name[2] != D):
                continue
            C = aa.name[0] if rev else aa.name[2]
            if C in [A, B, D, P, Q]:
                continue
            p_C = diagram.point_dict[C]
            # first check if a valid quad
            if on_same_line(p_A, p_D, p_C) or on_same_line(
                    p_A, p_B, p_C) or on_same_line(p_B, p_D, p_C):
                continue
            if intersect(p_A, p_B, p_C, p_D) or intersect(p_B, p_C, p_D, p_A):
                continue
            if on_same_line(p_B, p_C, p_P) or on_same_line(
                    p_C, p_D,
                    p_P):  # B, C, P not collinear, C, D, P not collinear
                continue
            if diagram.database.is_eqangle(Angle(A, B, P), Angle(Q, B, C)):
                f = Fact(
                    "eqangle", [Angle(B, C, P), Angle(Q, C, D)],
                    "eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [Angle(A, P, B), Angle(D, P, C)]))
                f.add_parent(Fact("eqangle", [Angle(A, B, P), Angle(Q, B, C)]))
                facts.append(f)
                f = Fact(
                    "eqangle", [Angle(C, D, P), Angle(Q, D, A)],
                    "eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [Angle(A, P, B), Angle(D, P, C)]))
                f.add_parent(Fact("eqangle", [Angle(A, B, P), Angle(Q, B, C)]))
                facts.append(f)
                f = Fact(
                    "eqangle", [Angle(A, Q, B), Angle(D, Q, C)],
                    "eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [Angle(A, P, B), Angle(D, P, C)]))
                f.add_parent(Fact("eqangle", [Angle(A, B, P), Angle(Q, B, C)]))
                facts.append(f)
            elif diagram.database.is_eqangle(Angle(C, D, P), Angle(Q, D, A)):
                f = Fact(
                    "eqangle", [Angle(B, C, P), Angle(Q, C, D)],
                    "eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [Angle(A, P, B), Angle(D, P, C)]))
                f.add_parent(Fact("eqangle", [Angle(C, D, P), Angle(Q, D, A)]))
                facts.append(f)
                f = Fact(
                    "eqangle", [Angle(A, B, P), Angle(Q, B, C)],
                    "eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [Angle(A, P, B), Angle(D, P, C)]))
                f.add_parent(Fact("eqangle", [Angle(C, D, P), Angle(Q, D, A)]))
                facts.append(f)
                f = Fact(
                    "eqangle", [Angle(A, Q, B), Angle(D, Q, C)],
                    "eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad")
                f.add_parent(fact)
                f.add_parent(Fact("eqangle", [Angle(A, P, B), Angle(D, P, C)]))
                f.add_parent(Fact("eqangle", [Angle(C, D, P), Angle(Q, D, A)]))
                facts.append(f)
    return facts


def eqangle_and_eqangle_and_eqangle_to_eqangle_isoquadtri(
        diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Isogonal conjugate for quadrilater (from quad to tri)."""
    facts = []
    a1, a2 = fact.objects
    if a1.name[1] != a2.name[1]:
        return facts
    if len(set([a1.name[0], a1.name[1], a1.name[2], a2.name[0],
                a2.name[2]])) != 5:
        return facts
    P = a1.name[1]
    A, B, D, C = a1.name[0], a1.name[2], a2.name[0], a2.name[2]
    p_A, p_B, p_C, p_D, p_P = [diagram.point_dict[p] for p in [A, B, C, D, P]]
    for check_quadruple in [(p_A, p_B, p_C, p_D), (p_B, p_C, p_A, p_D)]:
        if intersect(*check_quadruple):
            return facts
    for check_triple in [(p_A, p_B, p_C), (p_A, p_B, p_D), (p_A, p_C, p_D),
                         (p_B, p_C, p_D), (p_A, p_B, p_P), (p_A, p_C, p_P),
                         (p_A, p_D, p_P), (p_B, p_C, p_P), (p_B, p_D, p_P)
                         ]:  # P not in CD as P not in AB and APB=DPC
        if on_same_line(*check_triple):
            return facts
    checker_list = [(D, A), (A, B), (B, C), (C, D)]
    for idx in range(4):
        one_pts = checker_list[idx]
        two_pts = checker_list[(idx + 1) % 4]
        three_pts = checker_list[(idx + 2) % 4]
        four_pts = checker_list[(idx + 3) % 4]
        a = Angle(one_pts[0], one_pts[1], P)
        if a not in diagram.database.inverse_eqangle:
            continue
        rep = diagram.database.inverse_eqangle[a]
        rep_eqclass = diagram.database.eqangle[rep]
        rev = rep_eqclass[a].name != a.name
        for aa in rep_eqclass:
            if aa == a:
                continue
            if aa.name[1] != a.name[1] or (not rev and aa.name[2] != two_pts[1]
                                           ) or (rev
                                                 and aa.name[0] != two_pts[1]):
                continue
            Q = aa.name[0] if not rev else aa.name[2]
            if diagram.database.is_eqangle(
                    Angle(two_pts[0], two_pts[1], P),
                    Angle(Q, three_pts[0], three_pts[1])):
                f = Fact(
                    "eqangle", [
                        Angle(three_pts[0], three_pts[1], P),
                        Angle(Q, four_pts[0], four_pts[1])
                    ], "eqangle_and_eqangle_and_eqangle_to_eqangle_isoquadtri")
                f.add_parent(fact)
                f.add_parent(
                    Fact("eqangle", [
                        Angle(one_pts[0], one_pts[1], P),
                        Angle(Q, two_pts[0], two_pts[1])
                    ]))
                f.add_parent(
                    Fact("eqangle", [
                        Angle(two_pts[0], two_pts[1], P),
                        Angle(Q, three_pts[0], three_pts[1])
                    ]))
                facts.append(f)
                f = Fact(
                    "eqangle", [
                        Angle(four_pts[0], four_pts[1], P),
                        Angle(Q, one_pts[0], one_pts[1])
                    ], "eqangle_and_eqangle_and_eqangle_to_eqangle_isoquadtri")
                f.add_parent(fact)
                f.add_parent(
                    Fact("eqangle", [
                        Angle(one_pts[0], one_pts[1], P),
                        Angle(Q, two_pts[0], two_pts[1])
                    ]))
                f.add_parent(
                    Fact("eqangle", [
                        Angle(two_pts[0], two_pts[1], P),
                        Angle(Q, three_pts[0], three_pts[1])
                    ]))
                facts.append(f)
                f = Fact(
                    "eqangle", [Angle(A, Q, B), Angle(D, Q, C)],
                    "eqangle_and_eqangle_and_eqangle_to_eqangle_isoquadtri")
                f.add_parent(fact)
                f.add_parent(
                    Fact("eqangle", [
                        Angle(one_pts[0], one_pts[1], P),
                        Angle(Q, two_pts[0], two_pts[1])
                    ]))
                f.add_parent(
                    Fact("eqangle", [
                        Angle(two_pts[0], two_pts[1], P),
                        Angle(Q, three_pts[0], three_pts[1])
                    ]))
                facts.append(f)
    return facts
