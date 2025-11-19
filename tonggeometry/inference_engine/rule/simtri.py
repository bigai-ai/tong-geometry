r"""Similar-triangle-related rules.
    The following are the rules implemented:
    1. SSS
    2. AAA
    3. SAS
    4. SSA for right triangles
"""

from typing import TYPE_CHECKING, List

from tonggeometry.constructor.primitives import on_same_line, same_dir
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import (Angle, Ratio, Segment,
                                                      Triangle)
from tonggeometry.inference_engine.util import four_joint

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def eqratio_and_eqratio_to_simtri(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """Simtri from eqratio."""
    facts = []
    r1, r2 = fact.objects
    if ((diagram.database.is_cong(r1.s1, r2.s2)
         and diagram.database.is_cong(r1.s2, r2.s1))
            or diagram.database.is_cong(r1.s1, r1.s2)
            or diagram.database.is_cong(r2.s1, r2.s2)):
        return facts
    eqclass_rep = diagram.database.inverse_eqratio[r1]
    eqclass = diagram.database.eqratio[eqclass_rep]
    dir_r1 = eqclass[r1].s1 == r1.s1
    dir_r2 = eqclass[r2].s1 == r2.s1
    if dir_r1 != dir_r2:
        return facts
    s1, s4 = r1.s1, r1.s2
    s2, s5 = r2.s1, r2.s2
    if s1 == s2 or s4 == s5:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    s45_joint, s45_p_set = four_joint(s4.p1, s4.p2, s5.p1, s5.p2)
    if not (s12_joint and s45_joint):
        return facts
    s12_p_set.pop(s12_joint)
    B = s12_joint
    A, C = s12_p_set
    s45_p_set.pop(s45_joint)
    E = s45_joint
    D, F = s45_p_set
    if on_same_line(*[diagram.point_dict[p]
                      for p in [A, B, C]]) or on_same_line(
                          *[diagram.point_dict[p] for p in [D, E, F]]):
        return facts
    s3 = Segment(A, C)
    s6 = Segment(D, F)
    r3 = Ratio(s3, s6)
    if diagram.database.is_cong(s3, s6):
        return facts
    for rr in [r1, r2]:
        if diagram.database.is_eqratio(rr, r3):
            f = Fact("simtri",
                     [Triangle(A, B, C), Triangle(D, E, F)],
                     "eqratio_and_eqratio_to_simtri")
            f.add_parent(fact)
            f.add_parent(Fact("eqratio", [rr, r3]))
            facts.append(f)
            break
    return facts


def eqangle_and_eqangle_and_eqangle_to_simtri(diagram: 'Diagram',
                                              fact: Fact) -> List[Fact]:
    """Simtri from two eqangles."""
    facts = []
    a1, a2 = fact.objects
    if sorted(a1.name) == sorted(a2.name):
        return facts
    dir_val = same_dir(*[diagram.point_dict[p] for p in a1.name + a2.name])
    if dir_val != 1:
        return facts
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if (len(set([s1, s2, s3, s4])) < 3
            or on_same_line(*[diagram.point_dict[p] for p in a1.name])):
        return facts
    A, B, C = a1.name
    D, E, F = a2.name
    a_A = Angle(B, A, C)
    a_C = Angle(A, C, B)
    a_D = Angle(E, D, F)
    a_F = Angle(D, F, E)
    if diagram.database.is_eqangle(a_A, a_D):
        f = Fact("simtri",
                 [Triangle(A, B, C), Triangle(D, E, F)],
                 "eqangle_and_eqangle_and_eqangle_to_simtri")
        f.add_parent(fact)
        f.add_parent(Fact("eqangle", [a_A, a_D]))
        facts.append(f)
    elif diagram.database.is_eqangle(a_C, a_F):
        f = Fact("simtri",
                 [Triangle(A, B, C), Triangle(D, E, F)],
                 "eqangle_and_eqangle_and_eqangle_to_simtri")
        f.add_parent(fact)
        f.add_parent(Fact("eqangle", [a_C, a_F]))
        facts.append(f)
    elif diagram.database.is_eqangle(a_A, a_F):
        f = Fact("simtri",
                 [Triangle(A, B, C), Triangle(F, E, D)],
                 "eqangle_and_eqangle_and_eqangle_to_simtri")
        f.add_parent(fact)
        f.add_parent(Fact("eqangle", [a_A, a_F]))
        facts.append(f)
    elif diagram.database.is_eqangle(a_C, a_D):
        f = Fact("simtri",
                 [Triangle(A, B, C), Triangle(F, E, D)],
                 "eqangle_and_eqangle_and_eqangle_to_simtri")
        f.add_parent(fact)
        f.add_parent(Fact("eqangle", [a_C, a_D]))
        facts.append(f)
    return facts


def eqratio_and_eqangle_to_simtri(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """Simtri from eqratio and eqangle."""
    facts = []
    r1, r2 = fact.objects
    if ((diagram.database.is_cong(r1.s1, r2.s2)
         and diagram.database.is_cong(r1.s2, r2.s1))
            or diagram.database.is_cong(r1.s1, r1.s2)
            or diagram.database.is_cong(r2.s1, r2.s2)):
        return facts
    eqclass_rep = diagram.database.inverse_eqratio[r1]
    eqclass = diagram.database.eqratio[eqclass_rep]
    dir_r1 = eqclass[r1].s1 == r1.s1
    dir_r2 = eqclass[r2].s1 == r2.s1
    if dir_r1 != dir_r2:
        return facts
    s1, s4 = r1.s1, r1.s2
    s2, s5 = r2.s1, r2.s2
    if s1 == s2 or s4 == s5:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    s45_joint, s45_p_set = four_joint(s4.p1, s4.p2, s5.p1, s5.p2)
    if not (s12_joint and s45_joint):
        return facts
    s12_p_set.pop(s12_joint)
    B = s12_joint
    A, C = s12_p_set
    s45_p_set.pop(s45_joint)
    E = s45_joint
    D, F = s45_p_set
    if on_same_line(*[diagram.point_dict[p]
                      for p in [A, B, C]]) or on_same_line(
                          *[diagram.point_dict[p] for p in [D, E, F]]):
        return facts
    dir_val = same_dir(*[diagram.point_dict[p] for p in [A, B, C, D, E, F]])
    a_B = Angle(A, B, C)
    a_E = Angle(D, E, F)
    a_E_p = Angle(*a_E.name[::-1])
    a_A = Angle(B, A, C)
    a_C = Angle(A, C, B)
    a_D = Angle(E, D, F)
    a_F = Angle(D, F, E)
    # SAS
    if dir_val == 1 and diagram.database.is_eqangle(a_B, a_E):
        f = Fact("simtri",
                 [Triangle(A, B, C), Triangle(D, E, F)],
                 "eqratio_and_eqangle_to_simtri")
        f.add_parent(fact)
        f.add_parent(Fact("eqangle", [a_B, a_E]))
        facts.append(f)
    if dir_val == -1 and diagram.database.is_eqangle(a_B, a_E_p):
        f = Fact("simtri",
                 [Triangle(A, B, C), Triangle(D, E, F)],
                 "eqratio_and_eqangle_to_simtri")
        f.add_parent(fact)
        f.add_parent(Fact("eqangle", [a_B, a_E_p]))
        facts.append(f)
    # SSA for right triangles
    conds = [(a_A, a_D), (a_B, a_E), (a_C, a_F)]
    for cond in conds:
        a1, a2 = cond
        if diagram.database.is_perp(a1) or diagram.database.is_perp(a2):
            f = Fact("simtri",
                     [Triangle(A, B, C), Triangle(D, E, F)],
                     "eqratio_and_eqangle_to_simtri")
            f.add_parent(fact)
            if diagram.database.is_perp(a1) and diagram.database.is_perp(a2):
                f.add_parent(Fact("perp", [a1]))
                f.add_parent(Fact("perp", [a2]))
                facts.append(f)
            elif diagram.database.is_perp(a1) and diagram.database.is_eqangle(
                    a1, a2):
                f.add_parent(Fact("perp", [a1]))
                f.add_parent(Fact("eqangle", [a1, a2]))
                facts.append(f)
            elif diagram.database.is_perp(a1) and diagram.database.is_eqangle(
                    a1, Angle(*a2.name[::-1])):
                f.add_parent(Fact("perp", [a1]))
                f.add_parent(Fact("eqangle", [a1, Angle(*a2.name[::-1])]))
                facts.append(f)
            elif diagram.database.is_perp(a2) and diagram.database.is_eqangle(
                    a1, a2):
                f.add_parent(Fact("perp", [a2]))
                f.add_parent(Fact("eqangle", [a1, a2]))
                facts.append(f)
            elif diagram.database.is_perp(a2) and diagram.database.is_eqangle(
                    a1, Angle(*a2.name[::-1])):
                f.add_parent(Fact("perp", [a2]))
                f.add_parent(Fact("eqangle", [a1, Angle(*a2.name[::-1])]))
                facts.append(f)
            break
    return facts


def eqangle_and_eqratio_to_simtri(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """Simtri from eqratio and eqangle."""
    facts = []
    a1, a2 = fact.objects
    if sorted(a1.name) == sorted(a2.name):
        return facts
    dir_val = same_dir(*[diagram.point_dict[p] for p in a1.name + a2.name])
    if dir_val != 1:
        return facts
    s1, s2 = a1.s1, a1.s2
    s4, s5 = a2.s1, a2.s2
    if len(set([s1, s2, s4, s5])) < 3 or on_same_line(
            *[diagram.point_dict[p] for p in a1.name]):
        return facts
    # SAS
    if diagram.database.is_eqratio(Ratio(s1, s4), Ratio(s2, s5)):
        if not diagram.database.is_cong(
                s1, s4) and not diagram.database.is_cong(s2, s5):
            f = Fact(
                "simtri",
                [Triangle(*a1.name), Triangle(*a2.name)],
                "eqangle_and_eqratio_to_simtri")
            f.add_parent(fact)
            f.add_parent(Fact("eqratio", [Ratio(s1, s4), Ratio(s2, s5)]))
            facts.append(f)
    elif diagram.database.is_eqratio(Ratio(s1, s5), Ratio(s2, s4)):
        if not diagram.database.is_cong(
                s1, s5) and not diagram.database.is_cong(s2, s4):
            f = Fact("simtri", [Triangle(*a1.name),
                                Triangle(*a2.name[::-1])],
                     "eqangle_and_eqratio_to_simtri")
            f.add_parent(fact)
            f.add_parent(Fact("eqratio", [Ratio(s1, s5), Ratio(s2, s4)]))
            facts.append(f)
    # SSA for right triangles
    if diagram.database.is_perp(a1) or diagram.database.is_perp(a2):
        s3 = Segment(a1.name[0], a1.name[2])
        s6 = Segment(a2.name[0], a2.name[2])
        common_parents = []
        if diagram.database.is_perp(a1) and diagram.database.is_perp(a2):
            common_parents.append(Fact("perp", [a1]))
            common_parents.append(Fact("perp", [a2]))
            common_parents.append(fact)
        elif diagram.database.is_perp(a1):
            common_parents.append(Fact("perp", [a1]))
            common_parents.append(fact)
        else:
            common_parents.append(Fact("perp", [a2]))
            common_parents.append(fact)
        if not diagram.database.is_cong(s3, s6):
            if diagram.database.is_eqratio(Ratio(s3, s6), Ratio(
                    s1, s4)) and not diagram.database.is_cong(s1, s4):
                f = Fact("simtri", [Triangle(*a1.name),
                                    Triangle(*a2.name)],
                         "eqangle_and_eqratio_to_simtri")
                f.add_parent(Fact("eqratio", [Ratio(s3, s6), Ratio(s1, s4)]))
                f.add_parents(common_parents)
                facts.append(f)
            elif diagram.database.is_eqratio(Ratio(s3, s6), Ratio(
                    s2, s5)) and not diagram.database.is_cong(s2, s5):
                f = Fact("simtri", [Triangle(*a1.name),
                                    Triangle(*a2.name)],
                         "eqangle_and_eqratio_to_simtri")
                f.add_parent(Fact("eqratio", [Ratio(s3, s6), Ratio(s2, s5)]))
                f.add_parents(common_parents)
                facts.append(f)
            elif diagram.database.is_eqratio(Ratio(s3, s6), Ratio(
                    s1, s5)) and not diagram.database.is_cong(s1, s5):
                f = Fact("simtri",
                         [Triangle(*a1.name),
                          Triangle(*a2.name[::-1])],
                         "eqangle_and_eqratio_to_simtri")
                f.add_parent(Fact("eqratio", [Ratio(s3, s6), Ratio(s1, s5)]))
                f.add_parents(common_parents)
                facts.append(f)
            elif diagram.database.is_eqratio(Ratio(s3, s6), Ratio(
                    s2, s4)) and not diagram.database.is_cong(s2, s4):
                f = Fact("simtri",
                         [Triangle(*a1.name),
                          Triangle(*a2.name[::-1])],
                         "eqangle_and_eqratio_to_simtri")
                f.add_parent(Fact("eqratio", [Ratio(s3, s6), Ratio(s2, s4)]))
                f.add_parents(common_parents)
                facts.append(f)
    return facts


def simtri_to_eqratio(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Simtri implies eqratio."""
    facts = []
    t1, t2 = fact.objects
    if on_same_line(*[diagram.point_dict[p] for p in t1.name]):
        return facts
    A, B, C = t1.name
    D, E, F = t2.name
    s_AB = Segment(A, B)
    s_AC = Segment(A, C)
    s_BC = Segment(B, C)
    s_DE = Segment(D, E)
    s_DF = Segment(D, F)
    s_EF = Segment(E, F)
    f = Fact("eqratio",
             [Ratio(s_AB, s_DE), Ratio(s_AC, s_DF)], "simtri_to_eqratio")
    f.add_parent(fact)
    facts.append(f)
    f = Fact("eqratio",
             [Ratio(s_AB, s_DE), Ratio(s_BC, s_EF)], "simtri_to_eqratio")
    f.add_parent(fact)
    facts.append(f)
    f = Fact("eqratio",
             [Ratio(s_AC, s_DF), Ratio(s_BC, s_EF)], "simtri_to_eqratio")
    f.add_parent(fact)
    facts.append(f)
    return facts


def simtri_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Simtri implies eqangle."""
    facts = []
    t1, t2 = fact.objects
    A, B, C = t1.name
    D, E, F = t2.name
    if on_same_line(*[diagram.point_dict[p] for p in t1.name]):
        return facts
    dir_val = same_dir(*[diagram.point_dict[p] for p in [A, B, C, D, E, F]])
    mirror = dir_val == -1
    a_DEF = Angle(D, E, F)
    a_EFD = Angle(E, F, D)
    a_FDE = Angle(F, D, E)
    if mirror:
        a_DEF = Angle(*a_DEF.name[::-1])
        a_EFD = Angle(*a_EFD.name[::-1])
        a_FDE = Angle(*a_FDE.name[::-1])
    f = Fact("eqangle", [Angle(A, B, C), a_DEF], "simtri_to_eqangle")
    f.add_parent(fact)
    facts.append(f)
    f = Fact("eqangle", [Angle(B, C, A), a_EFD], "simtri_to_eqangle")
    f.add_parent(fact)
    facts.append(f)
    f = Fact("eqangle", [Angle(C, A, B), a_FDE], "simtri_to_eqangle")
    f.add_parent(fact)
    facts.append(f)
    return facts


def simtri_and_cong_to_contri(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Simtri with cong become contri."""
    facts = []
    t1, t2 = fact.objects
    if on_same_line(*[diagram.point_dict[p] for p in t1.name]):
        return facts
    A, B, C = t1.name
    D, E, F = t2.name
    for s, ss in zip(
        [Segment(A, B), Segment(A, C),
         Segment(B, C)], [Segment(
             D, E), Segment(D, F), Segment(E, F)]):
        if diagram.database.is_cong(s, ss):
            f = Fact("contri", [t1, t2], "simtri_and_cong_to_contri")
            f.add_parent(fact)
            if s != ss:
                f.add_parent(Fact("cong", [s, ss]))
            facts.append(f)
            break
    return facts


def cong_and_simtri_to_contri(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Simtri with cong become contri."""
    facts = []
    s1, s2 = fact.objects
    if (s1 not in diagram.database.segments_simtris
            or s2 not in diagram.database.segments_simtris):
        return facts
    for t in diagram.database.segments_simtris[s1]:
        indices_t = set([t.name.index(s1.p1), t.name.index(s1.p2)])
        eqclass_rep = diagram.database.inverse_simtri[t]
        eqclass = diagram.database.simtri[eqclass_rep]
        for tt in eqclass:
            if tt == t or tt not in diagram.database.segments_simtris[s2]:
                continue
            indices_tt = set([tt.name.index(s2.p1), tt.name.index(s2.p2)])
            if indices_t != indices_tt:
                continue
            f = Fact("contri", [t, tt], "cong_and_simtri_to_contri")
            f.add_parent(fact)
            f.add_parent(Fact("simtri", [t, tt]))
            facts.append(f)
    return facts
