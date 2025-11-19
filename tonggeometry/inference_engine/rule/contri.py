r"""Congruent-triangle-related rules.
    The following are the rules implemented:
    1. SSS
    2. SAS
    3. SSA for right triangles
"""

from itertools import product
from typing import TYPE_CHECKING, List

from tonggeometry.constructor.primitives import on_same_line, same_dir
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Segment, Triangle
from tonggeometry.inference_engine.util import four_joint

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def cong_and_cong_and_cong_to_contri(diagram: 'Diagram',
                                     fact: Fact) -> List[Fact]:
    """Two pairs of cong and a pair of cong or trivial."""
    facts = []
    s1, s4 = fact.objects
    partition = {}
    for p in [s1.p1, s1.p2]:
        for s2 in diagram.database.points_congs[p]:
            if s2 == s1 or on_same_line(
                    *[diagram.point_dict[p] for p in set(str(s1) + str(s2))]):
                continue
            rep = diagram.database.inverse_cong[s2]
            if rep in partition:
                partition[rep][0].append(s2)
            else:
                partition[rep] = [[s2], []]
    for p in [s4.p1, s4.p2]:
        for s5 in diagram.database.points_congs[p]:
            if s5 == s4 or on_same_line(
                    *[diagram.point_dict[p] for p in set(str(s4) + str(s5))]):
                continue
            rep = diagram.database.inverse_cong[s5]
            if rep in partition:
                partition[rep][1].append(s5)
    for s2_s5_pair in partition.values():
        for s2, s5 in product(s2_s5_pair[0], s2_s5_pair[1]):
            if s2 == s5 or s2 == s4 and s1 == s5:
                continue
            s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
            s12_p_set.pop(s12_joint)
            A, C = s12_p_set
            B = s12_joint
            s45_joint, s45_p_set = four_joint(s4.p1, s4.p2, s5.p1, s5.p2)
            s45_p_set.pop(s45_joint)
            D, F = s45_p_set
            E = s45_joint
            s3 = Segment(A, C)
            s6 = Segment(D, F)
            if not diagram.database.is_cong(s3, s6):
                continue
            t_ABC = Triangle(A, B, C)
            t_DEF = Triangle(D, E, F)
            f = Fact("contri", [t_ABC, t_DEF],
                     "cong_and_cong_and_cong_to_contri")
            f.add_parent(fact)
            f.add_parent(Fact("cong", [s2, s5]))
            if s3 != s6:
                f.add_parent(Fact("cong", [s3, s6]))
            facts.append(f)
    return facts


def cong_and_cong_and_eqangle_to_contri(diagram: 'Diagram',
                                        fact: Fact) -> List[Fact]:
    """Two pairs of cong and a pair of eqangle."""
    facts = []
    s1, s4 = fact.objects
    # when s1, s4 on the sides
    if (s1 in diagram.database.segments_eqangles  # pylint:disable=too-many-nested-blocks
            and s4 in diagram.database.segments_eqangles):
        partition = {}
        for a in diagram.database.segments_eqangles[s1]:
            if on_same_line(*[diagram.point_dict[p] for p in a.name]):
                continue
            rep = diagram.database.inverse_eqangle[a]
            if a.s1 == s1:
                idx = 0
                s2 = a.s2
            else:
                idx = 1
                s2 = a.s1
            if rep in partition:
                partition[rep][0].append((a, idx, s2))
            else:
                partition[rep] = [[(a, idx, s2)], []]
        for a in diagram.database.segments_eqangles[s4]:
            if on_same_line(*[diagram.point_dict[p] for p in a.name]):
                continue
            rep = diagram.database.inverse_eqangle[a]
            if a.s1 == s4:
                idx = 0
                s5 = a.s2
            else:
                idx = 1
                s5 = a.s1
            if rep in partition:
                partition[rep][1].append((a, idx, s5))
        for a_aa_pair in partition.values():
            for a_idx_s2, aa_idx_s5 in product(a_aa_pair[0], a_aa_pair[1]):
                a, idx_s1, s2 = a_idx_s2
                aa, idx_s4, s5 = aa_idx_s5
                s3 = Segment(a.name[0], a.name[2])
                s6 = Segment(aa.name[0], aa.name[2])
                dir_val = same_dir(
                    *[diagram.point_dict[p] for p in a.name + aa.name])
                # SAS
                if dir_val == 1 and diagram.database.is_cong(s2, s5):
                    if idx_s1 == idx_s4:
                        f = Fact("contri",
                                 [Triangle(*a.name),
                                  Triangle(*aa.name)],
                                 "cong_and_cong_and_eqangle_to_contri")
                        f.add_parent(fact)
                        if s2 != s5:
                            f.add_parent(Fact("cong", [s2, s5]))
                        f.add_parent(Fact("eqangle", [a, aa]))
                        facts.append(f)
                    else:
                        f = Fact("contri",
                                 [Triangle(*a.name),
                                  Triangle(*aa.name[::-1])],
                                 "cong_and_cong_and_eqangle_to_contri")
                        f.add_parent(fact)
                        if s2 != s5:
                            f.add_parent(Fact("cong", [s2, s5]))
                        f.add_parent(Fact("eqangle", [a, aa]))
                        facts.append(f)
                # SSA for right triangles
                common_parents = []
                if diagram.database.is_perp(a) and diagram.database.is_perp(
                        aa):
                    common_parents.append(Fact("perp", [a]))
                    common_parents.append(Fact("perp", [aa]))
                elif diagram.database.is_perp(a):
                    common_parents.append(Fact("perp", [a]))
                    common_parents.append(Fact("eqangle", [a, aa]))
                elif diagram.database.is_perp(aa):
                    common_parents.append(Fact("perp", [aa]))
                    common_parents.append(Fact("eqangle", [a, aa]))
                if len(common_parents) > 0:
                    if diagram.database.is_cong(s2, s5):
                        if idx_s1 == idx_s4:
                            f = Fact("contri",
                                     [Triangle(*a.name),
                                      Triangle(*aa.name)],
                                     "cong_and_cong_and_eqangle_to_contri")
                            f.add_parent(fact)
                            f.add_parents(common_parents)
                            if s2 != s5:
                                f.add_parent(Fact("cong", [s2, s5]))
                            facts.append(f)
                        else:
                            f = Fact(
                                "contri",
                                [Triangle(*a.name),
                                 Triangle(*aa.name[::-1])],
                                "cong_and_cong_and_eqangle_to_contri")
                            f.add_parent(fact)
                            f.add_parents(common_parents)
                            if s2 != s5:
                                f.add_parent(Fact("cong", [s2, s5]))
                            facts.append(f)
                    if diagram.database.is_cong(s3, s6):
                        if idx_s1 == idx_s4:
                            f = Fact("contri",
                                     [Triangle(*a.name),
                                      Triangle(*aa.name)],
                                     "cong_and_cong_and_eqangle_to_contri")
                            f.add_parent(fact)
                            f.add_parents(common_parents)
                            if s3 != s6:
                                f.add_parent(Fact("cong", [s3, s6]))
                            facts.append(f)
                        else:
                            f = Fact(
                                "contri",
                                [Triangle(*a.name),
                                 Triangle(*aa.name[::-1])],
                                "cong_and_cong_and_eqangle_to_contri")
                            f.add_parent(fact)
                            f.add_parents(common_parents)
                            if s3 != s6:
                                f.add_parent(Fact("cong", [s3, s6]))
                            facts.append(f)
    # when s1 s4 on the hypotenuse
    if (s1 in diagram.database.h_segments_perps
            or s4 in diagram.database.h_segments_perps):
        angles_parents = {}
        if s1 in diagram.database.h_segments_perps:
            for a in diagram.database.h_segments_perps[s1]:
                if a not in diagram.database.inverse_eqangle:
                    continue
                eqclass_rep = diagram.database.inverse_eqangle[a]
                eqclass = diagram.database.eqangle[eqclass_rep]
                for aa in eqclass:
                    if Segment(aa.name[0], aa.name[2]) != s4:
                        continue
                    key = (str(a), str(aa))
                    if key in angles_parents:
                        continue
                    common_parents = [Fact("perp", [a])]
                    if diagram.database.is_perp(aa):
                        common_parents.append(Fact("perp", [aa]))
                    else:
                        common_parents.append(Fact("eqangle", [a, aa]))
                    angles_parents[key] = (a, aa, common_parents)
        elif s4 in diagram.database.h_segments_perps:
            for aa in diagram.database.h_segments_perps[s4]:
                if aa not in diagram.database.inverse_eqangle:
                    continue
                eqclass_rep = diagram.database.inverse_eqangle[aa]
                eqclass = diagram.database.eqangle[eqclass_rep]
                for a in eqclass:
                    if Segment(a.name[0], a.name[2]) != s1:
                        continue
                    key = (str(a), str(aa))
                    if key in angles_parents:
                        continue
                    common_parents = [Fact("perp", [aa])]
                    if diagram.database.is_perp(a):
                        common_parents.append(Fact("perp", [a]))
                    else:
                        common_parents.append(Fact("eqangle", [a, aa]))
                    angles_parents[key] = (a, aa, common_parents)
        for a, aa, common_parents in angles_parents.values():
            if diagram.database.is_cong(a.s1, aa.s1):
                f = Fact(
                    "contri",
                    [Triangle(*a.name), Triangle(*aa.name)],
                    "cong_and_cong_and_eqangle_to_contri")
                f.add_parent(fact)
                f.add_parents(common_parents)
                if a.s1 != aa.s1:
                    f.add_parent(Fact("cong", [a.s1, aa.s1]))
                facts.append(f)
            elif diagram.database.is_cong(a.s2, aa.s2):
                f = Fact(
                    "contri",
                    [Triangle(*a.name), Triangle(*aa.name)],
                    "cong_and_cong_and_eqangle_to_contri")
                f.add_parent(fact)
                f.add_parents(common_parents)
                if a.s2 != aa.s2:
                    f.add_parent(Fact("cong", [a.s2, aa.s2]))
                facts.append(f)
            elif diagram.database.is_cong(a.s1, aa.s2):
                f = Fact("contri",
                         [Triangle(*a.name),
                          Triangle(*aa.name[::-1])],
                         "cong_and_cong_and_eqangle_to_contri")
                f.add_parent(fact)
                f.add_parents(common_parents)
                if a.s1 != aa.s2:
                    f.add_parent(Fact("cong", [a.s1, aa.s2]))
                facts.append(f)
            elif diagram.database.is_cong(a.s2, aa.s1):
                f = Fact("contri",
                         [Triangle(*a.name),
                          Triangle(*aa.name[::-1])],
                         "cong_and_cong_and_eqangle_to_contri")
                f.add_parent(fact)
                f.add_parents(common_parents)
                if a.s2 != aa.s1:
                    f.add_parent(Fact("cong", [a.s2, aa.s1]))
                facts.append(f)
    return facts


def eqangle_and_cong_and_cong_to_contri(diagram: 'Diagram',
                                        fact: Fact) -> List[Fact]:
    """Two pairs of cong and a pair of eqangle."""
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
    if diagram.database.is_cong(s1, s4) and diagram.database.is_cong(s2, s5):
        f = Fact("contri",
                 [Triangle(*a1.name), Triangle(*a2.name)],
                 "eqangle_and_cong_and_cong_to_contri")
        f.add_parent(fact)
        if s1 != s4:
            f.add_parent(Fact("cong", [s1, s4]))
        if s2 != s5:
            f.add_parent(Fact("cong", [s2, s5]))
        facts.append(f)
    elif diagram.database.is_cong(s1, s5) and diagram.database.is_cong(s2, s4):
        f = Fact(
            "contri",
            [Triangle(*a1.name), Triangle(*a2.name[::-1])],
            "eqangle_and_cong_and_cong_to_contri")
        f.add_parent(fact)
        if s1 != s5:
            f.add_parent(Fact("cong", [s1, s5]))
        if s2 != s4:
            f.add_parent(Fact("cong", [s2, s4]))
        facts.append(f)
    # SSA for right triangles
    if diagram.database.is_perp(a1) or diagram.database.is_perp(a2):
        s3 = Segment(a1.name[0], a1.name[2])
        s6 = Segment(a2.name[0], a2.name[2])
        common_parents = []
        if diagram.database.is_perp(a1) and diagram.database.is_perp(a2):
            common_parents.append(Fact("perp", [a1]))
            common_parents.append(Fact("perp", [a2]))
        elif diagram.database.is_perp(a1):
            common_parents.append(Fact("perp", [a1]))
            common_parents.append(fact)
        else:
            common_parents.append(Fact("perp", [a2]))
            common_parents.append(fact)
        if diagram.database.is_cong(s3, s6):
            if diagram.database.is_cong(s1, s4):
                f = Fact("contri", [Triangle(*a1.name),
                                    Triangle(*a2.name)],
                         "eqangle_and_cong_and_cong_to_contri")
                if s1 != s4:
                    f.add_parent(Fact("cong", [s1, s4]))
                if s3 != s6:
                    f.add_parent(Fact("cong", [s3, s6]))
                f.add_parents(common_parents)
                facts.append(f)
            elif diagram.database.is_cong(s2, s5):
                f = Fact("contri", [Triangle(*a1.name),
                                    Triangle(*a2.name)],
                         "eqangle_and_cong_and_cong_to_contri")
                if s2 != s5:
                    f.add_parent(Fact("cong", [s2, s5]))
                if s3 != s6:
                    f.add_parent(Fact("cong", [s3, s6]))
                f.add_parents(common_parents)
                facts.append(f)
            elif diagram.database.is_cong(s1, s5):
                f = Fact("contri",
                         [Triangle(*a1.name),
                          Triangle(*a2.name[::-1])],
                         "eqangle_and_cong_and_cong_to_contri")
                if s1 != s5:
                    f.add_parent(Fact("cong", [s1, s5]))
                if s3 != s6:
                    f.add_parent(Fact("cong", [s3, s6]))
                f.add_parents(common_parents)
                facts.append(f)
            elif diagram.database.is_cong(s2, s4):
                f = Fact("contri",
                         [Triangle(*a1.name),
                          Triangle(*a2.name[::-1])],
                         "eqangle_and_cong_and_cong_to_contri")
                if s2 != s4:
                    f.add_parent(Fact("cong", [s2, s4]))
                if s3 != s6:
                    f.add_parent(Fact("cong", [s3, s6]))
                f.add_parents(common_parents)
                facts.append(f)
    return facts


def contri_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Contri three sides cong"""
    facts = []
    t1, t2 = fact.objects
    if on_same_line(*[diagram.point_dict[p] for p in t1.name]):
        return facts
    A, B, C = t1.name
    D, E, F = t2.name
    f = Fact("cong", [Segment(A, B), Segment(D, E)], "contri_to_cong")
    f.add_parent(fact)
    facts.append(f)
    f = Fact("cong", [Segment(B, C), Segment(E, F)], "contri_to_cong")
    f.add_parent(fact)
    facts.append(f)
    f = Fact("cong", [Segment(A, C), Segment(D, F)], "contri_to_cong")
    f.add_parent(fact)
    facts.append(f)
    return facts


def contri_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Contri three angles equal."""
    facts = []
    t1, t2 = fact.objects
    if on_same_line(*[diagram.point_dict[p] for p in t1.name]):
        return facts
    A, B, C = t1.name
    D, E, F = t2.name
    dir_val = same_dir(*[diagram.point_dict[p] for p in [A, B, C, D, E, F]])
    mirror = dir_val == -1
    a_DEF = Angle(D, E, F)
    a_EFD = Angle(E, F, D)
    a_FDE = Angle(F, D, E)
    if mirror:
        a_DEF = Angle(*a_DEF.name[::-1])
        a_EFD = Angle(*a_EFD.name[::-1])
        a_FDE = Angle(*a_FDE.name[::-1])
    f = Fact("eqangle", [Angle(A, B, C), a_DEF], "contri_to_eqangle")
    f.add_parent(fact)
    facts.append(f)
    f = Fact("eqangle", [Angle(B, C, A), a_EFD], "contri_to_eqangle")
    f.add_parent(fact)
    facts.append(f)
    f = Fact("eqangle", [Angle(C, A, B), a_FDE], "contri_to_eqangle")
    f.add_parent(fact)
    facts.append(f)
    return facts
