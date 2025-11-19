r"""Eqline-related rules."""

from itertools import combinations, permutations, product
from typing import TYPE_CHECKING, List, Tuple

from tonggeometry.constructor.primitives import angle_type, on_same_line
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Ratio, Segment
from tonggeometry.inference_engine.util import four_joint, sort_two_from_first
from tonggeometry.util import OrderedSet, isclose

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def eqline_to_eqline(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Generate all sublines in eqline."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    p1, p2 = s1.p1, s1.p2
    p3, p4 = s2.p1, s2.p2
    s = OrderedSet().fromkeys([p1, p2, p3, p4])
    segments_p = combinations(list(s.keys()), 2)
    for ss1_p, ss2_p in combinations(segments_p, 2):
        ss1 = Segment(*ss1_p)
        ss2 = Segment(*ss2_p)
        f = Fact("eqline", [ss1, ss2], "eqline_to_eqline")
        if f == fact:
            continue
        f.add_parent(fact)
        facts.append(f)
    return facts


def eqline_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Trivial eqangles from eqline."""
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
    for p in diagram.point_dict:
        if on_same_line(*[diagram.point_dict[pp] for pp in [p, B, A]]):
            continue
        f = Fact("eqangle", [Angle(p, B, A), Angle(p, B, C)],
                 "eqline_to_eqangle")
        f.add_parent(fact)
        facts.append(f)
    return facts


def eqline_and_eqline_to_x(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """X shape for Pappus, Desargues, and Cevian bundle for harmonic."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return []
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    p = s12_joint
    A, b = s12_p_set
    l_Ab = diagram.database.inverse_eqline[Segment(A, b)]
    for l in diagram.database.points_lines[p]:
        if l == l_Ab or len(diagram.database.lines_points[l]) < 3:
            continue
        B = l.p1 if l.p1 != p else l.p2
        if B in [A, b] or on_same_line(
                *[diagram.point_dict[pp] for pp in [A, b, B]], eps=1e-4):
            continue
        for B, a in permutations(diagram.database.lines_points[l], 2):
            if p in [B, a]:
                continue
            if B in [A, b] or a in [A, b]:
                diagram.terminal_flag = True
                diagram.inf_terminal = True
                return []
            str_x = min("".join(k)
                        for k in [(A, B, b, a), (B, A, a, b), (a, b, B,
                                                               A), (b, a, A,
                                                                    B)])
            key_x = str_x
            x = key_x + p
            if key_x not in diagram.database.inverse_x:
                # For Pappus
                facts += x_and_x_to_pappus(diagram, x)
                # For Desargues
                facts += x_and_x_and_x_to_desargues(diagram, x)
                key_Aa = Segment(x[0], x[3])
                key_Bb = Segment(x[1], x[2])
                if key_Aa not in diagram.database.x:
                    diagram.database.x[key_Aa] = {}
                if key_Bb not in diagram.database.x:
                    diagram.database.x[key_Bb] = {}
                diagram.database.x[key_Aa][x] = None
                diagram.database.x[key_Bb][x] = None
                diagram.database.inverse_x[key_x] = p
            # For Cevian
            if str_x in diagram.database.inverse_cevian:
                for cevian, vals in diagram.database.inverse_cevian[
                        str_x].items():
                    q, tp1, tp2 = vals
                    parents = diagram.database.cevian[cevian]
                    f = Fact("eqratio", [
                        Ratio(Segment(tp1, q), Segment(tp2, q)),
                        Ratio(Segment(tp1, p), Segment(tp2, p))
                    ], "eqline_and_eqline_to_x")
                    f.add_parent(Fact(
                        "eqline", [Segment(A, p), Segment(b, p)]))
                    f.add_parent(Fact(
                        "eqline", [Segment(B, p), Segment(a, p)]))
                    f.add_parents(parents)
                    facts.append(f)
            # For Cross Ratio
            # x = "".join(x)
            # if x not in diagram.database.inverse_star:
            #     key_star = p + x[0] + x[2]
            #     facts += star_and_sun_and_eqline_or_eqcircle_to_net(
            #         diagram, x)
            #     facts += star_and_star_and_eqline_or_eqcircle_to_sun(
            #         diagram, x)
            #     if key_star not in diagram.database.star:
            #         diagram.database.star[key_star] = {}
            #     diagram.database.star[key_star][x] = None
            #     diagram.database.inverse_star[x] = None
    return facts


def x_and_x_to_pappus(diagram: 'Diagram', x: Tuple) -> List[Fact]:
    """The Pappus theorem. For Pappus, any combination of ABC and abc works, as
    long as intersections are (Ab, aB), (Ac, aC), (Bc, bC).

    Keys are normalized, the smallest among ABba, BAab, abBA, baAB is used.
    """
    facts = []
    A, B, b, a, p = x
    key_Aa = Segment(A, a)
    key_Bb = Segment(B, b)
    for key, key_other in [(key_Aa, key_Bb), (key_Bb, key_Aa)]:
        if key not in diagram.database.x:
            continue
        for xx in diagram.database.x[key]:
            xx_key_Aa = Segment(xx[0], xx[3])
            xx_key_Bb = Segment(xx[1], xx[2])
            pp = xx[4]
            if xx_key_Aa == key:
                xx_key = xx_key_Aa
                Cc = xx_key_Bb
            else:
                xx_key = xx_key_Bb
                Cc = xx_key_Aa
            if (key.p1, key.p2) == (xx_key.p1, xx_key.p2):
                C, c = Cc.p1, Cc.p2
            else:
                C, c = Cc.p2, Cc.p1
            O, o = key_other.p1, key_other.p2
            parents = [
                Fact("eqline", [Segment(A, p), Segment(b, p)]),
                Fact("eqline", [Segment(B, p), Segment(a, p)]),
                Fact("eqline", [Segment(xx[0], pp),
                                Segment(xx[2], pp)]),
                Fact("eqline", [Segment(xx[1], pp),
                                Segment(xx[3], pp)]),
            ]
            # # check if the middle one is consistent
            # valid = False
            # p_A, p_B, p_C, p_a, p_b, p_c = [
            #     diagram.point_dict[ppp] for ppp in [A, B, C, a, b, c]
            # ]
            # for p1, p2, p3, p4, p5, p6 in [
            #     (p_A, p_B, p_C, p_a, p_b, p_c),
            #     (p_A, p_C, p_B, p_a, p_c, p_b),
            #     (p_B, p_A, p_C, p_b, p_a, p_c),
            # ]:
            #     if angle_type(p1 - p2, p3 - p2) == -1 and angle_type(
            #             p4 - p5, p6 - p5) == -1:
            #         valid = True
            #         break
            # if not valid:
            #     continue
            cond_x = min("".join(k)
                         for k in [(C, O, o, c), (O, C, c, o), (c, o, O,
                                                                C), (o, c, C,
                                                                     O)])
            if B < C:
                cond_eqline_ABC = A + B + A + C
            else:
                cond_eqline_ABC = A + C + A + B
            if b < c:
                cond_eqline_abc = a + b + a + c
            else:
                cond_eqline_abc = a + c + a + b
            if (cond_x in diagram.database.inverse_x
                    and diagram.database.is_eqline(
                        Segment(*cond_eqline_ABC[:2]),
                        Segment(*cond_eqline_ABC[2:]))
                    and diagram.database.is_eqline(
                        Segment(*cond_eqline_abc[:2]),
                        Segment(*cond_eqline_abc[2:]))):
                ppp = diagram.database.inverse_x[cond_x]
                f = Fact("eqline",
                         [Segment(p, ppp), Segment(pp, ppp)],
                         "x_and_x_to_pappus")
                f.add_parent(
                    Fact("eqline", [
                        Segment(*cond_eqline_ABC[:2]),
                        Segment(*cond_eqline_ABC[2:])
                    ]))
                f.add_parent(
                    Fact("eqline", [
                        Segment(*cond_eqline_abc[:2]),
                        Segment(*cond_eqline_abc[2:])
                    ]))
                f.add_parent(
                    Fact("eqline",
                         [Segment(cond_x[0], ppp),
                          Segment(cond_x[2], ppp)]))
                f.add_parent(
                    Fact("eqline",
                         [Segment(cond_x[1], ppp),
                          Segment(cond_x[3], ppp)]))
                f.add_parents(parents)
                facts.append(f)
                continue
            if cond_eqline_ABC not in diagram.database.pappus:
                diagram.database.pappus[cond_eqline_ABC] = OrderedSet()
            diagram.database.pappus[cond_eqline_ABC][cond_x,
                                                     cond_eqline_abc] = ((
                                                         p, pp), parents)
            if cond_eqline_abc not in diagram.database.pappus:
                diagram.database.pappus[cond_eqline_abc] = OrderedSet()
            diagram.database.pappus[cond_eqline_abc][cond_x,
                                                     cond_eqline_ABC] = ((
                                                         p, pp), parents)
    return facts


def eqline_and_pappus_to_eqline(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """The Pappus theorem."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return []
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    A = s12_joint
    B, C = s12_p_set
    if B < C:
        cond_eqline_ABC = A + B + A + C
    else:
        cond_eqline_ABC = A + C + A + B
    if cond_eqline_ABC not in diagram.database.pappus:
        return facts
    for conds, vals in diagram.database.pappus[cond_eqline_ABC].items():
        cond_x, cond_eqline_abc = conds
        if (cond_x in diagram.database.inverse_x
                and diagram.database.is_eqline(Segment(*cond_eqline_abc[:2]),
                                               Segment(*cond_eqline_abc[2:]))):
            ppp = diagram.database.inverse_x[cond_x]
            ps, parents = vals
            p, pp = ps
            f = Fact("eqline",
                     [Segment(p, ppp), Segment(pp, ppp)],
                     "eqline_and_pappus_to_eqline")
            f.add_parent(fact)
            f.add_parent(
                Fact("eqline", [
                    Segment(*cond_eqline_abc[:2]),
                    Segment(*cond_eqline_abc[2:])
                ]))
            f.add_parent(
                Fact("eqline",
                     [Segment(cond_x[0], ppp),
                      Segment(cond_x[2], ppp)]))
            f.add_parent(
                Fact("eqline",
                     [Segment(cond_x[1], ppp),
                      Segment(cond_x[3], ppp)]))
            f.add_parents(parents)
            facts.append(f)
    return facts


def x_and_x_and_x_to_desargues(diagram, x):
    """The Desargues theorem."""
    facts = []
    A, B, b, a, p = x
    l_Aa = Segment(A, a)
    l_Bb = Segment(B, b)
    q = diagram.database.itsll(l_Aa, l_Bb)
    Cs = []
    cs = []
    aas = []
    bbs = []
    p_A = diagram.point_dict[A]
    p_B = diagram.point_dict[B]
    p_a = diagram.point_dict[a]
    p_b = diagram.point_dict[b]
    for pp in diagram.point_dict:
        if pp in [A, a, B, b, p, q]:
            continue
        l_ppA = diagram.database.inverse_eqline[Segment(pp, A)]
        l_ppa = diagram.database.inverse_eqline[Segment(pp, a)]
        l_ppB = diagram.database.inverse_eqline[Segment(pp, B)]
        l_ppb = diagram.database.inverse_eqline[Segment(pp, b)]
        # p on the axis, config of (A, b, C) and (a, B, c)
        if len(diagram.database.lines_points[l_ppA]) >= 3 and len(
                diagram.database.lines_points[l_ppb]
        ) >= 3 and not on_same_line(p_A, p_b, diagram.point_dict[pp]):
            Cs.append(pp)
        elif len(diagram.database.lines_points[l_ppB]) >= 3 and len(
                diagram.database.lines_points[l_ppa]
        ) >= 3 and not on_same_line(p_B, p_a, diagram.point_dict[pp]):
            cs.append(pp)
        if not q:
            continue
        # p on the center, config of (A, a, aa) and (b, B, bb)
        if len(diagram.database.lines_points[l_ppA]) >= 3 and len(
                diagram.database.lines_points[l_ppa]
        ) >= 3 and not on_same_line(p_A, p_a, diagram.point_dict[pp]):
            aas.append(pp)
        elif len(diagram.database.lines_points[l_ppB]) >= 3 and len(
                diagram.database.lines_points[l_ppb]
        ) >= 3 and not on_same_line(p_B, p_b, diagram.point_dict[pp]):
            bbs.append(pp)
    if q:
        if q not in [A, a, B, b]:
            q_append = [
                Fact("eqline", [Segment(A, q), Segment(a, q)]),
                Fact("eqline", [Segment(B, q), Segment(b, q)])
            ]
        elif q in [A, a]:
            q_append = [Fact("eqline", [Segment(B, q), Segment(b, q)])]
        else:
            q_append = [Fact("eqline", [Segment(A, q), Segment(a, q)])]
    else:
        q_append = []
    for C, c in product(Cs, cs):
        if C == c:
            continue
        p_C = diagram.point_dict[C]
        p_c = diagram.point_dict[c]
        l_CA = Segment(C, A)
        l_Cb = Segment(C, b)
        l_ca = Segment(c, a)
        l_cB = Segment(c, B)
        l_Cc = Segment(C, c)
        if isclose((p_C - p_A).cross(p_c - p_a), 0) or isclose(
            (p_C - p_b).cross(p_c - p_B), 0):
            continue
        p_Y = diagram.database.itsll(l_CA, l_ca)
        p_X = diagram.database.itsll(l_Cb, l_cB)
        if not (p_Y and p_X and p_Y != p_X and p not in [p_Y, p_X]):
            continue
        if q:
            parents_append = q_append
            segments = [Segment(C, q), Segment(c, q)]
            ff_key = min(C + c, c + C) + q
        else:
            q_A = diagram.database.itsll(l_Cc, l_Aa)
            q_B = diagram.database.itsll(l_Cc, l_Bb)
            if q_A:
                if q_A not in [A, a, C, c]:
                    parents_append = [
                        Fact(
                            "eqline",
                            [Segment(A, q_A), Segment(a, q_A)]),
                        Fact(
                            "eqline",
                            [Segment(C, q_A), Segment(c, q_A)])
                    ]
                elif q_A in [A, a]:
                    parents_append = [
                        Fact(
                            "eqline",
                            [Segment(C, q_A), Segment(c, q_A)])
                    ]
                else:
                    parents_append = [
                        Fact(
                            "eqline",
                            [Segment(A, q_A), Segment(a, q_A)])
                    ]
                segments = [Segment(B, q_A), Segment(b, q_A)]
                ff_key = min(B + b, b + B) + q_A
            elif q_B:
                if q_B not in [B, b, C, c]:
                    parents_append = [
                        Fact(
                            "eqline",
                            [Segment(B, q_B), Segment(b, q_B)]),
                        Fact(
                            "eqline",
                            [Segment(C, q_B), Segment(c, q_B)])
                    ]
                elif q_B in [B, b]:
                    parents_append = [
                        Fact(
                            "eqline",
                            [Segment(C, q_B), Segment(c, q_B)])
                    ]
                else:
                    parents_append = [
                        Fact(
                            "eqline",
                            [Segment(B, q_B), Segment(b, q_B)])
                    ]
                segments = [Segment(A, q_B), Segment(a, q_B)]
                ff_key = min(A + a, a + A) + q_B
            else:
                continue
        parents = [
            Fact("eqline", [Segment(A, p), Segment(b, p)]),
            Fact("eqline", [Segment(B, p), Segment(a, p)])
        ]
        if p_Y not in [A, C, c, a]:
            parents += [
                Fact("eqline",
                     [Segment(A, p_Y), Segment(C, p_Y)]),
                Fact("eqline",
                     [Segment(c, p_Y), Segment(a, p_Y)])
            ]
        elif p_Y in [A, C]:
            parents.append(Fact("eqline", [Segment(c, p_Y), Segment(a, p_Y)]))
        else:
            parents.append(Fact("eqline", [Segment(A, p_Y), Segment(C, p_Y)]))
        if p_X not in [C, b, c, B]:
            parents += [
                Fact("eqline",
                     [Segment(C, p_X), Segment(b, p_X)]),
                Fact("eqline",
                     [Segment(c, p_X), Segment(B, p_X)])
            ]
        elif p_X in [C, b]:
            parents.append(Fact("eqline", [Segment(c, p_X), Segment(B, p_X)]))
        else:
            parents.append(Fact("eqline", [Segment(C, p_X), Segment(b, p_X)]))
        parents += parents_append
        if any(not diagram.database.is_eqline(*ff.objects) for ff in parents):
            continue
        f = Fact("eqline", segments, "x_and_x_and_x_to_desargues")
        f.add_parent(Fact("eqline", [Segment(p, p_Y), Segment(p, p_X)]))
        f.add_parents(parents)
        if diagram.database.is_eqline(Segment(p, p_Y), Segment(p, p_X)):
            facts.append(f)
        else:
            f_key = min(p_X + p_Y, p_Y + p_X) + p
            if f_key not in diagram.database.desargues:
                diagram.database.desargues[f_key] = OrderedSet()
            diagram.database.desargues[f_key][f] = None
        f = Fact("eqline", [Segment(p, p_Y), Segment(p, p_X)],
                 "x_and_x_and_x_to_desargues")
        f.add_parent(Fact("eqline", segments))
        f.add_parents(parents)
        if diagram.database.is_eqline(*segments):
            facts.append(f)
        else:
            if ff_key not in diagram.database.desargues:
                diagram.database.desargues[ff_key] = OrderedSet()
            diagram.database.desargues[ff_key][f] = None
    if not q:
        return facts
    for aa, bb in product(aas, bbs):
        if aa == bb:
            continue
        p_aa = diagram.point_dict[aa]
        p_bb = diagram.point_dict[bb]
        l_aaA = Segment(aa, A)
        l_aaa = Segment(aa, a)
        l_bbB = Segment(bb, B)
        l_bbb = Segment(bb, b)
        if isclose((p_aa - p_A).cross(p_bb - p_b), 0) or isclose(
            (p_aa - p_a).cross(p_bb - p_B), 0):
            continue
        p_Y = diagram.database.itsll(l_aaA, l_bbb)
        p_X = diagram.database.itsll(l_aaa, l_bbB)
        if not (p_Y and p_X and p_Y != p_X and q not in [p_Y, p_X]):
            continue
        parents = [
            Fact("eqline", [Segment(A, p), Segment(b, p)]),
            Fact("eqline", [Segment(B, p), Segment(a, p)])
        ] + q_append
        if p_Y not in [A, aa, bb, b]:
            parents += [
                Fact("eqline",
                     [Segment(A, p_Y), Segment(aa, p_Y)]),
                Fact("eqline",
                     [Segment(bb, p_Y), Segment(b, p_Y)]),
            ]
        elif p_Y in [A, aa]:
            parents.append(Fact("eqline", [Segment(bb, p_Y), Segment(b, p_Y)]))
        else:
            parents.append(Fact("eqline", [Segment(A, p_Y), Segment(aa, p_Y)]))
        if p_X not in [aa, a, bb, B]:
            parents += [
                Fact("eqline",
                     [Segment(aa, p_X), Segment(a, p_X)]),
                Fact("eqline",
                     [Segment(bb, p_X), Segment(B, p_X)])
            ]
        elif p_X in [aa, a]:
            parents.append(Fact("eqline", [Segment(bb, p_X), Segment(B, p_X)]))
        else:
            parents.append(Fact("eqline", [Segment(aa, p_X), Segment(a, p_X)]))
        if any(not diagram.database.is_eqline(*ff.objects) for ff in parents):
            continue
        f = Fact("eqline", [Segment(aa, p), Segment(bb, p)],
                 "x_and_x_and_x_to_desargues")
        f.add_parent(Fact("eqline", [Segment(p_Y, q), Segment(p_X, q)]))
        f.add_parents(parents)
        if diagram.database.is_eqline(Segment(p_Y, q), Segment(p_X, q)):
            facts.append(f)
        else:
            f_key = min(p_X + p_Y, p_Y + p_X) + q
            if f_key not in diagram.database.desargues:
                diagram.database.desargues[f_key] = OrderedSet()
            diagram.database.desargues[f_key][f] = None
        f = Fact("eqline", [Segment(p_Y, q), Segment(p_X, q)],
                 "x_and_x_and_x_to_desargues")
        f.add_parent(Fact("eqline", [Segment(aa, p), Segment(bb, p)]))
        f.add_parents(parents)
        if diagram.database.is_eqline(Segment(aa, p), Segment(bb, p)):
            facts.append(f)
        else:
            f_key = min(aa + bb, bb + aa) + p
            if f_key not in diagram.database.desargues:
                diagram.database.desargues[f_key] = OrderedSet()
            diagram.database.desargues[f_key][f] = None
    return facts


def eqline_and_desagues_to_eqline(diagram: 'Diagram',
                                  fact: Fact) -> List[Fact]:
    """The Desargue theorem."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return []
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    B = s12_joint
    A, C = s12_p_set
    f_key = min(A + C, C + A) + B
    if f_key not in diagram.database.desargues:
        return facts
    for f in diagram.database.desargues[f_key]:
        f.fn = "eqline_and_desagues_to_eqline"
        facts.append(f)
    return facts


def eqratio_and_eqline_to_harmonic(diagram: 'Diagram',
                                   fact: Fact) -> List[Fact]:
    """Cross-ratio of [A, C; B, D] equals -1."""
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
    if s1 == s2:
        return facts
    s3, s4 = r2.s1, r2.s2
    if s3 == s4:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s34_joint, s34_p_set = four_joint(s3.p1, s3.p2, s4.p1, s4.p2)
    if not s34_joint:
        return facts
    B = s12_joint
    D = s34_joint
    s12_p_set.pop(s12_joint)
    s34_p_set.pop(s34_joint)
    A, C = s12_p_set
    A_p, C_p = s34_p_set
    if not (B != D and A == A_p and C == C_p):
        return facts
    if len(set([A, B, C, D])) != 4:
        return facts
    if not diagram.database.is_eqline(Segment(A, C), Segment(B, D)):
        return facts
    p_A, p_B, p_C, p_D = [diagram.point_dict[p] for p in [A, B, C, D]]
    if angle_type(p_A - p_B, p_C - p_B) * angle_type(p_A - p_D,
                                                     p_C - p_D) != -1:
        return facts
    t1 = min(A + C, C + A)
    t2 = min(B + D, D + B)
    harmonic = min(t1 + t2, t2 + t1)
    if harmonic in diagram.database.harmonic:
        return facts
    diagram.database.harmonic_map[t1 + B] = (D, harmonic)
    diagram.database.harmonic_map[t1 + D] = (B, harmonic)
    diagram.database.harmonic_map[t2 + A] = (C, harmonic)
    diagram.database.harmonic_map[t2 + C] = (A, harmonic)
    diagram.database.harmonic[harmonic] = [
        fact, Fact("eqline", [Segment(A, C), Segment(B, D)])
    ]
    facts += harmonic_and_perp_to_eqangle(diagram, harmonic)
    facts += harmonic_and_eqangle_to_perp(diagram, harmonic)
    # facts += harmonic_and_sun_to_eqline(diagram, harmonic)
    # facts += harmonic_and_net_to_eqratio(diagram, harmonic)
    if t1 not in diagram.database.inverse_harmonic:
        diagram.database.inverse_harmonic[t1] = OrderedSet()
    if t2 not in diagram.database.inverse_harmonic:
        diagram.database.inverse_harmonic[t2] = OrderedSet()
    diagram.database.inverse_harmonic[t1][t2] = None
    diagram.database.inverse_harmonic[t2][t1] = None
    return facts


def eqline_and_eqratio_to_harmonic(diagram: 'Diagram',
                                   fact: Fact) -> List[Fact]:
    """Cross-ratio of [A, C; B, D] equals -1."""
    facts = []
    s1, s2 = fact.objects
    A, C = s1.p1, s1.p2
    B, D = s2.p1, s2.p2
    if len(set([A, B, C, D])) != 4:
        return facts
    p_A, p_B, p_C, p_D = [diagram.point_dict[p] for p in [A, B, C, D]]
    if angle_type(p_A - p_B, p_C - p_B) * angle_type(p_A - p_D,
                                                     p_C - p_D) != -1:
        return facts
    if diagram.database.is_cong(Segment(A, B), Segment(
            B, C)) or diagram.database.is_cong(Segment(A, D), Segment(
                D, C)) or not diagram.database.is_eqratio(
                    Ratio(Segment(A, B), Segment(B, C)),
                    Ratio(Segment(A, D), Segment(D, C))):
        return facts
    t1 = min(A + C, C + A)
    t2 = min(B + D, D + B)
    harmonic = min(t1 + t2, t2 + t1)
    if harmonic in diagram.database.harmonic:
        return facts
    diagram.database.harmonic_map[t1 + B] = (D, harmonic)
    diagram.database.harmonic_map[t1 + D] = (B, harmonic)
    diagram.database.harmonic_map[t2 + A] = (C, harmonic)
    diagram.database.harmonic_map[t2 + C] = (A, harmonic)
    diagram.database.harmonic[harmonic] = [
        fact,
        Fact("eqratio", [
            Ratio(Segment(A, B), Segment(B, C)),
            Ratio(Segment(A, D), Segment(D, C))
        ])
    ]
    facts += harmonic_and_perp_to_eqangle(diagram, harmonic)
    facts += harmonic_and_eqangle_to_perp(diagram, harmonic)
    # facts += harmonic_and_sun_to_eqline(diagram, harmonic)
    # facts += harmonic_and_net_to_eqratio(diagram, harmonic)
    if t1 not in diagram.database.inverse_harmonic:
        diagram.database.inverse_harmonic[t1] = OrderedSet()
    if t2 not in diagram.database.inverse_harmonic:
        diagram.database.inverse_harmonic[t2] = OrderedSet()
    diagram.database.inverse_harmonic[t1][t2] = None
    diagram.database.inverse_harmonic[t2][t1] = None
    return facts


# def eqratio_and_eqcircle_to_harmonic(diagram: 'Diagram',
#                                      fact: Fact) -> List[Fact]:
#     """Cross-ratio of [A, C; B, D] equals -1 on a circle."""
#     facts = []
#     r1, r2 = fact.objects
#     if ((diagram.database.is_cong(r1.s1, r2.s2)
#          and diagram.database.is_cong(r1.s2, r2.s1))
#             or diagram.database.is_cong(r1.s1, r1.s2)
#             or diagram.database.is_cong(r2.s1, r2.s2)):
#         return facts
#     s1, s2 = r1.s1, r1.s2
#     if s1 == s2:
#         return facts
#     s3, s4 = r2.s1, r2.s2
#     if s3 == s4:
#         return facts
#     s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
#     if not s12_joint:
#         return facts
#     s34_joint, s34_p_set = four_joint(s3.p1, s3.p2, s4.p1, s4.p2)
#     if not s34_joint:
#         return facts
#     B = s12_joint
#     D = s34_joint
#     s12_p_set.pop(s12_joint)
#     s34_p_set.pop(s34_joint)
#     A, C = s12_p_set
#     A_p, C_p = s34_p_set
#     if not (B != D and A == A_p and C == C_p):
#         return facts
#     if len(set([A, B, C, D])) != 4:
#         return facts
#     if not diagram.database.is_eqcircle(Circle(None, [A, B, C]),
#                                         Circle(None, [B, C, D])):
#         return facts
#     p_A, p_B, p_C, p_D = [diagram.point_dict[p] for p in [A, B, C, D]]
#     l_AC = Line(p_A, p_C)
#     p_pB = l_AC.project(p_B)
#     p_pD = l_AC.project(p_D)
#     if p_pB in [p_A, p_C] or p_pD in [p_A, p_C]:
#         return facts
#     if angle_type(p_A - p_pB, p_C - p_pB) * angle_type(p_A - p_pD,
#                                                        p_C - p_pD) != -1:
#         return facts
#     t1 = min(A + C, C + A)
#     t2 = min(B + D, D + B)
#     harmonic = min(t1 + t2, t2 + t1)
#     if harmonic in diagram.database.harmonic:
#         return facts
#     diagram.database.harmonic_map[t1 + B] = (D, harmonic)
#     diagram.database.harmonic_map[t1 + D] = (B, harmonic)
#     diagram.database.harmonic_map[t2 + A] = (C, harmonic)
#     diagram.database.harmonic_map[t2 + C] = (A, harmonic)
#     diagram.database.harmonic[harmonic] = [
#         fact,
#         Fact("eqcircle", [Circle(None, [A, B, C]),
#                           Circle(None, [B, C, D])])
#     ]
#     facts += harmonic_and_perp_to_eqangle(diagram, harmonic)
#     facts += harmonic_and_eqangle_to_perp(diagram, harmonic)
#     # facts += harmonic_and_sun_to_eqline(diagram, harmonic)
#     # facts += harmonic_and_net_to_eqratio(diagram, harmonic)
#     if t1 not in diagram.database.inverse_harmonic:
#         diagram.database.inverse_harmonic[t1] = OrderedSet()
#     if t2 not in diagram.database.inverse_harmonic:
#         diagram.database.inverse_harmonic[t2] = OrderedSet()
#     diagram.database.inverse_harmonic[t1][t2] = None
#     diagram.database.inverse_harmonic[t2][t1] = None
#     return facts

# def eqcircle_and_eqratio_to_harmonic(diagram: 'Diagram',
#                                      fact: Fact) -> List[Fact]:
#     """Cross-ratio of [A, C; B, D] equals -1 on a circle."""
#     facts = []
#     c1, c2 = fact.objects
#     if c1.center or c2.center:
#         return facts
#     A, B, C = c1.points
#     D, E, F = c2.points
#     if len(set([A, B, C, D, E, F])) != 4:
#         return facts
#     A, B, C, D = list(OrderedSet.fromkeys([A, B, C, D, E, F]))
#     if len(set([A, B, C, D])) != 4:
#         return facts
#     valid = False
#     for check_quad in [(A, B, C, D), (A, C, B, D), (A, D, B, C)]:
#         p1, p2, p3, p4 = [diagram.point_dict[p] for p in check_quad]
#         if intersect(p1, p3, p2, p4):
#             l = Line(p1, p3)
#             p_p2 = l.project(p2)
#             p_p4 = l.project(p4)
#             if p_p2 not in [p1, p3] and p_p4 not in [
#                     p1, p3
#             ] and angle_type(p1 - p_p2, p3 - p_p2) * angle_type(
#                     p1 - p_p4, p3 - p_p4) == -1:
#                 valid = True
#                 break
#     if not valid:
#         return facts
#     A, B, C, D = check_quad
#     if diagram.database.is_cong(Segment(A, B), Segment(
#             B, C)) or diagram.database.is_cong(Segment(A, D), Segment(
#                 D, C)) or not diagram.database.is_eqratio(
#                     Ratio(Segment(A, B), Segment(B, C)),
#                     Ratio(Segment(A, D), Segment(D, C))):
#         return facts
#     t1 = min(A + C, C + A)
#     t2 = min(B + D, D + B)
#     harmonic = min(t1 + t2, t2 + t1)
#     if harmonic in diagram.database.harmonic:
#         return facts
#     diagram.database.harmonic_map[t1 + B] = (D, harmonic)
#     diagram.database.harmonic_map[t1 + D] = (B, harmonic)
#     diagram.database.harmonic_map[t2 + A] = (C, harmonic)
#     diagram.database.harmonic_map[t2 + C] = (A, harmonic)
#     diagram.database.harmonic[harmonic] = [
#         fact,
#         Fact("eqratio", [
#             Ratio(Segment(A, B), Segment(B, C)),
#             Ratio(Segment(A, D), Segment(D, C))
#         ])
#     ]
#     facts += harmonic_and_perp_to_eqangle(diagram, harmonic)
#     facts += harmonic_and_eqangle_to_perp(diagram, harmonic)
#     # facts += harmonic_and_sun_to_eqline(diagram, harmonic)
#     # facts += harmonic_and_net_to_eqratio(diagram, harmonic)
#     if t1 not in diagram.database.inverse_harmonic:
#         diagram.database.inverse_harmonic[t1] = OrderedSet()
#     if t2 not in diagram.database.inverse_harmonic:
#         diagram.database.inverse_harmonic[t2] = OrderedSet()
#     diagram.database.inverse_harmonic[t1][t2] = None
#     diagram.database.inverse_harmonic[t2][t1] = None
#     return facts


def harmonic_and_perp_to_eqangle(diagram: 'Diagram',
                                 harmonic: str) -> List[Fact]:
    """Harmonic and perp implies eqangle."""
    facts = []
    A, C, B, D = harmonic
    for pts in [(A, C, B, D), (B, D, A, C)]:
        s_perp = Segment(pts[2], pts[3])
        if s_perp in diagram.database.h_segments_perps:
            for a in diagram.database.h_segments_perps[s_perp]:
                P = a.name[1]
                f = Fact("eqangle",
                         [Angle(pts[0], P, pts[2]),
                          Angle(pts[2], P, pts[1])],
                         "harmonic_and_perp_to_eqangle")
                f.add_parent(Fact("perp", [a]))
                f.add_parents(diagram.database.harmonic[harmonic])
                facts.append(f)
    return facts


def perp_and_harmonic_to_eqangle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Harmonic and perp implies eqangle."""
    facts = []
    a = fact.objects[0]
    B, P, D = a.name
    BD = min(B + D, D + B)
    if BD in diagram.database.inverse_harmonic:
        for AC in diagram.database.inverse_harmonic[BD]:
            A, C = AC
            harmonic = min(AC + BD, BD + AC)
            f = Fact("eqangle",
                     [Angle(A, P, B), Angle(B, P, C)],
                     "perp_and_harmonic_to_eqangle")
            f.add_parent(fact)
            f.add_parents(diagram.database.harmonic[harmonic])
            facts.append(f)
    return facts


def harmonic_and_eqangle_to_perp(diagram: 'Diagram',
                                 harmonic: str) -> List[Fact]:
    """Harmonic and eqangle implies perp."""
    facts = []
    A, C, B, D = harmonic
    for pts in [(A, C, B, D), (B, D, A, C)]:
        if pts[0] not in diagram.database.points_eqangles or pts[
                1] not in diagram.database.points_eqangles:
            continue
        for s in diagram.database.points_eqangles[pts[0]]:
            P = s.p1 if s.p1 != pts[0] else s.p2
            if Segment(P,
                       pts[1]) not in diagram.database.points_eqangles[pts[1]]:
                continue
            if on_same_line(
                    *[diagram.point_dict[p] for p in [pts[0], pts[1], P]]):
                continue
            a1 = Angle(pts[0], P, pts[2])
            a2 = Angle(pts[2], P, pts[1])
            a3 = Angle(pts[0], P, pts[3])
            a4 = Angle(pts[3], P, pts[1])
            if diagram.database.is_eqangle(a1, a2):
                f = Fact("perp", [Angle(pts[2], P, pts[3])],
                         "harmonic_and_eqangle_to_perp")
                f.add_parent(Fact("eqangle", [a1, a2]))
                f.add_parents(diagram.database.harmonic[harmonic])
                facts.append(f)
            elif diagram.database.is_eqangle(a3, a4):
                f = Fact("perp", [Angle(pts[2], P, pts[3])],
                         "harmonic_and_eqangle_to_perp")
                f.add_parent(Fact("eqangle", [a3, a4]))
                f.add_parents(diagram.database.harmonic[harmonic])
                facts.append(f)
    return facts


def eqangle_and_harmonic_to_perp(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Harmonic and eqangle implies perp."""
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
    P = a1.name[1]
    A = s1.p2
    C = s4.p2
    AC = min(A + C, C + A)
    B = s2.p2
    AC_B = AC + B
    if AC_B not in diagram.database.harmonic_map:
        return facts
    D, harmonic = diagram.database.harmonic_map[AC_B]
    f = Fact("perp", [Angle(B, P, D)], "eqangle_and_harmonic_to_perp")
    f.add_parent(fact)
    f.add_parents(diagram.database.harmonic[harmonic])
    facts.append(f)
    return facts


# def star_and_star_and_eqline_or_eqcircle_to_sun(diagram: 'Diagram',
#                                                 x: str) -> List[Fact]:
#     """From degree two to degree three."""
#     facts = []
#     A, B, b, a, p = x
#     key_pAb, val_pAb = p + A + b, p + B + a
#     key_pBa, val_pBa = sort_two_from_first(B + a, A + b)
#     key_pBa, val_pBa = p + key_pBa, p + val_pBa
#     for key, val in [(key_pAb, val_pAb), (key_pBa, val_pBa)]:
#         if key not in diagram.database.star:
#             return facts
#         C, c = val[1:]
#         for star in diagram.database.star[key]:
#             D, E, d, e = star[:-1]
#             cond_eqline_CDE = Fact("eqline", [Segment(C, D), Segment(D, E)])
#             cond_eqcircle_CDE = Fact(
#                 "eqcircle", [Circle(None, [p, C, D]),
#                              Circle(None, [p, D, E])])
#             cond_eqline_cde = Fact("eqline", [Segment(c, d), Segment(d, e)])
#             cond_eqcircle_cde = Fact(
#                 "eqcircle", [Circle(None, [p, c, d]),
#                              Circle(None, [p, d, e])])
#             CDE_good = diagram.database.is_eqline(
#                 *cond_eqline_CDE.objects) or diagram.database.is_eqcircle(
#                     *cond_eqcircle_CDE.objects)
#             cde_good = diagram.database.is_eqline(
#                 *cond_eqline_cde.objects) or diagram.database.is_eqcircle(
#                     *cond_eqcircle_cde.objects)
#             x = min("".join(sort_two_from_first(C + D + E, c + d + e)),
#                     "".join(sort_two_from_first(c + d + e, C + D + E)))
#             x = x + p
#             if x in diagram.database.inverse_sun:
#                 continue
#             if CDE_good and cde_good:
#                 facts += sun_and_harmonic_to_eqline(diagram, x)
#                 key_sun = x[-1] + x[0] + x[3]
#                 if key_sun not in diagram.database.sun:
#                     diagram.database.sun[key_sun] = {}
#                 diagram.database.sun[key_sun][x] = None
#                 diagram.database.inverse_sun[x] = None
#                 if x[:3] not in diagram.database.sun_map:
#                     diagram.database.sun_map[x[:3]] = {}
#                 diagram.database.sun_map[x[:3]][x[3:]] = None
#                 continue
#             if cond_eqline_CDE not in diagram.database.star_sun:
#                 diagram.database.star_sun[cond_eqline_CDE] = {}
#             diagram.database.star_sun[cond_eqline_CDE][cond_eqline_cde] = x
#             diagram.database.star_sun[cond_eqline_CDE][cond_eqcircle_cde] = x
#             if cond_eqcircle_CDE not in diagram.database.star_sun:
#                 diagram.database.star_sun[cond_eqcircle_CDE] = {}
#             diagram.database.star_sun[cond_eqcircle_CDE][cond_eqline_cde] = x
#             diagram.database.star_sun[cond_eqcircle_CDE][cond_eqcircle_cde] = x
#             if cond_eqline_cde not in diagram.database.star_sun:
#                 diagram.database.star_sun[cond_eqline_cde] = {}
#             diagram.database.star_sun[cond_eqline_cde][cond_eqline_CDE] = x
#             diagram.database.star_sun[cond_eqline_cde][cond_eqcircle_CDE] = x
#             if cond_eqcircle_cde not in diagram.database.star_sun:
#                 diagram.database.star_sun[cond_eqcircle_cde] = {}
#             diagram.database.star_sun[cond_eqcircle_cde][cond_eqline_CDE] = x
#             diagram.database.star_sun[cond_eqcircle_cde][cond_eqcircle_CDE] = x
#     return facts

# def eqline_and_star_and_star_to_sun(diagram: 'Diagram',
#                                     fact: Fact) -> List[Fact]:
#     """From degree two to degree three."""
#     facts = []
#     if fact not in diagram.database.star_sun:
#         return facts
#     for cond, x in diagram.database.star_sun[fact].items():
#         if x in diagram.database.inverse_sun:
#             continue
#         if cond.type == "eqline" and diagram.database.is_eqline(
#                 *cond.objects
#         ) or cond.type == "eqcircle" and diagram.database.is_eqcircle(
#                 *cond.objects):
#             facts += sun_and_harmonic_to_eqline(diagram, x)
#             key_sun = x[-1] + x[0] + x[3]
#             if key_sun not in diagram.database.sun:
#                 diagram.database.sun[key_sun] = {}
#             diagram.database.sun[key_sun][x] = None
#             diagram.database.inverse_sun[x] = None
#             if x[:3] not in diagram.database.sun_map:
#                 diagram.database.sun_map[x[:3]] = {}
#             diagram.database.sun_map[x[:3]][x[3:]] = None
#     return facts

# def eqcircle_and_star_and_star_to_sun(diagram: 'Diagram',
#                                       fact: Fact) -> List[Fact]:
#     """From degree two to degree three."""
#     facts = []
#     if fact not in diagram.database.star_sun:
#         return facts
#     for cond, x in diagram.database.star_sun[fact].items():
#         if x in diagram.database.inverse_sun:
#             continue
#         if cond.type == "eqline" and diagram.database.is_eqline(
#                 *cond.objects
#         ) or cond.type == "eqcircle" and diagram.database.is_eqcircle(
#                 *cond.objects):
#             facts += sun_and_harmonic_to_eqline(diagram, x)
#             key_sun = x[-1] + x[0] + x[3]
#             if key_sun not in diagram.database.sun:
#                 diagram.database.sun[key_sun] = {}
#             diagram.database.sun[key_sun][x] = None
#             diagram.database.inverse_sun[x] = None
#             if x[:3] not in diagram.database.sun_map:
#                 diagram.database.sun_map[x[:3]] = {}
#             diagram.database.sun_map[x[:3]][x[3:]] = None
#     return facts

# def harmonic_and_sun_to_eqline(diagram: 'Diagram',
#                                harmonic: str) -> List[Fact]:
#     """The shooting lemma."""
#     facts = []
#     A, C, B, D = harmonic
#     key_sort = "".join(sorted(harmonic))
#     for idx in range(4):
#         key_sun = key_sort[:idx] + key_sort[idx + 1:]
#         if key_sun not in diagram.database.sun_map:
#             continue
#         if key_sort[idx] == A:
#             D_key = B + D + C
#         elif key_sort[idx] == C:
#             D_key = B + D + A
#         elif key_sort[idx] == B:
#             D_key = A + C + D
#         else:
#             D_key = A + C + B
#         idx_0 = key_sun.index(D_key[0])
#         idx_1 = key_sun.index(D_key[1])
#         idx_2 = key_sun.index(D_key[2])
#         for val_sun_p in diagram.database.sun_map[key_sun]:
#             val_sun, p = val_sun_p[:-1], val_sun_p[-1]
#             d_key = val_sun[idx_0] + val_sun[idx_1]
#             d_key = min(d_key, d_key[::-1]) + val_sun[idx_2]
#             if d_key not in diagram.database.harmonic_map:
#                 continue
#             d, d_harmonic = diagram.database.harmonic_map[d_key]
#             f = Fact("eqline", [Segment(p, key_sort[idx]), Segment(p, d)])
#             f.add_parent(
#                 Fact("eqline",
#                      [Segment(p, D_key[0]),
#                       Segment(p, val_sun[idx_0])]))
#             f.add_parent(
#                 Fact("eqline",
#                      [Segment(p, D_key[1]),
#                       Segment(p, val_sun[idx_1])]))
#             f.add_parent(
#                 Fact("eqline",
#                      [Segment(p, D_key[2]),
#                       Segment(p, val_sun[idx_2])]))
#             f.parents += diagram.database.harmonic[harmonic]
#             f.parents += diagram.database.harmonic[d_harmonic]
#             facts.append(f)
#     return facts

# def sun_and_harmonic_to_eqline(diagram: 'Diagram', x: str) -> List[Fact]:
#     """The shooting lemma."""
#     facts = []
#     A, B, C, b, a, c, p = x
#     for indices in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
#         D_key = x[indices[0]] + x[indices[1]]
#         D_key = D_key + x[indices[2]]
#         d_key = x[indices[0] + 3] + x[indices[1] + 3]
#         d_key = min(d_key, d_key[::-1]) + x[indices[2] + 3]
#         if (D_key not in diagram.database.harmonic_map
#                 or d_key not in diagram.database.harmonic_map):
#             continue
#         D, D_harmonic = diagram.database.harmonic_map[D_key]
#         d, d_harmonic = diagram.database.harmonic_map[d_key]
#         f = Fact("eqline", [Segment(p, D), Segment(p, d)])
#         f.add_parent(Fact("eqline", [Segment(p, A), Segment(p, b)]))
#         f.add_parent(Fact("eqline", [Segment(p, B), Segment(p, a)]))
#         f.add_parent(Fact("eqline", [Segment(p, C), Segment(p, c)]))
#         f.parents += diagram.database.harmonic[D_harmonic]
#         f.parents += diagram.database.harmonic[d_harmonic]
#         facts.append(f)
#     return facts

# def star_and_sun_and_eqline_or_eqcircle_to_net(diagram: 'Diagram',
#                                                x: str) -> List[Fact]:
#     """From degree three to degree four."""
#     facts = []
#     A, B, b, a, p = x
#     key_pAb, val_pAb = p + A + b, p + B + b
#     key_pBa, val_pBa = sort_two_from_first(B + a, A + b)
#     key_pBa, val_pBa = p + key_pBa, p + val_pBa
#     for key, val in [(key_pAb, val_pAb), (key_pBa, val_pBa)]:
#         if key not in diagram.database.sun:
#             return facts
#         C, c = val[1:]
#         for sun in diagram.database.sun[key]:
#             D, E, F, d, e, f = sun[:-1]
#             if diagram.database.is_eqline(Segment(D, E), Segment(E, F)):
#                 cond_CDEF = Fact("eqline", [Segment(C, D), Segment(E, F)])
#             else:
#                 cond_CDEF = Fact(
#                     "eqcircle",
#                     [Circle(None, [p, C, D]),
#                      Circle(None, [p, E, F])])
#             if diagram.database.is_eqline(Segment(d, e), Segment(e, f)):
#                 cond_cdef = Fact("eqline", [Segment(c, d), Segment(e, f)])
#             else:
#                 cond_cdef = Fact(
#                     "eqcircle",
#                     [Circle(None, [p, c, d]),
#                      Circle(None, [p, e, f])])
#             CDEF_good = diagram.database.is_eqline(
#                 *cond_CDEF.objects
#             ) if cond_CDEF.type == "eqline" else diagram.database.is_eqcircle(
#                 *cond_CDEF.objects)
#             cdef_good = diagram.database.is_eqline(
#                 *cond_cdef.objects
#             ) if cond_cdef.type == "eqline" else diagram.database.is_eqcircle(
#                 *cond_cdef.objects)
#             x = min("".join(sort_two_from_first(C + D + E + F, c + d + e + f)),
#                     "".join(sort_two_from_first(c + d + e + f, C + D + E + F)))
#             x = x + p
#             if x in diagram.database.inverse_net:
#                 continue
#             if CDEF_good and cdef_good:
#                 facts += net_and_harmonic_to_eqratio(diagram, x)
#                 key_net = x[-1] + x[0] + x[4]
#                 if key_net not in diagram.database.net:
#                     diagram.database.net[key_net] = {}
#                 diagram.database.net[key_net][x] = None
#                 diagram.database.inverse_net[x] = None
#                 if x[:4] not in diagram.database.net_map:
#                     diagram.database.net_map[x[:4]] = {}
#                 diagram.database.net_map[x[:4]][x[4:]] = None
#                 continue
#             if cond_CDEF not in diagram.database.star_net:
#                 diagram.database.star_net[cond_CDEF] = {}
#             diagram.database.star_net[cond_CDEF][cond_cdef] = x
#             if cond_cdef not in diagram.database.star_net:
#                 diagram.database.star_net[cond_cdef] = {}
#             diagram.database.star_net[cond_cdef][cond_CDEF] = x
#     return facts

# def eqline_and_star_and_sun_to_net(diagram: 'Diagram',
#                                    fact: Fact) -> List[Fact]:
#     """From degree three to degree four."""
#     facts = []
#     if fact not in diagram.database.star_net:
#         return facts
#     for cond, x in diagram.database.star_net[fact].items():
#         if x in diagram.database.inverse_net:
#             continue
#         if (cond.type == "eqline" and diagram.database.is_eqline(
#                 *cond.objects)) or diagram.database.is_eqcircle(*cond.objects):
#             facts += net_and_harmonic_to_eqratio(diagram, x)
#             key = x[-1] + x[0] + x[4]
#             if key not in diagram.database.net:
#                 diagram.database.net[key] = {}
#             diagram.database.net[key][x] = None
#             diagram.database.inverse_net[x] = None
#             if x[:4] not in diagram.database.net_map:
#                 diagram.database.net_map[x[:4]] = {}
#             diagram.database.net_map[x[:4]][x[4:]] = None
#     return facts

# def eqcircle_and_star_and_sun_to_net(diagram: 'Diagram',
#                                      fact: Fact) -> List[Fact]:
#     """From degree three to degree four."""
#     facts = []
#     if fact not in diagram.database.star_net:
#         return facts
#     for cond, x in diagram.database.star_net[fact].items():
#         if x in diagram.database.inverse_net:
#             continue
#         if (cond.type == "eqline" and diagram.database.is_eqline(
#                 *cond.objects)) or diagram.database.is_eqcircle(*cond.objects):
#             facts += net_and_harmonic_to_eqratio(diagram, x)
#             key = x[-1] + x[0] + x[4]
#             if key not in diagram.database.net:
#                 diagram.database.net[key] = {}
#             diagram.database.net[key][x] = None
#             diagram.database.inverse_net[x] = None
#             if x[:4] not in diagram.database.net_map:
#                 diagram.database.net_map[x[:4]] = {}
#             diagram.database.net_map[x[:4]][x[4:]] = None
#     return facts

# def harmonic_and_net_to_eqratio(diagram: 'Diagram',
#                                 harmonic: str) -> List[Fact]:
#     """Harmonic and cross ratio implies eqratio."""
#     facts = []
#     A, C, B, D = harmonic
#     key = "".join(sorted(harmonic))
#     if key not in diagram.database.net_map:
#         return facts
#     idx_A = key.index(A)
#     idx_C = key.index(C)
#     idx_B = key.index(B)
#     idx_D = key.index(D)
#     for another_map_p in diagram.database.net_map[key]:
#         another_map, p = another_map_p[:-1], another_map_p[-1]
#         a, c, b, d = another_map[idx_A], another_map[idx_C], another_map[
#             idx_B], another_map[idx_D]
#         f = Fact("eqratio", [
#             Ratio(Segment(a, b), Segment(b, c)),
#             Ratio(Segment(a, d), Segment(d, c))
#         ])
#         parents = []
#         if diagram.database.is_eqline(Segment(a, c), Segment(b, d)):
#             parents.append(Fact("eqline", [Segment(a, c), Segment(b, d)]))
#         else:
#             parents.append(
#                 Fact("eqcircle",
#                      [Circle(None, [p, a, c]),
#                       Circle(None, [p, b, d])]))
#         parents.append(Fact("eqline", [Segment(p, A), Segment(p, a)]))
#         parents.append(Fact("eqline", [Segment(p, B), Segment(p, b)]))
#         parents.append(Fact("eqline", [Segment(p, C), Segment(p, c)]))
#         parents.append(Fact("eqline", [Segment(p, D), Segment(p, d)]))
#         parents += diagram.database.harmonic[harmonic]
#         if not diagram.database.is_eqline(Segment(A, C), Segment(B, D)):
#             parents.append(
#                 Fact("eqcircle",
#                      [Circle(None, [p, A, C]),
#                       Circle(None, [p, B, D])]))
#         facts.parents = parents
#         facts.append(f)
#     return facts

# def net_and_harmonic_to_eqratio(diagram: 'Diagram', net: str) -> List[Fact]:
#     """Harmonic and cross ratio implies eqratio."""
#     facts = []
#     p = net[-1]
#     key_ABCD, val_ABCD = net[:4], net[4:-1]
#     key_bacd, val_bacd = sort_two_from_first(val_ABCD, key_ABCD)
#     for key, val in [(key_ABCD, val_ABCD), (key_bacd, val_bacd)]:
#         for indices in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]:
#             AC = key[indices[0]] + key[indices[1]]
#             BD = key[indices[2]] + key[indices[3]]
#             if not (AC in diagram.database.inverse_harmonic
#                     and BD in diagram.database.inverse_harmonic[AC]):
#                 continue
#             A, C = AC
#             B, D = BD
#             harmonic = min(AC + BD, BD + AC)
#             a, c = val[indices[0]], val[indices[1]]
#             b, d = val[indices[2]], val[indices[3]]
#             f = Fact("eqratio", [
#                 Ratio(Segment(a, b), Segment(b, c)),
#                 Ratio(Segment(a, d), Segment(d, c))
#             ])
#             parents = []
#             if diagram.database.is_eqline(Segment(a, c), Segment(b, d)):
#                 parents.append(Fact("eqline", [Segment(a, c), Segment(b, d)]))
#             else:
#                 parents.append(
#                     Fact("eqcircle",
#                          [Circle(None, [p, a, c]),
#                           Circle(None, [p, b, d])]))
#             parents.append(Fact("eqline", [Segment(p, A), Segment(p, a)]))
#             parents.append(Fact("eqline", [Segment(p, B), Segment(p, b)]))
#             parents.append(Fact("eqline", [Segment(p, C), Segment(p, c)]))
#             parents.append(Fact("eqline", [Segment(p, D), Segment(p, d)]))
#             parents += diagram.database.harmonic[harmonic]
#             if not diagram.database.is_eqline(Segment(A, C), Segment(B, D)):
#                 parents.append(
#                     Fact("eqcircle",
#                          [Circle(None, [p, A, C]),
#                           Circle(None, [p, B, D])]))
#             facts.parents = parents
#             facts.append(f)
#     return facts


def eqline_to_cevian_middle(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Construct a harmonic base from Cevian bundle in eqline. Complete quad
    CABO, corresponding FDEO."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    O = s12_joint
    A, D = s12_p_set
    l = diagram.database.inverse_eqline[s1]
    for ll in diagram.database.points_lines[D]:
        if ll == l or len(diagram.database.lines_points[ll]) < 3:
            continue
        X = ll.p1 if ll.p1 != D else ll.p2
        if X in [A, O] or on_same_line(
                *[diagram.point_dict[p] for p in [X, D, A]], eps=1e-4):
            continue
        for B, C in combinations(diagram.database.lines_points[ll], 2):
            if D in [B, C]:
                continue
            if B in [A, O] or C in [A, O]:
                diagram.terminal_flag = True
                diagram.inf_terminal = True
                return []
            E = diagram.database.itsll(Segment(A, C), Segment(B, O))
            if not E:
                continue
            F = diagram.database.itsll(Segment(A, B), Segment(C, O))
            if not F:
                continue
            ABC, DEF = sort_two_from_first(A + B + C, D + E + F)
            key = ABC + DEF
            if key in diagram.database.cevian:
                continue
            parents = [
                fact,
                Fact("eqline", [Segment(B, O), Segment(O, E)]),
                Fact("eqline", [Segment(C, O), Segment(O, F)]),
                Fact("eqline", [Segment(B, D), Segment(D, C)]),
                Fact("eqline", [Segment(A, E), Segment(E, C)]),
                Fact("eqline", [Segment(A, F), Segment(F, B)]),
            ]
            diagram.database.cevian[key] = parents
            for indices in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
                tp1, tp2, cp1, cp2 = ABC[indices[0]], ABC[indices[1]], DEF[
                    indices[0]], DEF[indices[1]]
                q = DEF[indices[2]]
                p = diagram.database.itsll(Segment(tp1, tp2),
                                           Segment(cp1, cp2))
                if p:
                    f = Fact("eqratio", [
                        Ratio(Segment(tp1, q), Segment(tp2, q)),
                        Ratio(Segment(tp1, p), Segment(tp2, p))
                    ], "eqline_to_cevian_middle")
                    f.add_parent(
                        Fact(
                            "eqline",
                            [Segment(tp1, p), Segment(tp2, p)]))
                    f.add_parent(
                        Fact(
                            "eqline",
                            [Segment(cp1, p), Segment(cp2, p)]))
                    f.add_parents(parents)
                    facts.append(f)
                else:
                    key_inv = min("".join(k)
                                  for k in [(tp1, cp2, tp2,
                                             cp1), (cp2, tp1, cp1,
                                                    tp2), (cp1, tp2, cp2,
                                                           tp1)])
                    if key_inv not in diagram.database.inverse_cevian:
                        diagram.database.inverse_cevian[key_inv] = OrderedSet()
                    diagram.database.inverse_cevian[key_inv][key] = (q, tp1,
                                                                     tp2)
    return facts


def eqline_to_cevian_side(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Construct a harmonic base from Cevian bundle in eqline. Complete quad
    CABO, corresponding FDEO."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    D = s12_joint
    B, C = s12_p_set
    l = diagram.database.inverse_eqline[s1]
    for ll in diagram.database.points_lines[D]:
        if ll == l or len(diagram.database.lines_points[ll]) < 3:
            continue
        X = ll.p1 if ll.p1 != D else ll.p2
        if X in [B, C] or on_same_line(
                *[diagram.point_dict[p] for p in [X, D, B]], eps=1e-4):
            continue
        for A, O in permutations(diagram.database.lines_points[ll], 2):
            if D in [A, O]:
                continue
            if A in [B, C] or O in [B, C]:
                diagram.terminal_flag = True
                diagram.inf_terminal = True
                return []
            E = diagram.database.itsll(Segment(A, C), Segment(B, O))
            if not E:
                continue
            F = diagram.database.itsll(Segment(A, B), Segment(C, O))
            if not F:
                continue
            ABC, DEF = sort_two_from_first(A + B + C, D + E + F)
            key = ABC + DEF
            if key in diagram.database.cevian:
                continue
            parents = [
                fact,
                Fact("eqline", [Segment(B, O), Segment(O, E)]),
                Fact("eqline", [Segment(C, O), Segment(O, F)]),
                Fact("eqline", [Segment(A, O), Segment(O, D)]),
                Fact("eqline", [Segment(A, E), Segment(E, C)]),
                Fact("eqline", [Segment(A, F), Segment(F, B)]),
            ]
            diagram.database.cevian[key] = parents
            for indices in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
                tp1, tp2, cp1, cp2 = ABC[indices[0]], ABC[indices[1]], DEF[
                    indices[0]], DEF[indices[1]]
                q = DEF[indices[2]]
                p = diagram.database.itsll(Segment(tp1, tp2),
                                           Segment(cp1, cp2))
                if p:
                    f = Fact("eqratio", [
                        Ratio(Segment(tp1, q), Segment(tp2, q)),
                        Ratio(Segment(tp1, p), Segment(tp2, p))
                    ], "eqline_to_cevian_side")
                    f.add_parent(
                        Fact(
                            "eqline",
                            [Segment(tp1, p), Segment(tp2, p)]))
                    f.add_parent(
                        Fact(
                            "eqline",
                            [Segment(cp1, p), Segment(cp2, p)]))
                    f.add_parents(parents)
                    facts.append(f)
                else:
                    key_inv = min("".join(k)
                                  for k in [(tp1, cp2, tp2,
                                             cp1), (cp2, tp1, cp1,
                                                    tp2), (cp1, tp2, cp2,
                                                           tp1)])
                    if key_inv not in diagram.database.inverse_cevian:
                        diagram.database.inverse_cevian[key_inv] = OrderedSet()
                    diagram.database.inverse_cevian[key_inv][key] = (q, tp1,
                                                                     tp2)
    return facts
