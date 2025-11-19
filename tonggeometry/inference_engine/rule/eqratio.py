r"""Eqratio-related rules."""

from itertools import product
from typing import TYPE_CHECKING, List

from tonggeometry.constructor.primitives import same_dir
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Circle, Ratio, Segment
from tonggeometry.inference_engine.util import OrderedSet, four_joint

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def eqratio_to_eqratio(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Eqratio equivalent."""
    facts = []
    r1, r2 = fact.objects
    if ((diagram.database.is_cong(r1.s1, r2.s2)
         and diagram.database.is_cong(r1.s2, r2.s1))
            or diagram.database.is_cong(r1.s1, r1.s2)
            or diagram.database.is_cong(r2.s1, r2.s2)):
        return facts
    s1, s2 = r1.s1, r1.s2
    s3, s4 = r2.s1, r2.s2
    f = Fact("eqratio", [Ratio(s1, s3), Ratio(s2, s4)], "eqratio_to_eqratio")
    f.add_parent(fact)
    facts.append(f)
    return facts


def eqratio_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """eqratio(s1,s2,s2,s1) -> cong(s1,s2)."""
    facts = []
    r1, r2 = fact.objects
    if (diagram.database.is_cong(r1.s1, r2.s2)
            and diagram.database.is_cong(r1.s2, r2.s1)):
        s1, s2 = r1.s1, r1.s2
        s3, s4 = r2.s1, r2.s2
        f = Fact("cong", [s1, s2], "eqratio_to_cong")
        f.add_parent(fact)
        if s1 != s4:
            f.add_parent(Fact("cong", [s1, s4]))
        if s2 != s3:
            f.add_parent(Fact("cong", [s2, s3]))
        facts.append(f)
    return facts


def eqratio_and_cong_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """One pair cong in eqratio implies another pair cong."""
    facts = []
    r1, r2 = fact.objects
    if (diagram.database.is_cong(r1.s1, r2.s2)
            and diagram.database.is_cong(r1.s2, r2.s1)):
        return facts
    s1, s2 = r1.s1, r1.s2
    s3, s4 = r2.s1, r2.s2
    if diagram.database.is_cong(s1, s2):
        f = Fact("cong", [s3, s4], "eqratio_and_cong_to_cong")
        f.add_parent(fact)
        if s1 != s2:
            f.add_parent(Fact("cong", [s1, s2]))
        facts.append(f)
    elif diagram.database.is_cong(s3, s4):
        f = Fact("cong", [s1, s2], "eqratio_and_cong_to_cong")
        f.add_parent(fact)
        if s3 != s4:
            f.add_parent(Fact("cong", [s3, s4]))
        facts.append(f)
    return facts


def cong_and_eqratio_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """One pair cong in eqratio implies another pair cong."""
    facts = []
    s1, s2 = fact.objects
    r = Ratio(s1, s2)
    if r not in diagram.database.inverse_eqratio:
        return facts
    eqclass_rep = diagram.database.inverse_eqratio[r]
    eqclass = diagram.database.eqratio[eqclass_rep]
    r = eqclass[r]
    for rr in eqclass:
        if rr == r or diagram.database.is_cong(rr.s1, rr.s2):
            continue
        s3, s4 = rr.s1, rr.s2
        f = Fact("cong", [s3, s4], "cong_and_eqratio_to_cong")
        f.add_parent(fact)
        f.add_parent(Fact("eqratio", [r, rr]))
        facts.append(f)
    return facts


def eqratio_and_eqline_and_eqline_to_eqratio(diagram: 'Diagram',
                                             fact: Fact) -> List[Fact]:
    """Eqratio arithmetics."""
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
    if not (s12_joint and s34_joint):
        return facts
    s12_p_set.pop(s12_joint)
    A, C = s12_p_set
    B = s12_joint
    s34_p_set.pop(s34_joint)
    D, F = s34_p_set
    E = s34_joint
    if not (diagram.database.is_eqline(Segment(A, B), Segment(B, C))
            and diagram.database.is_eqline(Segment(D, E), Segment(E, F))):
        return facts
    dir_val = same_dir(*[diagram.point_dict[p] for p in [A, B, C, D, E, F]])
    if dir_val == 1:
        f = Fact("eqratio", [
            Ratio(Segment(A, C), Segment(B, C)),
            Ratio(Segment(D, F), Segment(E, F))
        ], "eqratio_and_eqline_and_eqline_to_eqratio")
        f.add_parent(fact)
        f.add_parent(Fact("eqline", [Segment(A, B), Segment(B, C)]))
        f.add_parent(Fact("eqline", [Segment(D, E), Segment(E, F)]))
        facts.append(f)
        f = Fact("eqratio", [
            Ratio(Segment(A, C), Segment(A, B)),
            Ratio(Segment(D, F), Segment(D, E))
        ], "eqratio_and_eqline_and_eqline_to_eqratio")
        f.add_parent(fact)
        f.add_parent(Fact("eqline", [Segment(A, B), Segment(B, C)]))
        f.add_parent(Fact("eqline", [Segment(D, E), Segment(E, F)]))
        facts.append(f)
    return facts


def eqline_and_eqline_and_eqratio_to_eqratio(diagram: 'Diagram',
                                             fact: Fact) -> List[Fact]:
    """Eqratio arithmetics."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2 or diagram.database.is_cong(s1, s2):
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    r = Ratio(s1, s2)
    if r not in diagram.database.inverse_eqratio:
        return facts
    s12_p_set.pop(s12_joint)
    A, C = s12_p_set
    B = s12_joint
    eqclass_rep = diagram.database.inverse_eqratio[r]
    eqclass = diagram.database.eqratio[eqclass_rep]
    r = eqclass[r]
    if r.s1 != s1:
        A, C = C, A
    for rr in eqclass:
        if rr == r:
            continue
        s34_joint, s34_p_set = four_joint(rr.s1.p1, rr.s1.p2, rr.s2.p1,
                                          rr.s2.p2)
        if not s34_joint:
            continue
        s34_p_set.pop(s34_joint)
        D, F = s34_p_set
        E = s34_joint
        if not diagram.database.is_eqline(Segment(D, E), Segment(E, F)):
            continue
        dir_val = same_dir(
            *[diagram.point_dict[p] for p in [A, B, C, D, E, F]])
        if dir_val == 1:
            f = Fact("eqratio", [
                Ratio(Segment(A, C), Segment(B, C)),
                Ratio(Segment(D, F), Segment(E, F))
            ], "eqline_and_eqline_and_eqratio_to_eqratio")
            f.add_parent(fact)
            f.add_parent(Fact("eqratio", [r, rr]))
            f.add_parent(Fact("eqline", [Segment(D, E), Segment(E, F)]))
            facts.append(f)
            f = Fact("eqratio", [
                Ratio(Segment(A, C), Segment(A, B)),
                Ratio(Segment(D, F), Segment(D, E))
            ], "eqline_and_eqline_and_eqratio_to_eqratio")
            f.add_parent(fact)
            f.add_parent(Fact("eqratio", [r, rr]))
            f.add_parent(Fact("eqline", [Segment(D, E), Segment(E, F)]))
            facts.append(f)
    return facts


def cong_and_eqratio_to_eqratio(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Eqratio equivalent with a segment replaced with its cong."""
    facts = []
    s1, s2 = fact.objects
    if s1 in diagram.database.segments_eqratios:
        for r in diagram.database.segments_eqratios[s1]:
            if r.s1 == s1:
                f = Fact("eqratio", [r, Ratio(s2, r.s2)],
                         "cong_and_eqratio_to_eqratio")
                f.add_parent(fact)
                facts.append(f)
            if r.s2 == s1:
                f = Fact("eqratio", [r, Ratio(r.s1, s2)])
                f.add_parent(fact)
                facts.append(f)
    if s2 in diagram.database.segments_eqratios:
        for r in diagram.database.segments_eqratios[s2]:
            if r.s1 == s2:
                f = Fact("eqratio", [r, Ratio(s1, r.s2)],
                         "cong_and_eqratio_to_eqratio")
                f.add_parent(fact)
                facts.append(f)
            if r.s2 == s2:
                f = Fact("eqratio", [r, Ratio(r.s1, s1)],
                         "cong_and_eqratio_to_eqratio")
                f.add_parent(fact)
                facts.append(f)
    return facts


def eqratio_and_cong_to_eqratio(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """Eqratio equivalent with a segment replaced with its cong. One-stop link
    between any two ratios."""
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
    if s1 in diagram.database.inverse_cong:
        eqclass_s1 = diagram.database.cong[diagram.database.inverse_cong[s1]]
        for s1_p in eqclass_s1:
            if s1_p == s1:
                continue
            f = Fact("eqratio", [Ratio(s1_p, s2), r1],
                     "eqratio_and_cong_to_eqratio")
            f.add_parent(Fact("cong", [s1, s1_p]))
            f.add_parent(fact)
            facts.append(f)
            f = Fact("eqratio", [Ratio(s1_p, s2), r2],
                     "eqratio_and_cong_to_eqratio")
            f.add_parent(Fact("cong", [s1, s1_p]))
            f.add_parent(fact)
            facts.append(f)
    if s2 in diagram.database.inverse_cong:
        eqclass_s2 = diagram.database.cong[diagram.database.inverse_cong[s2]]
        for s2_p in eqclass_s2:
            if s2_p == s2:
                continue
            f = Fact("eqratio", [Ratio(s1, s2_p), r1],
                     "eqratio_and_cong_to_eqratio")
            f.add_parent(Fact("cong", [s2, s2_p]))
            f.add_parent(fact)
            facts.append(f)
            f = Fact("eqratio", [Ratio(s1, s2_p), r2],
                     "eqratio_and_cong_to_eqratio")
            f.add_parent(Fact("cong", [s2, s2_p]))
            f.add_parent(fact)
            facts.append(f)
    if s3 in diagram.database.inverse_cong:
        eqclass_s3 = diagram.database.cong[diagram.database.inverse_cong[s3]]
        for s3_p in eqclass_s3:
            if s3_p == s3:
                continue
            f = Fact("eqratio", [Ratio(s3_p, s4), r2],
                     "eqratio_and_cong_to_eqratio")
            f.add_parent(Fact("cong", [s3, s3_p]))
            f.add_parent(fact)
            facts.append(f)
            f = Fact("eqratio", [Ratio(s3_p, s4), r1],
                     "eqratio_and_cong_to_eqratio")
            f.add_parent(Fact("cong", [s3, s3_p]))
            f.add_parent(fact)
            facts.append(f)
    if s4 in diagram.database.inverse_cong:
        eqclass_s4 = diagram.database.cong[diagram.database.inverse_cong[s4]]
        for s4_p in eqclass_s4:
            if s4_p == s4:
                continue
            f = Fact("eqratio", [Ratio(s3, s4_p), r2],
                     "eqratio_and_cong_to_eqratio")
            f.add_parent(Fact("cong", [s4, s4_p]))
            f.add_parent(fact)
            facts.append(f)
            f = Fact("eqratio", [Ratio(s3, s4_p), r1],
                     "eqratio_and_cong_to_eqratio")
            f.add_parent(Fact("cong", [s4, s4_p]))
            f.add_parent(fact)
            facts.append(f)
    return facts


def eqratio_and_eqratio_to_eqratio(diagram: 'Diagram',
                                   fact: Fact) -> List[Fact]:
    """eqratio(r1,r2) & eqratio(r3,r4) => eqratio(r1+r3,r2+r4)"""
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
    partition = {"prod": {}, "div": {}}
    for r1_p in diagram.database.segments_eqratios[s1]:
        if r1_p == r1 or diagram.database.is_cong(r1_p.s1, r1_p.s2):
            continue
        s2_p = r1_p.s1 if r1_p.s1 != s1 else r1_p.s2
        r = Ratio(s2_p, s2)
        if r1_p.s1 == s1:
            key = "div"
        else:
            key = "prod"
        rep = diagram.database.inverse_eqratio[r1_p]
        if rep in partition[key]:
            partition[key][rep][0].append((r1_p, r))
        else:
            partition[key][rep] = [[(r1_p, r)], []]
    for r1_p in diagram.database.segments_eqratios[s2]:
        if r1_p == r1 or diagram.database.is_cong(r1_p.s1, r1_p.s2):
            continue
        s1_p = r1_p.s1 if r1_p.s1 != s2 else r1_p.s2
        r = Ratio(s1, s1_p)
        if r1_p.s1 == s2:
            key = "prod"
        else:
            key = "div"
        rep = diagram.database.inverse_eqratio[r1_p]
        if rep in partition[key]:
            partition[key][rep][0].append((r1_p, r))
        else:
            partition[key][rep] = [[(r1_p, r)], []]
    for r2_p in diagram.database.segments_eqratios[s3]:
        if r2_p == r2 or diagram.database.is_cong(r2_p.s1, r2_p.s2):
            continue
        s4_p = r2_p.s1 if r2_p.s1 != s3 else r2_p.s2
        r = Ratio(s4_p, s4)
        if r2_p.s1 == s3:
            key = "div"
        else:
            key = "prod"
        rep = diagram.database.inverse_eqratio[r2_p]
        if rep in partition[key]:
            partition[key][rep][1].append((r2_p, r))
    for r2_p in diagram.database.segments_eqratios[s4]:
        if r2_p == r2 or diagram.database.is_cong(r2_p.s1, r2_p.s2):
            continue
        s3_p = r2_p.s1 if r2_p.s1 != s4 else r2_p.s2
        r = Ratio(s3, s3_p)
        if r2_p.s1 == s4:
            key = "prod"
        else:
            key = "div"
        rep = diagram.database.inverse_eqratio[r2_p]
        if rep in partition[key]:
            partition[key][rep][1].append((r2_p, r))
    for mode in partition.values():
        for pairs in mode.values():
            for r3_r, r4_rr in product(pairs[0], pairs[1]):
                r3, r = r3_r
                r4, rr = r4_rr
                if (r3 == r2 and r4 == r1) or r3 == r4:
                    continue
                f = Fact("eqratio", [r, rr], "eqratio_and_eqratio_to_eqratio")
                f.add_parent(fact)
                f.add_parent(Fact("eqratio", [r3, r4]))
                facts.append(f)
    return facts


def add_simili(diagram: 'Diagram', P: str, O1: str, R1: str, O2: str, R2: str,
               insimili: bool) -> bool:
    """Log similitude centers with respect to O1R1 and O2R2. Use rep for keys."""
    c1 = Circle(O1, [R1])
    c2 = Circle(O2, [R2])
    if c1 in diagram.database.inverse_eqcircle:
        c1_key = diagram.database.inverse_eqcircle[c1]
    else:
        c1_key = c1
    if c2 in diagram.database.inverse_eqcircle:
        c2_key = diagram.database.inverse_eqcircle[c2]
    else:
        c2_key = c2
    if c1_key in diagram.database.simili and c2_key in diagram.database.simili[
            c1_key] and diagram.database.simili[c1_key][c2_key][insimili]:
        return False
    if c1_key not in diagram.database.simili:
        diagram.database.simili[c1_key] = OrderedSet()
    if c2_key not in diagram.database.simili[c1_key]:
        diagram.database.simili[c1_key][c2_key] = {True: None, False: None}
    if c2_key not in diagram.database.simili:
        diagram.database.simili[c2_key] = OrderedSet()
    if c1_key not in diagram.database.simili[c2_key]:
        diagram.database.simili[c2_key][c1_key] = {True: None, False: None}
    diagram.database.simili[c1_key][c2_key][insimili] = (P, O1, R1, O2, R2)
    diagram.database.simili[c2_key][c1_key][insimili] = (P, O2, R2, O1, R1)
    return True


def eqratio_and_eqline_to_simili(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """O1P/O2P = O1R1/O2R1 and P O1 O2 eqline means similitude center."""
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
    s13_joint, _ = four_joint(s1.p1, s1.p2, s3.p1, s3.p2)
    s24_joint, _ = four_joint(s2.p1, s2.p2, s4.p1, s4.p2)
    if not (s13_joint and s24_joint and s13_joint != s24_joint):
        return facts
    O1 = s13_joint
    O2 = s24_joint
    s12_joint, _ = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    s34_joint, _ = four_joint(s3.p1, s3.p2, s4.p1, s4.p2)
    p_O1, p_O2 = [diagram.point_dict[p] for p in [O1, O2]]
    if s12_joint and diagram.database.is_eqline(Segment(O1, s12_joint),
                                                Segment(O2, s12_joint)):
        P = s12_joint
        R1 = s3.p1 if s3.p1 != O1 else s3.p2
        R2 = s4.p1 if s4.p1 != O2 else s4.p2
        p_P = diagram.point_dict[P]
        insimili = p_O1 < p_P < p_O2 or p_O2 < p_P < p_O1
        if add_simili(diagram, P, O1, R1, O2, R2, insimili):
            facts += monge(diagram, P, O1, R1, O2, R2, insimili)
    if s34_joint and diagram.database.is_eqline(Segment(O1, s34_joint),
                                                Segment(O2, s34_joint)):
        P = s34_joint
        R1 = s1.p1 if s1.p1 != O1 else s1.p2
        R2 = s2.p1 if s2.p1 != O2 else s2.p2
        p_P = diagram.point_dict[P]
        insimili = p_O1 < p_P < p_O2 or p_O2 < p_P < p_O1
        if add_simili(diagram, P, O1, R1, O2, R2, insimili):
            facts += monge(diagram, P, O1, R1, O2, R2, insimili)
    return facts


def eqline_and_eqratio_to_simili(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """O1P/O2P = O1R1/O2R1 and P O1 O2 eqline means similitude center."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    P = s12_joint
    O1, O2 = s12_p_set
    r = Ratio(Segment(O1, P), Segment(O2, P))
    if r not in diagram.database.inverse_eqratio:
        return facts
    eqclass_rep = diagram.database.inverse_eqratio[r]
    eqclass = diagram.database.eqratio[eqclass_rep]
    r = eqclass[r]
    O1 = r.s1.p1 if r.s1.p1 != P else r.s1.p2
    O2 = r.s2.p1 if r.s2.p1 != P else r.s2.p2
    p_P, p_O1, p_O2 = [diagram.point_dict[p] for p in [P, O1, O2]]
    insimili = p_O1 < p_P < p_O2 or p_O2 < p_P < p_O1
    for rr in eqclass:
        if rr == r:
            continue
        if not (O1 in [rr.s1.p1, rr.s1.p2] and O2 in [rr.s2.p1, rr.s2.p2]):
            continue
        R1 = rr.s1.p1 if rr.s1.p1 != O1 else rr.s1.p2
        R2 = rr.s2.p1 if rr.s2.p1 != O2 else rr.s2.p2
        if add_simili(diagram, P, O1, R1, O2, R2, insimili):
            facts += monge(diagram, P, O1, R1, O2, R2, insimili)
    return facts


def monge(diagram: 'Diagram', P: str, O1: str, R1: str, O2: str, R2: str,
          insimili: bool) -> List[Fact]:
    """Monge theorem regarding three circles with similitude centers.
    1. Three exsimilitude centers are collinear.
    2. One exsimilitude center and two insimilitude centers are collinear.
    """
    facts = []
    c1 = Circle(O1, [R1])
    c2 = Circle(O2, [R2])
    if c1 in diagram.database.inverse_eqcircle:
        c1_key = diagram.database.inverse_eqcircle[c1]
    else:
        c1_key = c1
    if c2 in diagram.database.inverse_eqcircle:
        c2_key = diagram.database.inverse_eqcircle[c2]
    else:
        c2_key = c2
    parents_P = [
        Fact("eqratio", [
            Ratio(Segment(O1, P), Segment(O2, P)),
            Ratio(Segment(O1, R1), Segment(O2, R2))
        ]),
        Fact("eqline", [Segment(O1, P), Segment(O2, P)])
    ]
    for c3_key in diagram.database.simili[c1_key]:
        if c3_key == c2_key:
            continue
        if diagram.database.simili[c1_key][c3_key][
                True] and diagram.database.simili[c1_key][c3_key][True][0] != P:
            PPs = diagram.database.simili[c1_key][c3_key][True]
            PP, PPO1, PPR1, PPO2, PPR2 = PPs
            parents_PP = [
                Fact("eqratio", [
                    Ratio(Segment(PPO1, PP), Segment(PPO2, PP)),
                    Ratio(Segment(PPO1, PPR1), Segment(PPO2, PPR2))
                ]),
                Fact("eqline",
                     [Segment(PPO1, PP), Segment(PPO2, PP)])
            ]
            PPPs = None
            if c3_key in diagram.database.simili[
                    c2_key] and insimili and diagram.database.simili[c2_key][
                        c3_key][False]:
                PPPs = diagram.database.simili[c2_key][c3_key][False]
            if c3_key in diagram.database.simili[
                    c2_key] and not insimili and diagram.database.simili[
                        c2_key][c3_key][True]:
                PPPs = diagram.database.simili[c2_key][c3_key][True]
            if PPPs and PPPs[0] not in [P, PP]:
                PPP, PPPO1, PPPR1, PPPO2, PPPR2 = PPPs
                parents_PPP = [
                    Fact("eqratio", [
                        Ratio(Segment(PPPO1, PPP), Segment(PPPO2, PPP)),
                        Ratio(Segment(PPPO1, PPPR1), Segment(PPPO2, PPPR2))
                    ]),
                    Fact("eqline", [Segment(PPPO1, PPP),
                                    Segment(PPPO2, PPP)])
                ]
                f = Fact("eqline", [Segment(P, PP), Segment(P, PPP)], "monge")
                f.add_parents(parents_P)
                f.add_parents(parents_PP)
                f.add_parents(parents_PPP)
                if (O1, R1) != (PPO1, PPR1):
                    f.add_parent(
                        Fact("eqcircle",
                             [Circle(O1, [R1]),
                              Circle(PPO1, [PPR1])]))
                if (O2, R2) != (PPPO1, PPPR1):
                    f.add_parent(
                        Fact("eqcircle",
                             [Circle(O2, [R2]),
                              Circle(PPPO1, [PPPR1])]))
                if (PPO2, PPR2) != (PPPO2, PPPR2):
                    f.add_parent(
                        Fact("eqcircle",
                             [Circle(PPO2, [PPR2]),
                              Circle(PPPO2, [PPPR2])]))
                facts.append(f)
        if diagram.database.simili[c1_key][c3_key][
                False] and diagram.database.simili[c1_key][c3_key][False][
                    0] != P:
            PPs = diagram.database.simili[c1_key][c3_key][False]
            PP, PPO1, PPR1, PPO2, PPR2 = PPs
            parents_PP = [
                Fact("eqratio", [
                    Ratio(Segment(PPO1, PP), Segment(PPO2, PP)),
                    Ratio(Segment(PPO1, PPR1), Segment(PPO2, PPR2))
                ]),
                Fact("eqline",
                     [Segment(PPO1, PP), Segment(PPO2, PP)])
            ]
            PPPs = None
            if c3_key in diagram.database.simili[
                    c2_key] and insimili and diagram.database.simili[c2_key][
                        c3_key][True]:
                PPPs = diagram.database.simili[c2_key][c3_key][True]
            if c3_key in diagram.database.simili[
                    c2_key] and not insimili and diagram.database.simili[
                        c2_key][c3_key][False]:
                PPPs = diagram.database.simili[c2_key][c3_key][False]
            if PPPs and PPPs[0] not in [P, PP]:
                PPP, PPPO1, PPPR1, PPPO2, PPPR2 = PPPs
                parents_PPP = [
                    Fact("eqratio", [
                        Ratio(Segment(PPPO1, PPP), Segment(PPPO2, PPP)),
                        Ratio(Segment(PPPO1, PPPR1), Segment(PPPO2, PPPR2))
                    ]),
                    Fact("eqline", [Segment(PPPO1, PPP),
                                    Segment(PPPO2, PPP)])
                ]
                f = Fact("eqline", [Segment(P, PP), Segment(P, PPP)], "monge")
                f.add_parents(parents_P)
                f.add_parents(parents_PP)
                f.add_parents(parents_PPP)
                if (O1, R1) != (PPO1, PPR1):
                    f.add_parent(
                        Fact("eqcircle",
                             [Circle(O1, [R1]),
                              Circle(PPO1, [PPR1])]))
                if (O2, R2) != (PPPO1, PPPR1):
                    f.add_parent(
                        Fact("eqcircle",
                             [Circle(O2, [R2]),
                              Circle(PPPO1, [PPPR1])]))
                if (PPO2, PPR2) != (PPPO2, PPPR2):
                    f.add_parent(
                        Fact("eqcircle",
                             [Circle(PPO2, [PPR2]),
                              Circle(PPPO2, [PPPR2])]))
                facts.append(f)
    return facts
