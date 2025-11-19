r"""Midpoint-related rules."""

from typing import TYPE_CHECKING, List

from tonggeometry.constructor.primitives import on_same_line
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Ratio, Segment, Triangle
from tonggeometry.inference_engine.util import four_joint
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def midp_to_eqline(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """midp implies collinearity."""
    facts = []
    M, s = fact.objects
    f = Fact("eqline", [Segment(M, s.p1), Segment(M, s.p2)], "midp_to_eqline")
    f.add_parent(fact)
    facts.append(f)
    return facts


def midp_to_cong(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """midp implies equality of segments."""
    facts = []
    M, s = fact.objects
    f = Fact("cong", [Segment(M, s.p1), Segment(M, s.p2)], "midp_to_eqline")
    f.add_parent(fact)
    facts.append(f)
    return facts


def midp_to_eqratio(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """midp implies equal ratio."""
    facts = []
    M, s = fact.objects
    f = Fact("eqratio",
             [Ratio(Segment(M, s.p1), s),
              Ratio(Segment(M, s.p2), s)], "midp_to_eqline")
    f.add_parent(fact)
    facts.append(f)
    return facts


def midp_and_midp_to_eqratio(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """midp ratio is constant. One-stop link between any two ratios."""
    facts = []
    M, s = fact.objects
    for M_p in diagram.database.midp:
        for s_p in diagram.database.midp[M_p]:
            if M_p != M or s_p != s:
                f = Fact("eqratio", [
                    Ratio(Segment(M, s.p1), s),
                    Ratio(Segment(M_p, s_p.p1), s_p)
                ], "midp_and_midp_to_eqratio")
                f.add_parent(fact)
                f.add_parent(Fact("midp", [M_p, s_p]))
                facts.append(f)
                f = Fact("eqratio", [
                    Ratio(Segment(M, s.p2), s),
                    Ratio(Segment(M_p, s_p.p1), s_p)
                ], "midp_and_midp_to_eqratio")
                f.add_parent(fact)
                f.add_parent(Fact("midp", [M_p, s_p]))
                facts.append(f)
                f = Fact("eqratio", [
                    Ratio(Segment(M, s.p1), s),
                    Ratio(Segment(M_p, s_p.p2), s_p)
                ], "midp_and_midp_to_eqratio")
                f.add_parent(fact)
                f.add_parent(Fact("midp", [M_p, s_p]))
                facts.append(f)
                f = Fact("eqratio", [
                    Ratio(Segment(M, s.p2), s),
                    Ratio(Segment(M_p, s_p.p2), s_p)
                ], "midp_and_midp_to_eqratio")
                f.add_parent(fact)
                f.add_parent(Fact("midp", [M_p, s_p]))
                facts.append(f)
    return facts


def midp_and_eqline_to_centri(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """The median theorem."""
    facts = []
    m1, s1 = fact.objects
    for p, ppp in [(s1.p1, s1.p2), (s1.p2, s1.p1)]:
        for s2 in diagram.database.points_midps[p]:
            if s2 == s1 or on_same_line(
                    *[diagram.point_dict[p] for p in set(str(s1) + str(s2))],
                    eps=2e-5):
                continue
            pp = s2.p1 if s2.p1 != p else s2.p2
            m2 = diagram.database.inverse_midp[s2]
            G = diagram.database.itsll(Segment(m1, pp), Segment(m2, ppp))
            if not G:
                continue
            t = Triangle(p, pp, ppp)
            if t in diagram.database.centris:
                continue
            diagram.database.centris[t] = G
            if G not in diagram.database.inverse_centris:
                diagram.database.inverse_centris[G] = OrderedSet()
            diagram.database.inverse_centris[G][t] = None
            s3 = Segment(pp, ppp)
            if s3 not in diagram.database.h_segments_centris:
                diagram.database.h_segments_centris[s3] = OrderedSet()
            diagram.database.h_segments_centris[s3][t] = None
            f = Fact("eqratio", [
                Ratio(Segment(m1, G), Segment(G, pp)),
                Ratio(Segment(m2, G), Segment(G, ppp))
            ], "midp_and_eqline_to_centri")
            f.add_parent(fact)
            f.add_parent(Fact("midp", [m2, s2]))
            f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
            f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
            facts.append(f)
            m3 = diagram.database.itsll(Segment(p, G), s3)
            if s3 in diagram.database.inverse_midp:
                m3 = diagram.database.inverse_midp[s3]
                f = Fact("eqline",
                         [Segment(p, G), Segment(G, m3)],
                         "midp_and_eqline_to_centri")
                f.add_parent(fact)
                f.add_parent(Fact("midp", [m2, s2]))
                f.add_parent(Fact("midp", [m3, s3]))
                f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
                f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
                facts.append(f)
                f = Fact("eqratio", [
                    Ratio(Segment(m1, G), Segment(G, pp)),
                    Ratio(Segment(m3, G), Segment(G, p))
                ], "midp_and_eqline_to_centri")
                f.add_parent(fact)
                f.add_parent(Fact("midp", [m2, s2]))
                f.add_parent(Fact("midp", [m3, s3]))
                f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
                f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
                facts.append(f)
            elif m3:
                f = Fact("midp", [m3, s3], "midp_and_eqline_to_centri")
                f.add_parent(fact)
                f.add_parent(Fact("midp", [m2, s2]))
                f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
                f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
                f.add_parent(Fact("eqline", [Segment(m3, G), Segment(G, p)]))
                f.add_parent(
                    Fact("eqline",
                         [Segment(pp, m3), Segment(ppp, m3)]))
                facts.append(f)
                f = Fact("eqratio", [
                    Ratio(Segment(m1, G), Segment(G, pp)),
                    Ratio(Segment(m3, G), Segment(G, p))
                ], "midp_and_eqline_to_centri")
                f.add_parent(fact)
                f.add_parent(Fact("midp", [m2, s2]))
                f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
                f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
                f.add_parent(Fact("eqline", [Segment(m3, G), Segment(G, p)]))
                f.add_parent(
                    Fact("eqline",
                         [Segment(pp, m3), Segment(ppp, m3)]))
                facts.append(f)
    return facts


def eqline_and_midp_to_centri(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """The median theorem."""
    facts = []
    s1, s2 = fact.objects
    if s1 == s2:
        return facts
    s12_joint, s12_p_set = four_joint(s1.p1, s1.p2, s2.p1, s2.p2)
    if not s12_joint:
        return facts
    s12_p_set.pop(s12_joint)
    G = s12_joint
    A, C = s12_p_set
    p_A, p_G, p_C = [diagram.point_dict[p] for p in [A, G, C]]
    if not (p_A < p_G < p_C or p_C < p_G < p_A):
        return facts
    for m1, pp in [(A, C), (C, A)]:
        if m1 not in diagram.database.midp:
            continue
        for s1 in diagram.database.midp[m1]:
            if on_same_line(
                    *[diagram.point_dict[p] for p in [pp, s1.p1, s1.p2]]):
                continue
            for p, ppp in [(s1.p1, s1.p2), (s1.p2, s1.p1)]:
                s2 = Segment(p, pp)
                if s2 not in diagram.database.inverse_midp:
                    continue
                m2 = diagram.database.inverse_midp[s2]
                if not diagram.database.is_eqline(Segment(m2, G),
                                                  Segment(G, ppp)):
                    continue
                t = Triangle(p, pp, ppp)
                if t in diagram.database.centris:
                    continue
                diagram.database.centris[t] = G
                if G not in diagram.database.inverse_centris:
                    diagram.database.inverse_centris[G] = OrderedSet()
                diagram.database.inverse_centris[G][t] = None
                s3 = Segment(pp, ppp)
                if s3 not in diagram.database.h_segments_centris:
                    diagram.database.h_segments_centris[s3] = OrderedSet()
                diagram.database.h_segments_centris[s3][t] = None
                f = Fact("eqratio", [
                    Ratio(Segment(m1, G), Segment(G, pp)),
                    Ratio(Segment(m2, G), Segment(G, ppp))
                ], "eqline_and_midp_to_centri")
                f.add_parent(fact)
                f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
                f.add_parent(Fact("midp", [m1, s1]))
                f.add_parent(Fact("midp", [m2, s2]))
                facts.append(f)
                m3 = diagram.database.itsll(Segment(p, G), s3)
                if s3 in diagram.database.inverse_midp:
                    m3 = diagram.database.inverse_midp[s3]
                    f = Fact("eqline",
                             [Segment(p, G), Segment(G, m3)],
                             "eqline_and_midp_to_centri")
                    f.add_parent(fact)
                    f.add_parent(Fact("midp", [m1, s1]))
                    f.add_parent(Fact("midp", [m2, s2]))
                    f.add_parent(Fact("midp", [m3, s3]))
                    f.add_parent(
                        Fact("eqline",
                             [Segment(m2, G), Segment(G, ppp)]))
                    facts.append(f)
                    f = Fact("eqratio", [
                        Ratio(Segment(m1, G), Segment(G, pp)),
                        Ratio(Segment(m3, G), Segment(G, p))
                    ], "eqline_and_midp_to_centri")
                    f.add_parent(fact)
                    f.add_parent(Fact("midp", [m1, s1]))
                    f.add_parent(Fact("midp", [m2, s2]))
                    f.add_parent(Fact("midp", [m3, s3]))
                    f.add_parent(
                        Fact("eqline",
                             [Segment(m2, G), Segment(G, ppp)]))
                    facts.append(f)
                elif m3:
                    f = Fact("midp", [m3, s3], "eqline_and_midp_to_centri")
                    f.add_parent(fact)
                    f.add_parent(
                        Fact("eqline",
                             [Segment(m2, G), Segment(G, ppp)]))
                    f.add_parent(
                        Fact("eqline",
                             [Segment(p, G), Segment(G, m3)]))
                    f.add_parent(
                        Fact("eqline", [Segment(pp, m3),
                                        Segment(ppp, m3)]))
                    f.add_parent(Fact("midp", [m1, s1]))
                    f.add_parent(Fact("midp", [m2, s2]))
                    facts.append(f)
                    f = Fact("eqratio", [
                        Ratio(Segment(m1, G), Segment(G, pp)),
                        Ratio(Segment(m3, G), Segment(G, p))
                    ], "eqline_and_midp_to_centri")
                    f.add_parent(fact)
                    f.add_parent(
                        Fact("eqline",
                             [Segment(m2, G), Segment(G, ppp)]))
                    f.add_parent(
                        Fact("eqline",
                             [Segment(p, G), Segment(G, m3)]))
                    f.add_parent(
                        Fact("eqline", [Segment(pp, m3),
                                        Segment(ppp, m3)]))
                    f.add_parent(Fact("midp", [m1, s1]))
                    f.add_parent(Fact("midp", [m2, s2]))
                    facts.append(f)
    return facts


def midp_to_centri(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """The median theorem."""
    facts = []
    m3, s3 = fact.objects
    if s3 not in diagram.database.h_segments_centris:
        return facts
    for t in diagram.database.h_segments_centris[s3]:
        p, pp, ppp = t.name
        s1 = Segment(p, ppp)
        s2 = Segment(p, pp)
        m1 = diagram.database.inverse_midp[s1]
        m2 = diagram.database.inverse_midp[s2]
        G = diagram.database.centris[t]
        f = Fact("eqline", [Segment(p, G), Segment(G, m3)], "midp_to_centri")
        f.add_parent(fact)
        f.add_parent(Fact("midp", [m1, s1]))
        f.add_parent(Fact("midp", [m2, s2]))
        f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
        f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
        facts.append(f)
        f = Fact("eqratio", [
            Ratio(Segment(m1, G), Segment(G, pp)),
            Ratio(Segment(m3, G), Segment(G, p))
        ], "midp_to_centri")
        f.add_parent(fact)
        f.add_parent(Fact("midp", [m1, s1]))
        f.add_parent(Fact("midp", [m2, s2]))
        f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
        f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
        facts.append(f)
    return facts


def eqline_to_centri(diagram: 'Diagram', fact: Fact) -> List[Fact]:
    """The median theorem."""
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
    p_A, p_B, p_C = [diagram.point_dict[p] for p in [A, B, C]]
    if not (p_A < p_B < p_C or p_C < p_B < p_A):
        return facts
    s3 = Segment(A, C)
    if s3 in diagram.database.h_segments_centris:
        for t in diagram.database.h_segments_centris[s3]:
            p, pp, ppp = t.name
            s1 = Segment(p, ppp)
            s2 = Segment(p, pp)
            m1 = diagram.database.inverse_midp[s1]
            m2 = diagram.database.inverse_midp[s2]
            G = diagram.database.centris[t]
            if not diagram.database.is_eqline(Segment(B, G), Segment(G, p)):
                continue
            m3 = B
            f = Fact("midp", [m3, s3], "eqline_to_centri")
            f.add_parent(fact)
            f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
            f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
            f.add_parent(Fact("eqline", [Segment(m3, G), Segment(G, p)]))
            f.add_parent(Fact("midp", [m1, s1]))
            f.add_parent(Fact("midp", [m2, s2]))
            facts.append(f)
            f = Fact("eqratio", [
                Ratio(Segment(m1, G), Segment(G, pp)),
                Ratio(Segment(m3, G), Segment(G, p))
            ], "eqline_to_centri")
            f.add_parent(fact)
            f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
            f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
            f.add_parent(Fact("eqline", [Segment(m3, G), Segment(G, p)]))
            f.add_parent(Fact("midp", [m1, s1]))
            f.add_parent(Fact("midp", [m2, s2]))
            facts.append(f)
    if B in diagram.database.inverse_centris:
        G = B
        for t in diagram.database.inverse_centris[G]:
            _, pp, ppp = t.name
            s3 = Segment(pp, ppp)
            for p, m3 in [(A, C), (C, A)]:
                if not (p == t.name[0] and diagram.database.is_eqline(
                        Segment(m3, pp), Segment(m3, ppp))):
                    continue
                s1 = Segment(p, ppp)
                s2 = Segment(p, pp)
                m1 = diagram.database.inverse_midp[s1]
                m2 = diagram.database.inverse_midp[s2]
                f = Fact("midp", [m3, s3], "eqline_to_centri")
                f.add_parent(fact)
                f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
                f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
                f.add_parent(
                    Fact("eqline",
                         [Segment(m3, pp), Segment(m3, ppp)]))
                f.add_parent(Fact("midp", [m1, s1]))
                f.add_parent(Fact("midp", [m2, s2]))
                facts.append(f)
                f = Fact("eqratio", [
                    Ratio(Segment(m1, G), Segment(G, pp)),
                    Ratio(Segment(m3, G), Segment(G, p))
                ], "eqline_to_centri")
                f.add_parent(fact)
                f.add_parent(Fact("eqline", [Segment(m1, G), Segment(G, pp)]))
                f.add_parent(Fact("eqline", [Segment(m2, G), Segment(G, ppp)]))
                f.add_parent(
                    Fact("eqline",
                         [Segment(m3, pp), Segment(m3, ppp)]))
                f.add_parent(Fact("midp", [m1, s1]))
                f.add_parent(Fact("midp", [m2, s2]))
                facts.append(f)
    return facts
