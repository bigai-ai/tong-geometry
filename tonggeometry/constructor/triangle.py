r"""Define constructors for triangles."""

import itertools
from typing import TYPE_CHECKING, List, Tuple

from tonggeometry.constructor.parent import (CANNOT, EXIST, SAMELINE, SSUCCESS,
                                             Constructor)
from tonggeometry.constructor.primitives import (Line, Point,
                                                 intersection_of_lines,
                                                 on_same_line, parallel, perp)
from tonggeometry.inference_engine.predicate import Predicate
from tonggeometry.inference_engine.primitives import Segment

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram

__all__ = [
    "BisectorLine",
    "PerpendicularLine",
    "Centroid",
    "Orthocenter",
    "IsogonalConjugate",
    "InCenter",
]


class BisectorLine(Constructor):
    """Constructor for the internal angle bisector point.

    Total number of actions 3 * C_k^3.
    """
    from_names_len = 3
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C = froms
        BA = A - B
        unit_BA = BA * 10**3 / BA.norm()
        BC = C - B
        unit_BC = BC * 10**3 / BC.norm()
        mid = (unit_BA + unit_BC) / 2 + B
        l_CA = Line(C, A)
        l_midB = Line(mid, B)
        P = intersection_of_lines(l_CA, l_midB)
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names

        A, B, C = from_names

        diagram.add_point(P, p)

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + A)
        diagram.add_line_to_draw(C + B)
        diagram.add_line_to_draw(P + B)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C = from_names
        P = to_names
        return [
            Predicate("eqangle", [A, B, P, P, B, C]),
            Predicate("eqline", [A, C, A, P])
        ]

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        froms = [diagram.point_dict[name] for name in from_names]
        if on_same_line(*froms):
            return False, SAMELINE

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_three_points = []
        for combo in itertools.combinations(diagram.parent_points, 2):
            for p in new_points:
                three_tuple = combo + (p, )
                list_of_three_points.append(
                    (three_tuple[1], three_tuple[0], three_tuple[2]))
                list_of_three_points.append(three_tuple)
                list_of_three_points.append(
                    (three_tuple[0], three_tuple[2], three_tuple[1]))
        for p in diagram.parent_points:
            for combo in itertools.combinations(new_points, 2):
                three_tuple = (p, ) + combo
                list_of_three_points.append(
                    (three_tuple[1], three_tuple[0], three_tuple[2]))
                list_of_three_points.append(three_tuple)
                list_of_three_points.append(
                    (three_tuple[0], three_tuple[2], three_tuple[1]))
        for three_tuple in itertools.combinations(new_points, 3):
            list_of_three_points.append(
                (three_tuple[1], three_tuple[0], three_tuple[2]))
            list_of_three_points.append(three_tuple)
            list_of_three_points.append(
                (three_tuple[0], three_tuple[2], three_tuple[1]))
        return list_of_three_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        if from_names[0] > from_names[2]:
            new_from_names = from_names[::-1]
        else:
            new_from_names = from_names
        return new_from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        num_new_A = sum(int(name in new_points) for name in from_names_A)
        num_new_B = sum(int(name in new_points) for name in from_names_B)
        if num_new_A != num_new_B:
            return num_new_A < num_new_B
        sort_A = "".join(sorted(from_names_A))
        sort_B = "".join(sorted(from_names_B))
        if sort_A != sort_B:
            return sort_A < sort_B
        return from_names_A[1] < from_names_B[1]


class PerpendicularLine(Constructor):
    """Constructor for the perpendicular line for a triangle.

    Total number of actions 3 * C_k^3.
    """
    from_names_len = 3
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C = froms
        l_CA = Line(C, A)
        l_perpB = Line(l_CA.ny, -l_CA.nx, l_CA.nx * B.y - l_CA.ny * B.x)
        P = intersection_of_lines(l_CA, l_perpB)
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names

        A, B, C = from_names

        diagram.add_point(P, p)

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + A)
        diagram.add_line_to_draw(P + A)
        diagram.add_line_to_draw(C + B)
        diagram.add_line_to_draw(P + B)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C = from_names
        P = to_names
        return [
            Predicate("perp", [A, P, B]),
            Predicate("eqline", [A, C, A, P])
        ]

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, C = from_names
        l_AC = Segment(A, C)
        l_AC_rep = diagram.database.inverse_eqline[l_AC]
        it = iter(diagram.database.lines_points[l_AC_rep])
        pA = next(it)
        pC = next(it)
        pB = B
        if all(p in diagram.parent_points for p in [pA, pB, pC]) or any(
                x != y for x, y in zip([A, C], [pA, pC])):
            return False, EXIST
        return True, SSUCCESS

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        froms = [diagram.point_dict[name] for name in from_names]
        if on_same_line(*froms):
            return False, SAMELINE

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_three_points = []
        for combo in itertools.combinations(diagram.parent_points, 2):
            for p in new_points:
                three_tuple = combo + (p, )
                list_of_three_points.append(
                    (three_tuple[1], three_tuple[0], three_tuple[2]))
                list_of_three_points.append(three_tuple)
                list_of_three_points.append(
                    (three_tuple[0], three_tuple[2], three_tuple[1]))
        for p in diagram.parent_points:
            for combo in itertools.combinations(new_points, 2):
                three_tuple = (p, ) + combo
                list_of_three_points.append(
                    (three_tuple[1], three_tuple[0], three_tuple[2]))
                list_of_three_points.append(three_tuple)
                list_of_three_points.append(
                    (three_tuple[0], three_tuple[2], three_tuple[1]))
        for three_tuple in itertools.combinations(new_points, 3):
            list_of_three_points.append(
                (three_tuple[1], three_tuple[0], three_tuple[2]))
            list_of_three_points.append(three_tuple)
            list_of_three_points.append(
                (three_tuple[0], three_tuple[2], three_tuple[1]))
        return list_of_three_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C = from_names
        l_AC = Segment(A, C)
        l_AC_rep = diagram.database.inverse_eqline[l_AC]
        it = iter(diagram.database.lines_points[l_AC_rep])
        pA = next(it)
        pC = next(it)
        pB = B
        new_from_names = min(pA, pC) + pB + max(pA, pC)
        return new_from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        num_new_A = sum(int(name in new_points) for name in from_names_A)
        num_new_B = sum(int(name in new_points) for name in from_names_B)
        if num_new_A != num_new_B:
            return num_new_A < num_new_B
        sort_A = "".join(sorted(from_names_A))
        sort_B = "".join(sorted(from_names_B))
        if sort_A != sort_B:
            return sort_A < sort_B
        return from_names_A[1] < from_names_B[1]


# The computation methods are from
# https://web.evanchen.cc/handouts/bary/bary-full.pdf
# https://web.math.sinica.edu.tw/math_media/d341/34103.pdf


class Centroid(Constructor):
    """Constructor for the centroid of a triangle.

    Total number of actions C_k^3.
    """
    from_names_len = 3
    to_names_len = 4

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C = froms
        M = (A + B + C) / 3
        D = 0.5 * A + 0.5 * B
        E = 0.5 * A + 0.5 * C
        F = 0.5 * B + 0.5 * C
        return M, D, E, F

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        for p, name in zip(to, to_names):
            diagram.add_point(name, p)

        A, B, C = from_names
        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + A)
        diagram.add_line_to_draw(C + B)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        A, B, C = from_names
        M, D, E, F = to_names
        return [
            Predicate("midp", [D, A, B]),
            Predicate("midp", [E, A, C]),
            Predicate("midp", [F, B, C]),
            Predicate("eqline", [C, M, C, D]),
            Predicate("eqline", [B, M, B, E]),
            Predicate("eqline", [A, M, A, F]),
            Predicate("eqratio", [A, M, M, F, B, M, M, E]),
            Predicate("eqratio", [A, M, M, F, C, M, M, D])
        ]

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        froms = [diagram.point_dict[name] for name in from_names]
        if on_same_line(*froms):
            return False, SAMELINE

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_three_points = []
        for combo in itertools.combinations(diagram.parent_points, 2):
            for p in new_points:
                three_tuple = combo + (p, )
                list_of_three_points.append(three_tuple)
        for p in diagram.parent_points:
            for combo in itertools.combinations(new_points, 2):
                three_tuple = (p, ) + combo
                list_of_three_points.append(three_tuple)
        list_of_three_points += itertools.combinations(new_points, 3)
        return list_of_three_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C = from_names
        P, D, E, F = to_names
        mapping = {
            min(A + B, B + A): D,
            min(A + C, C + A): E,
            min(B + C, C + B): F
        }
        new_from_names = "".join(sorted(from_names))
        new_to_names = P + mapping[min(
            new_from_names[0] + new_from_names[1],
            new_from_names[1] + new_from_names[0])] + mapping[min(
                new_from_names[0] + new_from_names[2],
                new_from_names[2] + new_from_names[0])] + mapping[min(
                    new_from_names[1] + new_from_names[2],
                    new_from_names[2] + new_from_names[1])]
        return new_from_names, new_to_names


class Orthocenter(Constructor):
    """Constructor for the orthocenter of a triangle.

    Total number of actions C_k^3.
    """
    from_names_len = 3
    to_names_len = 4

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C = froms
        BC = C - B
        a = BC.norm()
        CA = A - C
        b = CA.norm()
        AB = B - A
        c = AB.norm()
        area = 0.5 * abs(AB.cross(CA))
        coeff_A = (a**4 - (b**2 - c**2)**2) / (16 * area**2)
        coeff_B = (b**4 - (c**2 - a**2)**2) / (16 * area**2)
        coeff_C = (c**4 - (a**2 - b**2)**2) / (16 * area**2)
        H = coeff_A * A + coeff_B * B + coeff_C * C
        l_HC = Line(H, C)
        l_BA = Line(B, A)
        D = intersection_of_lines(l_HC, l_BA)
        l_HB = Line(H, B)
        l_CA = Line(C, A)
        E = intersection_of_lines(l_HB, l_CA)
        l_HA = Line(H, A)
        l_CB = Line(C, B)
        F = intersection_of_lines(l_HA, l_CB)
        return (H, D, E, F)

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        for p, name in zip(to, to_names):
            diagram.add_point(name, p)

        A, B, C = from_names
        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + A)
        diagram.add_line_to_draw(C + B)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C = from_names
        H, D, E, F = to_names
        return [
            Predicate("perp", [A, D, H]),
            Predicate("perp", [B, F, H]),
            Predicate("perp", [C, E, H]),
            Predicate("eqline", [A, D, A, B]),
            Predicate("eqline", [B, F, B, C]),
            Predicate("eqline", [C, E, C, A]),
            Predicate("eqline", [A, H, A, F]),
            Predicate("eqline", [B, H, B, E]),
            Predicate("eqline", [C, H, C, D])
        ]

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        froms = [diagram.point_dict[name] for name in from_names]
        A, B, C = froms
        if on_same_line(*froms) or perp(Line(B, A), Line(C, A)) or perp(
                Line(A, B), Line(C, B)) or perp(Line(A, C), Line(B, C)):
            return False, SAMELINE

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_three_points = []
        for combo in itertools.combinations(diagram.parent_points, 2):
            for p in new_points:
                three_tuple = combo + (p, )
                list_of_three_points.append(three_tuple)
        for p in diagram.parent_points:
            for combo in itertools.combinations(new_points, 2):
                three_tuple = (p, ) + combo
                list_of_three_points.append(three_tuple)
        list_of_three_points += itertools.combinations(new_points, 3)
        return list_of_three_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C = from_names
        P, D, E, F = to_names
        mapping = {
            min(A + B, B + A): D,
            min(A + C, C + A): E,
            min(B + C, C + B): F
        }
        new_from_names = "".join(sorted(from_names))
        new_to_names = P + mapping[min(
            new_from_names[0] + new_from_names[1],
            new_from_names[1] + new_from_names[0])] + mapping[min(
                new_from_names[0] + new_from_names[2],
                new_from_names[2] + new_from_names[0])] + mapping[min(
                    new_from_names[1] + new_from_names[2],
                    new_from_names[2] + new_from_names[1])]
        return new_from_names, new_to_names


class IsogonalConjugate(Constructor):
    """Constructor for the isogonal conjugate."""
    from_names_len = 4
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C, Q = froms
        BC = C - B
        a = BC.norm()
        CA = A - C
        b = CA.norm()
        AB = B - A
        c = AB.norm()
        I = (a * A + b * B + c * C) / (a + b + c)
        l_AI = Line(I, A)
        l_BI = Line(I, B)
        AQ = Q - A
        BQ = Q - B
        Q_AI = AQ - 2 * AQ.vector.dot(l_AI.norm) * l_AI.norm + A
        Q_BI = BQ - 2 * BQ.vector.dot(l_BI.norm) * l_BI.norm + B
        l_Q_AI = Line(Q_AI, A)
        l_Q_BI = Line(Q_BI, B)
        P = intersection_of_lines(l_Q_AI, l_Q_BI)
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names
        diagram.add_point(P, p)

        A, B, C, Q = from_names

        diagram.add_line_to_draw(Q + A)
        diagram.add_line_to_draw(Q + B)
        diagram.add_line_to_draw(Q + C)
        diagram.add_line_to_draw(P + A)
        diagram.add_line_to_draw(P + B)
        diagram.add_line_to_draw(P + C)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        A, B, C, Q = from_names
        P = to_names
        return [
            Predicate("eqangle", [B, A, Q, P, A, C]),
            Predicate("eqangle", [A, B, Q, P, B, C]),
            Predicate("eqangle", [B, C, Q, P, C, A]),
        ]

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        froms = [diagram.point_dict[name] for name in from_names]
        A, B, C, Q = froms
        if on_same_line(A, B, C) or on_same_line(Q, A, B) or on_same_line(
                Q, A, C) or on_same_line(Q, B, C):
            return False, SAMELINE
        BC = C - B
        a = BC.norm()
        CA = A - C
        b = CA.norm()
        AB = B - A
        c = AB.norm()
        I = (a * A + b * B + c * C) / (a + b + c)
        l_AI = Line(I, A)
        l_BI = Line(I, B)
        AQ = Q - A
        BQ = Q - B
        Q_AI = AQ - 2 * AQ.vector.dot(l_AI.norm) * l_AI.norm + A
        Q_BI = BQ - 2 * BQ.vector.dot(l_BI.norm) * l_BI.norm + B
        l_Q_AI = Line(Q_AI, A)
        l_Q_BI = Line(Q_BI, B)
        if parallel(l_Q_AI, l_Q_BI):
            return False, CANNOT

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_four_points = []
        for olds in itertools.combinations(diagram.parent_points, 3):
            for new in new_points:
                list_of_four_points.append((olds[0], olds[1], olds[2], new))
                list_of_four_points.append((olds[0], olds[1], new, olds[2]))
                list_of_four_points.append((olds[0], olds[2], new, olds[1]))
                list_of_four_points.append((olds[1], olds[2], new, olds[0]))
        for olds in itertools.combinations(diagram.parent_points, 2):
            for news in itertools.combinations(new_points, 2):
                list_of_four_points.append(
                    (olds[0], olds[1], news[0], news[1]))
                list_of_four_points.append(
                    (olds[0], olds[1], news[1], news[0]))
                list_of_four_points.append(
                    (olds[0], news[0], news[1], olds[1]))
                list_of_four_points.append(
                    (olds[1], news[0], news[1], olds[0]))
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 3):
                list_of_four_points.append((old, news[0], news[1], news[2]))
                list_of_four_points.append((old, news[0], news[2], news[1]))
                list_of_four_points.append((old, news[1], news[2], news[0]))
                list_of_four_points.append((news[0], news[1], news[2], old))
        for news in itertools.combinations(new_points, 4):
            list_of_four_points.append((news[0], news[1], news[2], news[3]))
            list_of_four_points.append((news[0], news[1], news[3], news[2]))
            list_of_four_points.append((news[0], news[2], news[3], news[1]))
            list_of_four_points.append((news[1], news[2], news[3], news[0]))
        return list_of_four_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        new_from_names = "".join(sorted(from_names[:3])) + from_names[3]
        return new_from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        num_new_A = sum(int(name in new_points) for name in from_names_A)
        num_new_B = sum(int(name in new_points) for name in from_names_B)
        if num_new_A != num_new_B:
            return num_new_A < num_new_B
        sort_A = "".join(sorted(from_names_A))
        sort_B = "".join(sorted(from_names_B))
        if sort_A != sort_B:
            return sort_A < sort_B
        return from_names_A < from_names_B


class InCenter(Constructor):
    """Constructor for the incenter for a triangle.

    Total number of actions C_k^3.
    """
    from_names_len = 3
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C = froms
        BC = C - B
        a = BC.norm()
        CA = A - C
        b = CA.norm()
        AB = B - A
        c = AB.norm()
        center = (a * A + b * B + c * C) / (a + b + c)
        return (center, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        A, B, C = from_names
        center = to[0]

        Center = to_names

        diagram.add_point(Center, center)

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + A)
        diagram.add_line_to_draw(C + B)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C = from_names
        I = to_names
        return [
            Predicate("eqangle", [A, B, I, I, B, C]),
            Predicate("eqangle", [B, C, I, I, C, A]),
            Predicate("eqangle", [C, A, I, I, A, B])
        ]

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        froms = [diagram.point_dict[name] for name in from_names]
        if on_same_line(*froms):
            return False, SAMELINE

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_three_points = []
        for combo in itertools.combinations(diagram.parent_points, 2):
            for p in new_points:
                three_tuple = combo + (p, )
                list_of_three_points.append(three_tuple)
        for p in diagram.parent_points:
            for combo in itertools.combinations(new_points, 2):
                three_tuple = (p, ) + combo
                list_of_three_points.append(three_tuple)
        list_of_three_points += itertools.combinations(new_points, 3)
        return list_of_three_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        new_from_names = "".join(sorted(from_names))
        return new_from_names, to_names


# class Circumcenter(Constructor):
#     """Constructor for the circumcenter of a triangle."""

#     @staticmethod
#     def compute(froms: List[Point]) -> Point:
#         A, B, C = froms
#         BC = C - B
#         a = BC.norm()
#         CA = A - C
#         b = CA.norm()
#         AB = B - A
#         c = AB.norm()
#         area = 0.5 * abs(AB.cross(CA))
#         coeff_A = a**2 * (b**2 + c**2 - a**2) / (16 * area**2)
#         coeff_B = b**2 * (c**2 + a**2 - b**2) / (16 * area**2)
#         coeff_C = c**2 * (a**2 + b**2 - c**2) / (16 * area**2)
#         return coeff_A * A + coeff_B * B + coeff_C * C

#     @staticmethod
#     def construct(diagram: 'Diagram',
#                   from_names: str,
#                   to_names: str,
#                   output: Optional[Tuple] = None):
#         if not output:
#             valid, output, message_id = Circumcenter.static_check(
#                 diagram, from_names, to_names)
#             if not valid:
#                 raise AssertionError(ErrorMessages[message_id])
#             to, _ = output
#         else:
#             to, _ = output
#             to = copy.deepcopy(to)

#         to.name = to_names

#         diagram.add_point(to)
#         diagram.lines_from_point(to)

#         return to.name

#     @staticmethod
#     def given_facts(from_names: str, to_names: str) -> List[Predicate]:
#         A, B, C = from_names
#         O = to_names
#         return [Predicate("circle", [O, A, B, C])]

#     @staticmethod
#     def static_check(
#             diagram: 'Diagram',
#             from_names: str,
#             to_names: Optional[str] = None) -> Tuple[bool, Optional[Tuple],
#                                                      str]:

#         # check input
#         if len(from_names) != 3 or len(set(from_names)) != 3:
#             return False, None, DISTINCT
#         if to_names and len(to_names) != 1:
#             return False, None, INVALID
#         if not all(p_name in diagram.point_dict
#                    for p_name in from_names):
#             return False, None, POINT
#         if to_names and to_names in diagram.point_dict:
#             return False, None, USED

#         # check if can be drawn
#         froms = [diagram.point_dict[name] for name in from_names]
#         if on_same_line(*froms):
#             return False, None, SAMELINE

#         # check existing facts
#         to = Circumcenter.compute(froms)
#         if to in diagram.point_dict.values():
#             return False, None, EXIST

#         return True, (to, froms), SUCCESS

#     @staticmethod
#     def new_from_names(diagram: 'Diagram',
#                       new_points: Optional[str] = None) -> List[Action]:
#         valid = []
#         if new_points is not None:
#             if not new_points:
#                 return []
#             list_of_three_points = []
#             list_of_three_points += itertools.combinations(new_points, 3)
#             for combo in itertools.combinations(new_points, 2):
#                 for p in diagram.parent_points:
#                     three_tuple = (p, ) + combo
#                     list_of_three_points.append(three_tuple)
#             for combo in itertools.combinations(diagram.parent_points, 2):
#                 for p in new_points:
#                     three_tuple = combo + (p, )
#                     list_of_three_points.append(three_tuple)
#         else:
#             list_of_three_points = itertools.combinations(
#                 diagram.point_dict.keys(), 3)
#         for three_points in list_of_three_points:
#             from_names = "".join(three_points)
#             go, output, _ = Circumcenter.static_check(diagram, from_names)
#             if go:
#                 valid.append(
#                     Action(Circumcenter,
#                            from_names,
#                            to_names_len=1,
#                            output=output))
#         return valid

# class Excenter(Constructor):
#     """Constructor for the excenter of a triangle."""

#     @staticmethod
#     def compute(froms: List[Point]) -> Point:
#         A, B, C = froms
#         BC = C - B
#         a = BC.norm()
#         CA = A - C
#         b = CA.norm()
#         AB = B - A
#         c = AB.norm()
#         return (a * A - b * B + c * C) / (a - b + c)

#     @staticmethod
#     def construct(diagram: 'Diagram',
#                   from_names: str,
#                   to_names: str,
#                   output: Optional[Tuple] = None):
#         if not output:
#             valid, output, message_id = Excenter.static_check(
#                 diagram, from_names, to_names)
#             if not valid:
#                 raise AssertionError(ErrorMessages[message_id])
#             to, _ = output
#         else:
#             to, _ = output
#             to = copy.deepcopy(to)

#         to.name = to_names

#         diagram.add_point(to)
#         diagram.lines_from_point(to)

#     @staticmethod
#     def given_facts(from_names: str, to_names: str) -> List[Predicate]:
#         return []

#     @staticmethod
#     def static_check(
#             diagram: 'Diagram',
#             from_names: str,
#             to_names: Optional[str] = None) -> Tuple[bool, Optional[Tuple],
#                                                      str]:

#         # check input
#         if len(from_names) != 3 or len(set(from_names)) != 3:
#             return False, None, DISTINCT
#         if to_names and len(to_names) != 1:
#             return False, None, INVALID
#         if not all(p_name in diagram.point_dict
#                    for p_name in from_names):
#             return False, None, POINT
#         if to_names and to_names in diagram.point_dict:
#             return False, None, USED

#         # check if can be drawn
#         froms = [diagram.point_dict[name] for name in from_names]
#         if on_same_line(*froms):
#             return False, None, SAMELINE

#         # check existing facts
#         to = Excenter.compute(froms)
#         if to in diagram.point_dict.values():
#             return False, None, EXIST

#         return True, (to, froms), SUCCESS

#     @staticmethod
#     def new_from_names(diagram: 'Diagram',
#                       new_points: Optional[str] = None) -> List[Action]:
#         valid = []
#         if new_points is not None:
#             if not new_points:
#                 return []
#             list_of_three_points = []
#             for three_tuple in itertools.combinations(new_points, 3):
#                 list_of_three_points.append(three_tuple)
#                 list_of_three_points.append(
#                     (three_tuple[0], three_tuple[2], three_tuple[1]))
#                 list_of_three_points.append(
#                     (three_tuple[1], three_tuple[0], three_tuple[2]))
#             for combo in itertools.combinations(new_points, 2):
#                 for p in diagram.parent_points:
#                     three_tuple = (p, ) + combo
#                     list_of_three_points.append(three_tuple)
#                     list_of_three_points.append(
#                         (three_tuple[0], three_tuple[2], three_tuple[1]))
#                     list_of_three_points.append(
#                         (three_tuple[1], three_tuple[0], three_tuple[2]))
#             for combo in itertools.combinations(diagram.parent_points, 2):
#                 for p in new_points:
#                     three_tuple = combo + (p, )
#                     list_of_three_points.append(three_tuple)
#                     list_of_three_points.append(
#                         (three_tuple[0], three_tuple[2], three_tuple[1]))
#                     list_of_three_points.append(
#                         (three_tuple[1], three_tuple[0], three_tuple[2]))
#         else:
#             list_of_three_points = []
#             for three_tuple in itertools.combinations(
#                     diagram.point_dict.keys(), 3):
#                 list_of_three_points.append(three_tuple)
#                 list_of_three_points.append(
#                     (three_tuple[0], three_tuple[2], three_tuple[1]))
#                 list_of_three_points.append(
#                     (three_tuple[1], three_tuple[0], three_tuple[2]))
#         for three_points in list_of_three_points:
#             from_names = "".join(three_points)
#             go, output, _ = Excenter.static_check(diagram, from_names)
#             if go:
#                 valid.append(
#                     Action(Excenter, from_names, to_names_len=1,
#                            output=output))
#         return valid

# class EqualAngles(Constructor):
#     """Find point P in the interior such that PAB = PCB.

#     From [A, B, C], find point P such that P in the interior of ABC
#     angle(P,A,B) = angle(P,C,B).

#     Total number of actions 3 * C_k^3.
#     """

#     @staticmethod
#     def compute(froms: List[Point]) -> Tuple[Point]:
#         A, B, C = froms

#         AB = B - A
#         AC = C - A
#         angle_BAC = get_angle(AB, AC)

#         CB = B - C
#         CA = A - C
#         angle_BCA = get_angle(CB, CA)

#         # angle randomization
#         # note: remove randomization here to avoid errors of
#         # follow up steps
#         alpha = 0.3 * min(angle_BAC, angle_BCA)

#         cos_cw, sin_cw = math.cos(-alpha), math.sin(-alpha)
#         rotation_cw = np.array([[cos_cw, -sin_cw], [sin_cw, cos_cw]])
#         cos_ccw, sin_ccw = math.cos(alpha), math.sin(alpha)
#         rotation_ccw = np.array([[cos_ccw, -sin_ccw], [sin_ccw, cos_ccw]])

#         if AB.cross(AC) > 0:  # C to the left of AB
#             A_p = Point(np.dot(rotation_ccw, AB.vector) + A.vector)
#             C_p = Point(np.dot(rotation_cw, CB.vector) + C.vector)
#         else:
#             A_p = Point(np.dot(rotation_cw, AB.vector) + A.vector)
#             C_p = Point(np.dot(rotation_ccw, CB.vector) + C.vector)
#         l_1 = Line(A_p, A)
#         l_2 = Line(C_p, C)

#         P = intersection_of_lines(l_1, l_2)
#         return (P, )

#     @staticmethod
#     def construct(diagram: 'Diagram', from_names: str, to_names: str,
#                   output: Tuple):
#         to = output
#         p = to[0]

#         P = to_names

#         A, B, C = from_names

#         diagram.add_point(P, p)

#         diagram.add_line_to_draw(B + A)
#         diagram.add_line_to_draw(C + B)
#         diagram.add_line_to_draw(P + A)
#         diagram.add_line_to_draw(P + C)

#     @staticmethod
#     def given_facts(from_names: str, to_names: str) -> List[Predicate]:
#         A, B, C = from_names
#         P = to_names

#         return [Predicate("eqangle", [B, A, P, P, C, B])]

#     @staticmethod
#     def static_check(diagram: 'Diagram',
#                      from_names: str,
#                      to_names: Optional[str] = None) -> Tuple[bool, int]:

#         # check input
#         if len(from_names) != 3 or len(set(from_names)) != 3:
#             return False, DISTINCT
#         if to_names and len(to_names) != 1:
#             return False, INVALID
#         if not all(p_name in diagram.point_dict for p_name in from_names):
#             return False, POINT
#         if to_names and to_names in diagram.point_dict:
#             return False, USED

#         # check if can be drawn
#         froms = [diagram.point_dict[name] for name in from_names]
#         if on_same_line(*froms):
#             return False, SAMELINE

#         return True, SSUCCESS

#     @staticmethod
#     def runtime_check(diagram: 'Diagram',
#                       to: Tuple) -> Tuple[Optional[Tuple], int, int]:
#         p = to[0]
#         if p in diagram.point_dict.values():
#             return None, 0, EXIST

#         return to, 1, RSUCCESS

#     @staticmethod
#     def new_from_names(diagram: 'Diagram',
#                       new_points: Optional[str] = None) -> List[Action]:
#         valid = []
#         if new_points is not None:
#             if not new_points:
#                 return []
#             list_of_three_points = []
#             for three_tuple in itertools.combinations(new_points, 3):
#                 list_of_three_points.append(three_tuple)
#                 list_of_three_points.append(
#                     (three_tuple[0], three_tuple[2], three_tuple[1]))
#                 list_of_three_points.append(
#                     (three_tuple[1], three_tuple[0], three_tuple[2]))
#             for combo in itertools.combinations(new_points, 2):
#                 for p in diagram.parent_points:
#                     three_tuple = (p, ) + combo
#                     list_of_three_points.append(three_tuple)
#                     list_of_three_points.append(
#                         (three_tuple[0], three_tuple[2], three_tuple[1]))
#                     list_of_three_points.append(
#                         (three_tuple[1], three_tuple[0], three_tuple[2]))
#             for combo in itertools.combinations(diagram.parent_points, 2):
#                 for p in new_points:
#                     three_tuple = combo + (p, )
#                     list_of_three_points.append(three_tuple)
#                     list_of_three_points.append(
#                         (three_tuple[0], three_tuple[2], three_tuple[1]))
#                     list_of_three_points.append(
#                         (three_tuple[1], three_tuple[0], three_tuple[2]))
#         else:
#             list_of_three_points = []
#             for three_tuple in itertools.combinations(
#                     diagram.point_dict.keys(), 3):
#                 list_of_three_points.append(three_tuple)
#                 list_of_three_points.append(
#                     (three_tuple[0], three_tuple[2], three_tuple[1]))
#                 list_of_three_points.append(
#                     (three_tuple[1], three_tuple[0], three_tuple[2]))
#         for three_points in list_of_three_points:
#             from_names = "".join(three_points)
#             sgo, _ = EqualAngles.static_check(diagram, from_names)
#             if sgo:
#                 froms = [diagram.point_dict[name] for name in from_names]
#                 to = EqualAngles.compute(froms)
#                 valid.append(Action(EqualAngles, from_names, output=to))
#         return valid
