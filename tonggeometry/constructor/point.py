r"""Define constructors for points."""

import itertools
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from tonggeometry.constructor.parent import (CANNOT, DISTINCT, EXIST, INVALID,
                                             NOINTERSECT, POINT, RSUCCESS,
                                             SAMELINE, SSUCCESS, USED,
                                             Constructor)
from tonggeometry.constructor.primitives import (Circle, Line, Point,
                                                 intersection_of_circles,
                                                 intersection_of_line_circle,
                                                 intersection_of_lines,
                                                 on_same_line, parallel)
from tonggeometry.inference_engine.predicate import Predicate
from tonggeometry.inference_engine.primitives import Circle as LogicCircle
from tonggeometry.inference_engine.primitives import Segment
from tonggeometry.util import isclose

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram

__all__ = [
    "ExtendEqual", "MidPoint", "AnyPoint", "MidArc", "AnyArc",
    "IntersectLineLine", "IntersectCircleCircle", "IntersectLineCircleOn",
    "IntersectLineCircleOff", "Reflect", "InSimiliCenter", "ExSimiliCenter"
]


def key_func(x):
    """Joining everything in x to be its key."""
    return "".join(x)


class ExtendEqual(Constructor):
    """Extend a line segment to equal length.

    Find C with A,B s.t. CB = AB, B is midpoint of A,C.

    Total number of actions A_k^2.
    """
    from_names_len = 2
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B = froms
        P = 2 * B - A
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names

        A = from_names[0]

        diagram.add_point(P, p)

        diagram.add_line_to_draw(P + A)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B = from_names
        P = to_names
        return [Predicate("midp", [B, A, P])]

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_two_points = []
        for old_p in diagram.parent_points:
            for new_p in new_points:
                list_of_two_points.append((old_p, new_p))
        for new_p in new_points:
            for old_p in diagram.parent_points:
                list_of_two_points.append((new_p, old_p))
        list_of_two_points += itertools.permutations(new_points, 2)
        return list_of_two_points


class MidPoint(Constructor):
    """Mid point constructor.

    Total number of actions C_k^2.
    """
    from_names_len = 2
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B = froms
        P = 0.5 * A + 0.5 * B
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names

        A, B = from_names

        diagram.add_point(P, p)

        diagram.add_line_to_draw(B + A)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B = from_names
        M = to_names
        return [Predicate("midp", [M, A, B])]

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_two_points = []
        for old_p in diagram.parent_points:
            for new_p in new_points:
                list_of_two_points.append((old_p, new_p))
        list_of_two_points += itertools.combinations(new_points, 2)
        return list_of_two_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        if from_names[0] > from_names[1]:
            new_from_names = from_names[::-1]
        else:
            new_from_names = from_names
        return new_from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        return from_names_A < from_names_B


class AnyPoint(Constructor):
    """Any point constructor.

    Total number of actions C_k^2.
    """
    from_names_len = 2
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B = froms
        if B < A:
            A, B = B, A
        P = 0.7 * A + 0.3 * B
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names

        A, B = from_names

        diagram.add_point(P, p)

        diagram.add_line_to_draw(B + A)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B = from_names
        P = to_names
        return [Predicate("eqline", [P, A, P, B])]

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_two_points = []
        for old_p in diagram.parent_points:
            for new_p in new_points:
                list_of_two_points.append((old_p, new_p))
        list_of_two_points += itertools.combinations(new_points, 2)
        return list_of_two_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        if from_names[0] > from_names[1]:
            new_from_names = from_names[::-1]
        else:
            new_from_names = from_names
        return new_from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        return from_names_A < from_names_B


class MidArc(Constructor):
    """Mid arc constructor. Directional, counter-clockwise."""
    from_names_len = 3
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C = froms
        r = np.linalg.norm(A.vector - C.vector)
        CA = A - C
        CB = B - C
        if isclose(CA.cross(CB), 0):
            P = Point(-CA.y, CA.x) + C
        else:
            dir_vec = CA + CB
            unit_vec = dir_vec / dir_vec.norm()
            vec = r * unit_vec
            if CA.cross(CB) > 0:
                P = C + vec
            else:
                P = C - vec
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names
        diagram.add_point(P, p)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        A, B, O = from_names
        P = to_names
        if A < B:
            circle_pred = Predicate("eqcircle", [[O, A], [None, A, B, P]])
        else:
            circle_pred = Predicate("eqcircle", [[O, B], [None, B, A, P]])
        return [
            circle_pred,
            Predicate("cong", [A, P, P, B]),
            Predicate("eqangle", [A, O, P, P, O, B]),
        ]

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        if not draw_only:
            logic_CA = LogicCircle(from_names[2], [from_names[0]])
            logic_CB = LogicCircle(from_names[2], [from_names[1]])
            if not diagram.database.is_eqcircle(logic_CA, logic_CB):
                return False, CANNOT

        A, B, C = [diagram.point_dict[name] for name in from_names]
        c_CA = Circle(C, A)

        if B not in c_CA:
            return False, CANNOT

        found = False
        for name, c in diagram.circle_dict.items():
            logic_c = LogicCircle(name[0], [name[1]])
            if c == c_CA and (draw_only or diagram.database.is_eqcircle(
                    logic_c, logic_CA)):
                found = True
                break
        if not found:
            return False, CANNOT

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_three_points = []
        for num_new in range(1, 4):
            num_old = 3 - num_new
            from_old = itertools.combinations(diagram.parent_points, num_old)
            from_new = itertools.combinations(new_points, num_new)
            combs = itertools.product(from_old, from_new)
            for comb in combs:
                list_of_three_points += itertools.permutations(
                    comb[0] + comb[1], 3)
        return list_of_three_points

    @staticmethod
    def new_from_circle(diagram: 'Diagram', circle: str = "") -> List[Tuple]:
        list_of_three_points = []
        logic_circle = LogicCircle(circle[0], [circle[1]])
        if logic_circle not in diagram.database.inverse_eqcircle:
            return list_of_three_points
        rep = diagram.database.inverse_eqcircle[logic_circle]
        circle_all = diagram.database.circles_circles[rep]
        points_sorted = sorted(circle_all.points)
        pairs = []
        for p1, p2 in itertools.combinations(points_sorted, 2):
            pairs.append(p1 + p2)
            pairs.append(p2 + p1)
        pairs_sorted = sorted(pairs)
        for pair_sorted in pairs_sorted:
            list_of_three_points.append(
                (pair_sorted[0], pair_sorted[1], circle[0]))
        return list_of_three_points

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        if new_points and len(new_points) > 0:
            # the new_from_names case
            num_new_A = sum(int(name in new_points) for name in from_names_A)
            num_new_B = sum(int(name in new_points) for name in from_names_B)
            if num_new_A != num_new_B:
                return num_new_A < num_new_B
            sort_A = "".join(sorted(from_names_A))
            sort_B = "".join(sorted(from_names_B))
            if sort_A != sort_B:
                return sort_A < sort_B
            return from_names_A < from_names_B
        # the new_from_circle case
        return from_names_A < from_names_B


class AnyArc(Constructor):
    """Any arc constructor. Directional, counter-clockwise."""
    from_names_len = 3
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C = froms
        CA = A - C
        CB = B - C
        ccw_angle = np.arctan2(np.linalg.det([CA.vector, CB.vector]),
                               (CA.vector * CB.vector).sum())
        if ccw_angle < 0:
            ccw_angle += 2 * np.pi
        angle = 0.37 * ccw_angle
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        P = R * CA + C
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names
        diagram.add_point(P, p)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        A, B, O = from_names
        P = to_names
        if A < B:
            circle_pred = Predicate("eqcircle", [[O, A], [None, A, B, P]])
        else:
            circle_pred = Predicate("eqcircle", [[O, B], [None, B, A, P]])
        return [circle_pred]

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        if not draw_only:
            logic_CA = LogicCircle(from_names[2], [from_names[0]])
            logic_CB = LogicCircle(from_names[2], [from_names[1]])
            if not diagram.database.is_eqcircle(logic_CA, logic_CB):
                return False, CANNOT

        A, B, C = [diagram.point_dict[name] for name in from_names]
        c_CA = Circle(C, A)

        if B not in c_CA:
            return False, CANNOT

        found = False
        for name, c in diagram.circle_dict.items():
            logic_c = LogicCircle(name[0], [name[1]])
            if c == c_CA and (draw_only or diagram.database.is_eqcircle(
                    logic_c, logic_CA)):
                found = True
                break
        if not found:
            return False, CANNOT

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_three_points = []
        for num_new in range(1, 4):
            num_old = 3 - num_new
            from_old = itertools.combinations(diagram.parent_points, num_old)
            from_new = itertools.combinations(new_points, num_new)
            combs = itertools.product(from_old, from_new)
            for comb in combs:
                list_of_three_points += itertools.permutations(
                    comb[0] + comb[1], 3)
        return list_of_three_points

    @staticmethod
    def new_from_circle(diagram: 'Diagram', circle: str = "") -> List[Tuple]:
        list_of_three_points = []
        logic_circle = LogicCircle(circle[0], [circle[1]])
        if logic_circle not in diagram.database.inverse_eqcircle:
            return list_of_three_points
        rep = diagram.database.inverse_eqcircle[logic_circle]
        circle_all = diagram.database.circles_circles[rep]
        points_sorted = sorted(circle_all.points)
        pairs = []
        for p1, p2 in itertools.combinations(points_sorted, 2):
            pairs.append(p1 + p2)
            pairs.append(p2 + p1)
        pairs_sorted = sorted(pairs)
        for pair_sorted in pairs_sorted:
            list_of_three_points.append(
                (pair_sorted[0], pair_sorted[1], circle[0]))
        return list_of_three_points

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        if new_points and len(new_points) > 0:
            # the new_from_names case
            num_new_A = sum(int(name in new_points) for name in from_names_A)
            num_new_B = sum(int(name in new_points) for name in from_names_B)
            if num_new_A != num_new_B:
                return num_new_A < num_new_B
            sort_A = "".join(sorted(from_names_A))
            sort_B = "".join(sorted(from_names_B))
            if sort_A != sort_B:
                return sort_A < sort_B
            return from_names_A < from_names_B
        # the new_from_circle case
        return from_names_A < from_names_B


class IntersectLineLine(Constructor):
    """IntersectLineLine constructor. Create new intersection for lines.

    Total number of actions C_{C_k^2}^2.
    """
    from_names_len = 4
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C, D = froms
        l_BA = Line(B, A)
        l_DC = Line(D, C)
        P = intersection_of_lines(l_BA, l_DC)
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names
        A, B, C, D = from_names

        diagram.add_point(P, p)

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(P + A)
        diagram.add_line_to_draw(D + C)
        diagram.add_line_to_draw(P + C)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C, D = from_names
        P = to_names
        return [
            Predicate("eqline", [A, B, A, P]),
            Predicate("eqline", [C, D, C, P])
        ]

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, C, D = from_names
        l_AB = Segment(A, B)
        l_AB_rep = diagram.database.inverse_eqline[l_AB]
        it = iter(diagram.database.lines_points[l_AB_rep])
        pA = next(it)
        pB = next(it)
        l_CD = Segment(C, D)
        l_CD_rep = diagram.database.inverse_eqline[l_CD]
        it = iter(diagram.database.lines_points[l_CD_rep])
        pC = next(it)
        pD = next(it)
        if all(p in diagram.parent_points for p in [pA, pB, pC, pD]) or any(
                x != y for x, y in zip([A, B, C, D], [pA, pB, pC, pD])):
            return False, EXIST
        return True, SSUCCESS

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        froms = [diagram.point_dict[name] for name in from_names]
        A, B, C, D = froms
        l_BA = Line(B, A)
        l_DC = Line(D, C)
        if parallel(l_BA, l_DC):
            return False, CANNOT

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_four_points = []
        for olds in itertools.combinations(diagram.parent_points, 3):
            for new in new_points:
                list_of_four_points.append((olds[0], olds[1], olds[2], new))
                list_of_four_points.append((olds[0], olds[2], olds[1], new))
                list_of_four_points.append((olds[0], new, olds[1], olds[2]))
        for olds in itertools.combinations(diagram.parent_points, 2):
            for news in itertools.combinations(new_points, 2):
                list_of_four_points.append(
                    (olds[0], olds[1], news[0], news[1]))
                list_of_four_points.append(
                    (olds[0], news[0], olds[1], news[1]))
                list_of_four_points.append(
                    (olds[0], news[1], olds[1], news[0]))
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 3):
                list_of_four_points.append((old, news[0], news[1], news[2]))
                list_of_four_points.append((old, news[1], news[0], news[2]))
                list_of_four_points.append((old, news[2], news[0], news[1]))
        for news in itertools.combinations(new_points, 4):
            list_of_four_points.append((news[0], news[1], news[2], news[3]))
            list_of_four_points.append((news[0], news[2], news[1], news[3]))
            list_of_four_points.append((news[0], news[3], news[1], news[2]))
        return list_of_four_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C, D = from_names
        l_AB = Segment(A, B)
        l_AB_rep = diagram.database.inverse_eqline[l_AB]
        it = iter(diagram.database.lines_points[l_AB_rep])
        pA = next(it)
        pB = next(it)
        l_CD = Segment(C, D)
        l_CD_rep = diagram.database.inverse_eqline[l_CD]
        it = iter(diagram.database.lines_points[l_CD_rep])
        pC = next(it)
        pD = next(it)
        if pA > pC:
            new_from_names = pC + pD + pA + pB
        else:
            new_from_names = pA + pB + pC + pD
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


class IntersectCircleCircle(Constructor):
    """IntersectCircleCircle constructor for two new intersections."""
    from_names_len = 4
    to_names_len = 2

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C, D = froms
        c_AB = Circle(A, B)
        c_CD = Circle(C, D)
        P = intersection_of_circles(c_AB, c_CD)
        return P

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        for p, name in zip(to, to_names):
            diagram.add_point(name, p)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        O1, A, O2, B = from_names
        P, Q = to_names
        return [
            Predicate("eqcircle", [[O1, A], [None, A, P, Q]]),
            Predicate("eqcircle", [[O2, B], [None, B, P, Q]])
        ]

    @classmethod
    def good_to_in(cls,
                   diagram: 'Diagram',
                   from_names: str,
                   to_names: Optional[str] = None) -> Tuple[bool, int]:
        # check input
        if len(from_names) != cls.from_names_len or len(set(
                from_names[:2])) != 2 or len(set(from_names[-2:])) != 2:
            return False, DISTINCT
        if (len(set(from_names)) == 2
                and from_names[:2] != from_names[-2:][::-1]
            ) or (len(set(from_names)) == 3 and from_names[-1] != from_names[0]
                  and from_names[-2] != from_names[1]):
            return False, DISTINCT
        if not all(p_name in diagram.point_dict for p_name in from_names):
            return False, POINT
        if to_names:
            if cls.to_names_len == 0:
                return False, INVALID
            if len(to_names) != cls.to_names_len or (cls.to_names_len > 1
                                                     and len(set(to_names))
                                                     != cls.to_names_len):
                return False, INVALID
            if any(p_name in diagram.point_dict for p_name in to_names):
                return False, USED
        return True, SSUCCESS

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, C, D = from_names
        if A > C:
            return False, CANNOT
        logic_AB = LogicCircle(A, [B])
        if logic_AB in diagram.database.inverse_eqcircle:
            logic_AB_rep = diagram.database.inverse_eqcircle[logic_AB]
            logic_AB_all = diagram.database.circles_circles[logic_AB_rep]
            pB = logic_AB_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pB = B
        logic_CD = LogicCircle(C, [D])
        if logic_CD in diagram.database.inverse_eqcircle:
            logic_CD_rep = diagram.database.inverse_eqcircle[logic_CD]
            logic_CD_all = diagram.database.circles_circles[logic_CD_rep]
            pD = logic_CD_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pD = D
        if all(p in diagram.parent_points
               for p in [A, pB, C, pD]) or B != pB or D != pD:
            return False, EXIST
        return True, SSUCCESS

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        logic_AB = LogicCircle(from_names[0], [from_names[1]])
        logic_CD = LogicCircle(from_names[2], [from_names[3]])
        froms = [diagram.point_dict[name] for name in from_names]
        A, B, C, D = froms
        c_AB = Circle(A, B)
        c_CD = Circle(C, D)

        if D in c_AB or B in c_CD:
            return False, CANNOT

        found = False
        for name, c in diagram.circle_dict.items():
            logic_c = LogicCircle(name[0], [name[1]])
            if c == c_AB and (draw_only or diagram.database.is_eqcircle(
                    logic_c, logic_AB)):
                found = True
                break
        if not found:
            return False, CANNOT

        found = False
        for name, c in diagram.circle_dict.items():
            logic_c = LogicCircle(name[0], [name[1]])
            if c == c_CD and (draw_only or diagram.database.is_eqcircle(
                    logic_c, logic_CD)):
                found = True
                break
        if not found:
            return False, CANNOT

        dist = np.linalg.norm(c_AB.center - c_CD.center)
        upper_bound = c_AB.r + c_CD.r
        lower_bound = abs(c_AB.r - c_CD.r)
        if not isclose(dist, upper_bound) and not isclose(
                dist, lower_bound) and lower_bound < dist < upper_bound:
            return True, SSUCCESS

        return False, NOINTERSECT

    @staticmethod
    def runtime_check(diagram: 'Diagram', to: Tuple) -> Tuple[bool, int]:
        num_dups = 0
        for p in to:
            if p is None:
                return False, CANNOT
            if p in diagram.point_dict.values():
                num_dups += 1
            if abs(p.x) >= 2000 or abs(p.y) >= 2000:
                diagram.OOB_terminal = True
                return False, CANNOT
        if num_dups > 0:
            diagram.full_dup = num_dups == 2
            return False, EXIST
        return True, RSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram', new_points: Optional[str] = None):

        def circle_circle_from_four(points: Tuple):
            """Get two circle intersection from four points."""
            circle_circle = []
            index_pairs = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3),
                           (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1),
                           (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 3, 0),
                           (1, 3, 2, 0), (2, 0, 3, 1), (2, 1, 3, 0)]
            for index_pair in index_pairs:
                circle_1 = (points[index_pair[0]], points[index_pair[1]])
                circle_2 = (points[index_pair[2]], points[index_pair[3]])
                circle_circle.append(circle_1 + circle_2)
            return circle_circle

        list_of_four_points = []
        # ABBA
        for old in diagram.parent_points:
            for new in new_points:
                list_of_four_points.append((old, new, new, old))
        for news in itertools.combinations(new_points, 2):
            list_of_four_points.append((news[0], news[1], news[1], news[0]))
        # ABBC, ABCA
        for olds in itertools.combinations(diagram.parent_points, 2):
            for new in new_points:
                list_of_four_points.append((olds[0], olds[1], olds[1], new))
                list_of_four_points.append((olds[0], olds[1], new, olds[0]))
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 2):
                list_of_four_points.append((old, news[0], news[0], news[1]))
                list_of_four_points.append((old, news[0], news[1], old))
        for news in itertools.combinations(new_points, 3):
            list_of_four_points.append((news[0], news[1], news[1], news[2]))
            list_of_four_points.append((news[0], news[1], news[2], news[0]))
        # ABCD
        for olds in itertools.combinations(diagram.parent_points, 3):
            for new in new_points:
                list_of_four_points += circle_circle_from_four(olds + (new, ))
        for olds in itertools.combinations(diagram.parent_points, 2):
            for news in itertools.combinations(new_points, 2):
                list_of_four_points += circle_circle_from_four(olds + news)
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 3):
                list_of_four_points += circle_circle_from_four((old, ) + news)
        for news in itertools.combinations(new_points, 4):
            list_of_four_points += circle_circle_from_four(news)
        return list_of_four_points

    @staticmethod
    def new_from_circle(diagram: 'Diagram', circle: str = "") -> List[Tuple]:
        list_of_four_points = []
        O, B = circle
        logic_circle = LogicCircle(O, [B])
        if logic_circle in diagram.database.inverse_eqcircle:
            rep = diagram.database.inverse_eqcircle[logic_circle]
            circle_all = diagram.database.circles_circles[rep]
            pD = circle_all.min_point
            points = circle_all.points
        else:
            pD = B
            points = {B: None}
        for c in diagram.circle_dict:
            if c[0] == O:
                continue
            logic_circle = LogicCircle(c[0], [c[1]])
            if logic_circle in diagram.database.inverse_eqcircle:
                continue
            pA = c[1]
            if pA in points:
                continue
            center = c[0]
            if center > O:
                list_of_four_points.append((O, pD, center, pA))
            else:
                list_of_four_points.append((center, pA, O, pD))
        for c_all in diagram.database.circles_circles.values():
            if c_all.center is None or c_all.center == O:
                continue
            center = c_all.center
            pA = c_all.min_point
            if len(c_all.points.keys() & points.keys()) > 0:
                continue
            if center > O:
                list_of_four_points.append((O, pD, center, pA))
            else:
                list_of_four_points.append((center, pA, O, pD))
        return sorted(list_of_four_points, key=key_func)

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C, D = from_names
        logic_AB = LogicCircle(A, [B])
        if logic_AB in diagram.database.inverse_eqcircle:
            logic_AB_rep = diagram.database.inverse_eqcircle[logic_AB]
            logic_AB_all = diagram.database.circles_circles[logic_AB_rep]
            pB = logic_AB_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pB = B
        logic_CD = LogicCircle(C, [D])
        if logic_CD in diagram.database.inverse_eqcircle:
            logic_CD_rep = diagram.database.inverse_eqcircle[logic_CD]
            logic_CD_all = diagram.database.circles_circles[logic_CD_rep]
            pD = logic_CD_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pD = D
        if A > C:
            new_from_names = C + pD + A + pB
        else:
            new_from_names = A + pB + C + pD
        return new_from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        if new_points and len(new_points) > 0:
            # the new_from_names case
            unique_A = set(from_names_A)
            unique_B = set(from_names_B)
            if len(unique_A) != len(unique_B):
                return len(unique_A) < len(unique_B)
            if len(unique_A) == 2:
                return from_names_A[:2] < from_names_B[:2]
            num_new_A = sum(int(name in new_points) for name in unique_A)
            num_new_B = sum(int(name in new_points) for name in unique_B)
            if num_new_A != num_new_B:
                return num_new_A < num_new_B
            sort_A = "".join(sorted(unique_A))
            sort_B = "".join(sorted(unique_B))
            if sort_A != sort_B:
                return sort_A < sort_B
            return from_names_A < from_names_B
        # the new_from_circle case
        return from_names_A < from_names_B


class IntersectLineCircleOff(Constructor):
    """IntersectLineCircle constructor for two new intersections."""
    from_names_len = 4
    to_names_len = 2

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C, D = froms
        c_CD = Circle(C, D)
        P = intersection_of_line_circle(A, B, c_CD)
        return P

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        A, B, _, _ = from_names
        for p, name in zip(to, to_names):
            diagram.add_point(name, p)

        diagram.add_line_to_draw(B + A)
        for p, name in zip(to, to_names):
            diagram.add_line_to_draw(name + A)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        A, B, O, D = from_names
        P, Q = to_names
        return [
            Predicate("eqline", [A, B, A, P]),
            Predicate("eqline", [A, B, A, Q]),
            Predicate("eqcircle", [[O, D], [None, D, P, Q]])
        ]

    @classmethod
    def good_to_in(cls,
                   diagram: 'Diagram',
                   from_names: str,
                   to_names: Optional[str] = None) -> Tuple[bool, int]:
        # check input
        if len(from_names) != cls.from_names_len or len(set(
                from_names[:2])) != 2 or len(set(from_names[-2:])) != 2 or len(
                    set(from_names[:2] + from_names[-1])) != 3:
            return False, DISTINCT
        if not all(p_name in diagram.point_dict for p_name in from_names):
            return False, POINT
        if to_names:
            if cls.to_names_len == 0:
                return False, INVALID
            if len(to_names) != cls.to_names_len or (cls.to_names_len > 1
                                                     and len(set(to_names))
                                                     != cls.to_names_len):
                return False, INVALID
            if any(p_name in diagram.point_dict for p_name in to_names):
                return False, USED
        return True, SSUCCESS

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, C, D = from_names
        if A > B:
            return False, CANNOT
        l_AB = Segment(A, B)
        l_AB_rep = diagram.database.inverse_eqline[l_AB]
        it = iter(diagram.database.lines_points[l_AB_rep])
        pA = next(it)
        pB = next(it)
        logic_CD = LogicCircle(C, [D])
        if logic_CD in diagram.database.inverse_eqcircle:
            logic_CD_rep = diagram.database.inverse_eqcircle[logic_CD]
            logic_CD_all = diagram.database.circles_circles[logic_CD_rep]
            pD = logic_CD_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pD = D
        if all(p in diagram.parent_points for p in [pA, pB, C, pD]) or any(
                x != y for x, y in zip([A, B, D], [pA, pB, pD])):
            return False, EXIST
        return True, SSUCCESS

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        logic_CD = LogicCircle(from_names[2], [from_names[3]])
        froms = [diagram.point_dict[name] for name in from_names]
        A, B, C, D = froms
        c_CD = Circle(C, D)

        if A in c_CD or B in c_CD:
            return False, CANNOT

        found = False
        for name, c in diagram.circle_dict.items():
            logic_c = LogicCircle(name[0], [name[1]])
            if c == c_CD and (draw_only or diagram.database.is_eqcircle(
                    logic_c, logic_CD)):
                found = True
                break
        if not found:
            return False, CANNOT

        l_BA = Line(B, A)
        p_proj = l_BA.project(c_CD.center)
        dist = np.linalg.norm(c_CD.center - p_proj.vector)
        if not isclose(dist, c_CD.r) and dist < c_CD.r:
            return True, SSUCCESS

        return False, NOINTERSECT

    @staticmethod
    def runtime_check(diagram: 'Diagram', to: Tuple) -> Tuple[bool, int]:
        num_dups = 0
        for p in to:
            if p is None:
                return False, CANNOT
            if p in diagram.point_dict.values():
                num_dups += 1
            if abs(p.x) >= 2000 or abs(p.y) >= 2000:
                diagram.OOB_terminal = True
                return False, CANNOT
        if num_dups > 0:
            diagram.full_dup = num_dups == 2
            return False, EXIST
        return True, RSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:

        def line_circle_from_four(points: Tuple) -> List[Tuple]:
            """Get line circle intersection from four points."""
            line_circle = []
            index_pairs = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2),
                           (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]
            for index_pair in index_pairs:
                line = (points[index_pair[0]], points[index_pair[1]])
                circle = (points[index_pair[2]], points[index_pair[3]])
                circle_p = (points[index_pair[3]], points[index_pair[2]])
                line_circle += [line + circle, line + circle_p]
            return line_circle

        def line_circle_from_three(points: Tuple) -> List[Tuple]:
            line_circle = []
            index_pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
            for index_pair in index_pairs:
                line = (points[index_pair[0]], points[index_pair[1]])
                circle = (points[index_pair[0]], points[index_pair[2]])
                circle_p = (points[index_pair[1]], points[index_pair[2]])
                line_circle += [line + circle, line + circle_p]
            return line_circle

        list_of_four_points = []
        for olds in itertools.combinations(diagram.parent_points, 2):
            for new in new_points:
                list_of_four_points += line_circle_from_three(olds + (new, ))
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 2):
                list_of_four_points += line_circle_from_three((old, ) + news)
        for news in itertools.combinations(new_points, 3):
            list_of_four_points += line_circle_from_three(news)
        for olds in itertools.combinations(diagram.parent_points, 3):
            for new in new_points:
                list_of_four_points += line_circle_from_four(olds + (new, ))
        for olds in itertools.combinations(diagram.parent_points, 2):
            for news in itertools.combinations(new_points, 2):
                list_of_four_points += line_circle_from_four(olds + news)
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 3):
                list_of_four_points += line_circle_from_four((old, ) + news)
        for news in itertools.combinations(new_points, 4):
            list_of_four_points += line_circle_from_four(news)
        return list_of_four_points

    @staticmethod
    def new_from_circle(diagram: 'Diagram', circle: str = "") -> List[Tuple]:
        list_of_four_points = []
        O, B = circle
        logic_circle = LogicCircle(O, [B])
        if logic_circle in diagram.database.inverse_eqcircle:
            rep = diagram.database.inverse_eqcircle[logic_circle]
            circle_all = diagram.database.circles_circles[rep]
            pD = circle_all.min_point
            points = circle_all.points
        else:
            pD = B
            points = {B: None}
        for l_points in diagram.database.lines_points.values():
            it = iter(l_points)
            pA = next(it)
            pB = next(it)
            if pA in points or pB in points:
                continue
            list_of_four_points.append((pA, pB, O, pD))
        return sorted(list_of_four_points, key=key_func)

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C, D = from_names
        l_AB = Segment(A, B)
        l_AB_rep = diagram.database.inverse_eqline[l_AB]
        it = iter(diagram.database.lines_points[l_AB_rep])
        pA = next(it)
        pB = next(it)
        logic_CD = LogicCircle(C, [D])
        if logic_CD in diagram.database.inverse_eqcircle:
            logic_CD_rep = diagram.database.inverse_eqcircle[logic_CD]
            logic_CD_all = diagram.database.circles_circles[logic_CD_rep]
            pD = logic_CD_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pD = D
        new_from_names = pA + pB + C + pD
        return new_from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        if new_points and len(new_points) > 0:
            unique_A = set(from_names_A)
            unique_B = set(from_names_B)
            if len(unique_A) != len(unique_B):
                return len(unique_A) < len(unique_B)
            num_new_A = sum(int(name in new_points) for name in unique_A)
            num_new_B = sum(int(name in new_points) for name in unique_B)
            if num_new_A != num_new_B:
                return num_new_A < num_new_B
            sort_A = "".join(sorted(unique_A))
            sort_B = "".join(sorted(unique_B))
            if sort_A != sort_B:
                return sort_A < sort_B
            return from_names_A < from_names_B
        return from_names_A < from_names_B


class IntersectLineCircleOn(Constructor):
    """IntersectLineCircle constructor for one additional intersection."""
    from_names_len = 3
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, O = froms
        l_AB = Line(A, B)
        BO = O - B
        P = 2 * BO - 2 * BO.vector.dot(l_AB.norm) * l_AB.norm + B
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        A, B, _ = from_names
        for p, name in zip(to, to_names):
            diagram.add_point(name, p)

        diagram.add_line_to_draw(B + A)
        for p, name in zip(to, to_names):
            diagram.add_line_to_draw(name + A)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        A, B, O = from_names
        P = to_names
        return [
            Predicate("eqline", [A, B, A, P]),
            Predicate("eqcircle", [[O, B], [O, P]])
        ]

    @classmethod
    def good_to_in(cls,
                   diagram: 'Diagram',
                   from_names: str,
                   to_names: Optional[str] = None) -> Tuple[bool, int]:
        # check input
        if len(from_names) != cls.from_names_len or len(set(
                from_names[:2])) != 2 or len(set(from_names[-2:])) != 2:
            return False, DISTINCT
        if not all(p_name in diagram.point_dict for p_name in from_names):
            return False, POINT
        if to_names:
            if cls.to_names_len == 0:
                return False, INVALID
            if len(to_names) != cls.to_names_len or (cls.to_names_len > 1
                                                     and len(set(to_names))
                                                     != cls.to_names_len):
                return False, INVALID
            if any(p_name in diagram.point_dict for p_name in to_names):
                return False, USED
        return True, SSUCCESS

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, C = from_names
        l_AB = Segment(A, B)
        l_AB_rep = diagram.database.inverse_eqline[l_AB]
        it = iter(diagram.database.lines_points[l_AB_rep])
        pA = next(it)
        if pA == B:
            pA = next(it)
        if all(p in diagram.parent_points for p in [pA, B, C]) or A != pA:
            return False, EXIST
        return True, SSUCCESS

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        logic_OB = LogicCircle(from_names[2], [from_names[1]])
        froms = [diagram.point_dict[name] for name in from_names]
        A, B, O = froms
        c_OB = Circle(O, B)

        if A in c_OB:
            return False, CANNOT

        found = False
        for name, c in diagram.circle_dict.items():
            logic_c = LogicCircle(name[0], [name[1]])
            if c == c_OB and (draw_only or diagram.database.is_eqcircle(
                    logic_c, logic_OB)):
                found = True
                break
        if not found:
            return False, CANNOT

        l_BA = Line(B, A)
        p_proj = l_BA.project(c_OB.center)
        dist = np.linalg.norm(c_OB.center - p_proj.vector)
        if not isclose(dist, c_OB.r) and dist < c_OB.r:
            return True, SSUCCESS

        return False, NOINTERSECT

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_three_points = []
        for old in diagram.parent_points:
            for new in new_points:
                list_of_three_points.append((old, new, old))
        for new in new_points:
            for old in diagram.parent_points:
                list_of_three_points.append((new, old, new))
        for news in itertools.permutations(new_points, 2):
            list_of_three_points.append((news[0], news[1], news[0]))
        for num_new in range(1, 4):
            num_old = 3 - num_new
            from_old = itertools.combinations(diagram.parent_points, num_old)
            from_new = itertools.combinations(new_points, num_new)
            combs = itertools.product(from_old, from_new)
            for comb in combs:
                list_of_three_points += itertools.permutations(
                    comb[0] + comb[1], 3)
        return list_of_three_points

    @staticmethod
    def new_from_circle(diagram: 'Diagram', circle: str = "") -> List[Tuple]:
        list_of_three_points = []
        O, B = circle
        logic_circle = LogicCircle(O, [B])
        points = {B: None}
        if logic_circle in diagram.database.inverse_eqcircle:
            rep = diagram.database.inverse_eqcircle[logic_circle]
            points.update(diagram.database.circles_circles[rep].points)
        for BB in points:
            for l in diagram.database.points_lines[BB]:
                it = iter(diagram.database.lines_points[l])
                pA = next(it)
                if pA == BB:
                    pA = next(it)
                if pA in points:
                    continue
                list_of_three_points.append((pA, BB, O))
        return sorted(list_of_three_points, key=key_func)

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C = from_names
        l_AB = Segment(A, B)
        l_AB_rep = diagram.database.inverse_eqline[l_AB]
        it = iter(diagram.database.lines_points[l_AB_rep])
        pA = next(it)
        if pA == B:
            pA = next(it)
        new_from_names = pA + B + C
        return new_from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        if new_points and len(new_points) > 0:
            # the new_from_names case
            unique_A = set(from_names_A)
            unique_B = set(from_names_B)
            if len(unique_A) != len(unique_B):
                return len(unique_A) < len(unique_B)
            if len(unique_A) == 2:
                return from_names_A[:2] < from_names_B[:2]
            num_new_A = sum(int(name in new_points) for name in unique_A)
            num_new_B = sum(int(name in new_points) for name in unique_B)
            if num_new_A != num_new_B:
                return num_new_A < num_new_B
            sort_A = "".join(sorted(unique_A))
            sort_B = "".join(sorted(unique_B))
            if sort_A != sort_B:
                return sort_A < sort_B
            return from_names_A < from_names_B
        # the new_from_circle case
        return from_names_A < from_names_B


class Reflect(Constructor):
    """Reflect a point over a line."""
    from_names_len = 3
    to_names_len = 2

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, Q = froms
        l_AB = Line(A, B)
        AQ = Q - A
        diff = AQ.vector.dot(l_AB.norm) * l_AB.norm
        H = AQ - diff + A
        P = AQ - 2 * diff + A
        return H, P

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        for p, P in zip(to, to_names):
            diagram.add_point(P, p)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, Q = from_names
        H, P = to_names
        return [
            Predicate("eqline", [A, H, H, B]),
            Predicate("eqline", [Q, H, H, P]),
            Predicate("perp", [A, H, Q]),
            Predicate("perp", [A, H, P]),
            Predicate("perp", [B, H, Q]),
            Predicate("perp", [B, H, P]),
            Predicate("cong", [A, Q, A, P]),
            Predicate("cong", [B, Q, B, P]),
            Predicate("cong", [Q, H, P, H])
        ]

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, P = from_names
        l_AB = Segment(A, B)
        l_AB_rep = diagram.database.inverse_eqline[l_AB]
        it = iter(diagram.database.lines_points[l_AB_rep])
        pA = next(it)
        pB = next(it)
        if all(p in diagram.parent_points
               for p in [pA, pB, P]) or A != pA or B != pB:
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
        for olds in itertools.combinations(diagram.parent_points, 2):
            for new in new_points:
                list_of_three_points.append((olds[0], olds[1], new))
                list_of_three_points.append((olds[0], new, olds[1]))
                list_of_three_points.append((olds[1], new, olds[0]))
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 2):
                list_of_three_points.append((old, news[0], news[1]))
                list_of_three_points.append((old, news[1], news[0]))
                list_of_three_points.append((news[0], news[1], old))
        for news in itertools.combinations(new_points, 3):
            list_of_three_points.append((news[0], news[1], news[2]))
            list_of_three_points.append((news[0], news[2], news[1]))
            list_of_three_points.append((news[1], news[2], news[0]))
        return list_of_three_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, P = from_names
        l_AB = Segment(A, B)
        l_AB_rep = diagram.database.inverse_eqline[l_AB]
        it = iter(diagram.database.lines_points[l_AB_rep])
        pA = next(it)
        pB = next(it)
        new_from_names = pA + pB + P
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


class InSimiliCenter(Constructor):
    """Internal similitude center constructor for two circles."""
    from_names_len = 4
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        O1, R1, O2, R2 = froms
        r1 = (O1 - R1).norm()
        r2 = (O2 - R2).norm()
        P = (r1 * O2 + r2 * O1) / (r1 + r2)
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names

        O1 = from_names[0]
        O2 = from_names[2]

        diagram.add_point(P, p)
        diagram.add_line_to_draw(P + O1)
        diagram.add_line_to_draw(P + O2)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        O1, R1, O2, R2 = from_names
        P = to_names
        return [
            Predicate("eqline", [O1, P, O2, P]),
            Predicate("eqratio", [P, O1, P, O2, O1, R1, O2, R2])
        ]

    @classmethod
    def good_to_in(cls,
                   diagram: 'Diagram',
                   from_names: str,
                   to_names: Optional[str] = None) -> Tuple[bool, int]:
        # check input
        if len(from_names) != cls.from_names_len or len(set(
                from_names[:2])) != 2 or len(set(from_names[-2:])) != 2:
            return False, DISTINCT
        if from_names[0] == from_names[2]:
            return False, DISTINCT
        if not all(p_name in diagram.point_dict for p_name in from_names):
            return False, POINT
        if to_names:
            if cls.to_names_len == 0:
                return False, INVALID
            if len(to_names) != cls.to_names_len or (cls.to_names_len > 1
                                                     and len(set(to_names))
                                                     != cls.to_names_len):
                return False, INVALID
            if any(p_name in diagram.point_dict for p_name in to_names):
                return False, USED
        return True, SSUCCESS

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, C, D = from_names
        if A > C:
            return False, CANNOT
        logic_AB = LogicCircle(A, [B])
        if logic_AB in diagram.database.inverse_eqcircle:
            logic_AB_rep = diagram.database.inverse_eqcircle[logic_AB]
            logic_AB_all = diagram.database.circles_circles[logic_AB_rep]
            pB = logic_AB_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pB = B
        logic_CD = LogicCircle(C, [D])
        if logic_CD in diagram.database.inverse_eqcircle:
            logic_CD_rep = diagram.database.inverse_eqcircle[logic_CD]
            logic_CD_all = diagram.database.circles_circles[logic_CD_rep]
            pD = logic_CD_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pD = D
        if all(p in diagram.parent_points
               for p in [A, pB, C, pD]) or B != pB or D != pD:
            return False, EXIST
        return True, SSUCCESS

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        logic_AB = LogicCircle(from_names[0], [from_names[1]])
        logic_CD = LogicCircle(from_names[2], [from_names[3]])
        froms = [diagram.point_dict[name] for name in from_names]
        A, B, C, D = froms
        c_AB = Circle(A, B)
        c_CD = Circle(C, D)

        found = False
        for name, c in diagram.circle_dict.items():
            logic_c = LogicCircle(name[0], [name[1]])
            if c == c_AB and (draw_only or diagram.database.is_eqcircle(
                    logic_c, logic_AB)):
                found = True
                break
        if not found:
            return False, CANNOT

        found = False
        for name, c in diagram.circle_dict.items():
            logic_c = LogicCircle(name[0], [name[1]])
            if c == c_CD and (draw_only or diagram.database.is_eqcircle(
                    logic_c, logic_CD)):
                found = True
                break
        if not found:
            return False, CANNOT

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram', new_points: Optional[str] = None):

        def circle_circle_from_four(points: Tuple):
            """Get two circle intersection from four points."""
            circle_circle = []
            index_pairs = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3),
                           (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1),
                           (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 3, 0),
                           (1, 3, 2, 0), (2, 0, 3, 1), (2, 1, 3, 0)]
            for index_pair in index_pairs:
                circle_1 = (points[index_pair[0]], points[index_pair[1]])
                circle_2 = (points[index_pair[2]], points[index_pair[3]])
                circle_circle.append(circle_1 + circle_2)
            return circle_circle

        list_of_four_points = []
        # ABBA
        for old in diagram.parent_points:
            for new in new_points:
                list_of_four_points.append((old, new, new, old))
        for news in itertools.combinations(new_points, 2):
            list_of_four_points.append((news[0], news[1], news[1], news[0]))
        # ABBC, ABCA, ABCB
        for olds in itertools.combinations(diagram.parent_points, 2):
            for new in new_points:
                list_of_four_points.append((olds[0], olds[1], olds[1], new))
                list_of_four_points.append((olds[0], olds[1], new, olds[0]))
                list_of_four_points.append((olds[0], olds[1], new, olds[1]))
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 2):
                list_of_four_points.append((old, news[0], news[0], news[1]))
                list_of_four_points.append((old, news[0], news[1], old))
                list_of_four_points.append((old, news[0], news[1], news[0]))
        for news in itertools.combinations(new_points, 3):
            list_of_four_points.append((news[0], news[1], news[1], news[2]))
            list_of_four_points.append((news[0], news[1], news[2], news[0]))
            list_of_four_points.append((news[0], news[1], news[2], news[1]))
        # ABCD
        for olds in itertools.combinations(diagram.parent_points, 3):
            for new in new_points:
                list_of_four_points += circle_circle_from_four(olds + (new, ))
        for olds in itertools.combinations(diagram.parent_points, 2):
            for news in itertools.combinations(new_points, 2):
                list_of_four_points += circle_circle_from_four(olds + news)
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 3):
                list_of_four_points += circle_circle_from_four((old, ) + news)
        for news in itertools.combinations(new_points, 4):
            list_of_four_points += circle_circle_from_four(news)
        return list_of_four_points

    @staticmethod
    def new_from_circle(diagram: 'Diagram', circle: str = "") -> List[Tuple]:
        list_of_four_points = []
        O, B = circle
        logic_circle = LogicCircle(O, [B])
        if logic_circle in diagram.database.inverse_eqcircle:
            rep = diagram.database.inverse_eqcircle[logic_circle]
            circle_all = diagram.database.circles_circles[rep]
            pD = circle_all.min_point
        else:
            pD = B
        for c in diagram.circle_dict:
            if c[0] == O:
                continue
            logic_circle = LogicCircle(c[0], [c[1]])
            if logic_circle in diagram.database.inverse_eqcircle:
                continue
            pA = c[1]
            center = c[0]
            if center > O:
                list_of_four_points.append((O, pD, center, pA))
            else:
                list_of_four_points.append((center, pA, O, pD))
        for c_all in diagram.database.circles_circles.values():
            if c_all.center is None or c_all.center == O:
                continue
            center = c_all.center
            pA = c_all.min_point
            if center > O:
                list_of_four_points.append((O, pD, center, pA))
            else:
                list_of_four_points.append((center, pA, O, pD))
        return sorted(list_of_four_points, key=key_func)

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C, D = from_names
        logic_AB = LogicCircle(A, [B])
        if logic_AB in diagram.database.inverse_eqcircle:
            logic_AB_rep = diagram.database.inverse_eqcircle[logic_AB]
            logic_AB_all = diagram.database.circles_circles[logic_AB_rep]
            pB = logic_AB_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pB = B
        logic_CD = LogicCircle(C, [D])
        if logic_CD in diagram.database.inverse_eqcircle:
            logic_CD_rep = diagram.database.inverse_eqcircle[logic_CD]
            logic_CD_all = diagram.database.circles_circles[logic_CD_rep]
            pD = logic_CD_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pD = D
        if A > C:
            new_from_names = C + pD + A + pB
        else:
            new_from_names = A + pB + C + pD
        return new_from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        if new_points and len(new_points) > 0:
            # the new_from_names case
            unique_A = set(from_names_A)
            unique_B = set(from_names_B)
            if len(unique_A) != len(unique_B):
                return len(unique_A) < len(unique_B)
            if len(unique_A) == 2:
                return from_names_A[:2] < from_names_B[:2]
            num_new_A = sum(int(name in new_points) for name in unique_A)
            num_new_B = sum(int(name in new_points) for name in unique_B)
            if num_new_A != num_new_B:
                return num_new_A < num_new_B
            sort_A = "".join(sorted(unique_A))
            sort_B = "".join(sorted(unique_B))
            if sort_A != sort_B:
                return sort_A < sort_B
            return from_names_A < from_names_B
        # the new_from_circle case
        return from_names_A < from_names_B


class ExSimiliCenter(Constructor):
    """External similitude center constructor for two circles."""
    from_names_len = 4
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        O1, R1, O2, R2 = froms
        r1 = (O1 - R1).norm()
        r2 = (O2 - R2).norm()
        P = (r1 * O2 - r2 * O1) / (r1 - r2)
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names

        O1 = from_names[0]
        O2 = from_names[2]

        diagram.add_point(P, p)
        diagram.add_line_to_draw(P + O1)
        diagram.add_line_to_draw(P + O2)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        O1, R1, O2, R2 = from_names
        P = to_names
        return [
            Predicate("eqline", [O1, P, O2, P]),
            Predicate("eqratio", [P, O1, P, O2, O1, R1, O2, R2])
        ]

    @classmethod
    def good_to_in(cls,
                   diagram: 'Diagram',
                   from_names: str,
                   to_names: Optional[str] = None) -> Tuple[bool, int]:
        # check input
        if len(from_names) != cls.from_names_len or len(set(
                from_names[:2])) != 2 or len(set(from_names[-2:])) != 2:
            return False, DISTINCT
        if from_names[0] == from_names[2] or len(set(from_names)) == 2:
            return False, DISTINCT
        if not all(p_name in diagram.point_dict for p_name in from_names):
            return False, POINT
        if to_names:
            if cls.to_names_len == 0:
                return False, INVALID
            if len(to_names) != cls.to_names_len or (cls.to_names_len > 1
                                                     and len(set(to_names))
                                                     != cls.to_names_len):
                return False, INVALID
            if any(p_name in diagram.point_dict for p_name in to_names):
                return False, USED
        return True, SSUCCESS

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, C, D = from_names
        if A > C:
            return False, CANNOT
        logic_AB = LogicCircle(A, [B])
        if logic_AB in diagram.database.inverse_eqcircle:
            logic_AB_rep = diagram.database.inverse_eqcircle[logic_AB]
            logic_AB_all = diagram.database.circles_circles[logic_AB_rep]
            pB = logic_AB_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pB = B
        logic_CD = LogicCircle(C, [D])
        if logic_CD in diagram.database.inverse_eqcircle:
            logic_CD_rep = diagram.database.inverse_eqcircle[logic_CD]
            logic_CD_all = diagram.database.circles_circles[logic_CD_rep]
            pD = logic_CD_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pD = D
        if all(p in diagram.parent_points
               for p in [A, pB, C, pD]) or B != pB or D != pD:
            return False, EXIST
        return True, SSUCCESS

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        logic_AB = LogicCircle(from_names[0], [from_names[1]])
        logic_CD = LogicCircle(from_names[2], [from_names[3]])
        froms = [diagram.point_dict[name] for name in from_names]
        A, B, C, D = froms
        c_AB = Circle(A, B)
        c_CD = Circle(C, D)

        r1 = (A - B).norm()
        r2 = (C - D).norm()
        if isclose(r1, r2, 1e-4):
            return False, CANNOT

        found = False
        for name, c in diagram.circle_dict.items():
            logic_c = LogicCircle(name[0], [name[1]])
            if c == c_AB and (draw_only or diagram.database.is_eqcircle(
                    logic_c, logic_AB)):
                found = True
                break
        if not found:
            return False, CANNOT

        found = False
        for name, c in diagram.circle_dict.items():
            logic_c = LogicCircle(name[0], [name[1]])
            if c == c_CD and (draw_only or diagram.database.is_eqcircle(
                    logic_c, logic_CD)):
                found = True
                break
        if not found:
            return False, CANNOT

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram', new_points: Optional[str] = None):

        def circle_circle_from_four(points: Tuple):
            """Get two circle intersection from four points."""
            circle_circle = []
            index_pairs = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3),
                           (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1),
                           (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 3, 0),
                           (1, 3, 2, 0), (2, 0, 3, 1), (2, 1, 3, 0)]
            for index_pair in index_pairs:
                circle_1 = (points[index_pair[0]], points[index_pair[1]])
                circle_2 = (points[index_pair[2]], points[index_pair[3]])
                circle_circle.append(circle_1 + circle_2)
            return circle_circle

        list_of_four_points = []
        # ABBA would have the same radii
        # ABBC, ABCA, ABCB
        for olds in itertools.combinations(diagram.parent_points, 2):
            for new in new_points:
                list_of_four_points.append((olds[0], olds[1], olds[1], new))
                list_of_four_points.append((olds[0], olds[1], new, olds[0]))
                list_of_four_points.append((olds[0], olds[1], new, olds[1]))
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 2):
                list_of_four_points.append((old, news[0], news[0], news[1]))
                list_of_four_points.append((old, news[0], news[1], old))
                list_of_four_points.append((old, news[0], news[1], news[0]))
        for news in itertools.combinations(new_points, 3):
            list_of_four_points.append((news[0], news[1], news[1], news[2]))
            list_of_four_points.append((news[0], news[1], news[2], news[0]))
            list_of_four_points.append((news[0], news[1], news[2], news[1]))
        # ABCD
        for olds in itertools.combinations(diagram.parent_points, 3):
            for new in new_points:
                list_of_four_points += circle_circle_from_four(olds + (new, ))
        for olds in itertools.combinations(diagram.parent_points, 2):
            for news in itertools.combinations(new_points, 2):
                list_of_four_points += circle_circle_from_four(olds + news)
        for old in diagram.parent_points:
            for news in itertools.combinations(new_points, 3):
                list_of_four_points += circle_circle_from_four((old, ) + news)
        for news in itertools.combinations(new_points, 4):
            list_of_four_points += circle_circle_from_four(news)
        return list_of_four_points

    @staticmethod
    def new_from_circle(diagram: 'Diagram', circle: str = "") -> List[Tuple]:
        list_of_four_points = []
        O, B = circle
        logic_circle = LogicCircle(O, [B])
        if logic_circle in diagram.database.inverse_eqcircle:
            rep = diagram.database.inverse_eqcircle[logic_circle]
            circle_all = diagram.database.circles_circles[rep]
            pD = circle_all.min_point
        else:
            pD = B
        for c in diagram.circle_dict:
            if c[0] == O:
                continue
            logic_circle = LogicCircle(c[0], [c[1]])
            if logic_circle in diagram.database.inverse_eqcircle:
                continue
            pA = c[1]
            center = c[0]
            if center > O:
                list_of_four_points.append((O, pD, center, pA))
            else:
                list_of_four_points.append((center, pA, O, pD))
        for c_all in diagram.database.circles_circles.values():
            if c_all.center is None or c_all.center == O:
                continue
            center = c_all.center
            pA = c_all.min_point
            if center > O:
                list_of_four_points.append((O, pD, center, pA))
            else:
                list_of_four_points.append((center, pA, O, pD))
        return sorted(list_of_four_points, key=key_func)

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C, D = from_names
        logic_AB = LogicCircle(A, [B])
        if logic_AB in diagram.database.inverse_eqcircle:
            logic_AB_rep = diagram.database.inverse_eqcircle[logic_AB]
            logic_AB_all = diagram.database.circles_circles[logic_AB_rep]
            pB = logic_AB_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pB = B
        logic_CD = LogicCircle(C, [D])
        if logic_CD in diagram.database.inverse_eqcircle:
            logic_CD_rep = diagram.database.inverse_eqcircle[logic_CD]
            logic_CD_all = diagram.database.circles_circles[logic_CD_rep]
            pD = logic_CD_all.min_point
        else:
            # special case when the circle is created from CenterCircle
            pD = D
        if A > C:
            new_from_names = C + pD + A + pB
        else:
            new_from_names = A + pB + C + pD
        return new_from_names, to_names

    @staticmethod
    def lt(from_names_A: str, from_names_B: str, new_points: str) -> bool:
        if new_points and len(new_points) > 0:
            # the new_from_names case
            unique_A = set(from_names_A)
            unique_B = set(from_names_B)
            if len(unique_A) != len(unique_B):
                return len(unique_A) < len(unique_B)
            if len(unique_A) == 2:
                return from_names_A[:2] < from_names_B[:2]
            num_new_A = sum(int(name in new_points) for name in unique_A)
            num_new_B = sum(int(name in new_points) for name in unique_B)
            if num_new_A != num_new_B:
                return num_new_A < num_new_B
            sort_A = "".join(sorted(unique_A))
            sort_B = "".join(sorted(unique_B))
            if sort_A != sort_B:
                return sort_A < sort_B
            return from_names_A < from_names_B
        # the new_from_circle case
        return from_names_A < from_names_B
