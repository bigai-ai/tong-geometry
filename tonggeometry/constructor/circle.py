r"""Define constructors for circles."""

import itertools
from typing import TYPE_CHECKING, List, Tuple

from tonggeometry.constructor.parent import (CANNOT, EXIST, RSUCCESS, SAMELINE,
                                             SSUCCESS, Constructor)
from tonggeometry.constructor.primitives import (Circle, Point,
                                                 intersection_of_line_circle,
                                                 on_same_line)
from tonggeometry.inference_engine.predicate import Predicate
from tonggeometry.inference_engine.primitives import Circle as LogicCircle
from tonggeometry.util import isclose

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram

__all__ = ["CircumscribedCircle", "InCircle", "ExCircle", "CenterCircle"]


class CircumscribedCircle(Constructor):
    """Constructor for the circumscribed circle for a triangle.

    Total number of actions C_k^3.
    """
    from_names_len = 3
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point, Circle]:
        A, B, C = froms
        BC = C - B
        a = BC.norm()
        CA = A - C
        b = CA.norm()
        AB = B - A
        c = AB.norm()
        area = 0.5 * abs(AB.cross(CA))
        coeff_A = a**2 * (b**2 + c**2 - a**2) / (16 * area**2)
        coeff_B = b**2 * (c**2 + a**2 - b**2) / (16 * area**2)
        coeff_C = c**2 * (a**2 + b**2 - c**2) / (16 * area**2)
        center = coeff_A * A + coeff_B * B + coeff_C * C
        circle = Circle(center, A)
        return center, circle

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        A = from_names[0]
        center, circle = to
        Center = to_names

        diagram.add_point(Center, center)
        diagram.add_circle(Center + A, circle)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C = from_names
        O = to_names
        return [Predicate("eqcircle", [[O, A], [None, A, B, C]])]

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, C = from_names
        logic_ABC = LogicCircle(None, [A, B, C])
        if logic_ABC in diagram.database.inverse_eqcircle:
            logic_ABC_rep = diagram.database.inverse_eqcircle[logic_ABC]
            logic_ABC_all = diagram.database.circles_circles[logic_ABC_rep]
            pA, pB, pC = sorted(list(logic_ABC_all.points.keys()))[:3]
        else:
            pA, pB, pC = A, B, C
        if all(p in diagram.parent_points for p in [pA, pB, pC]) or any(
                x != y for x, y in zip([A, B, C], [pA, pB, pC])):
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
    def runtime_check(diagram: 'Diagram', to: Tuple) -> Tuple[bool, int]:
        center, circle = to
        num_dups = 0
        if center in diagram.point_dict.values():
            num_dups += 1
        if circle in diagram.circle_dict.values():
            num_dups += 1
        if num_dups > 0:
            diagram.full_dup = num_dups == 2
            return False, EXIST
        left_end = center.x - circle.r
        right_end = center.x + circle.r
        down_end = center.y - circle.r
        up_end = center.y + circle.r
        if any(
                abs(val) >= 2000 for val in
            [center.x, center.y, left_end, right_end, up_end, down_end]):
            diagram.OOB_terminal = True
            return False, CANNOT
        return True, RSUCCESS

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
        logic_ABC = LogicCircle(None, [A, B, C])
        if logic_ABC in diagram.database.inverse_eqcircle:
            logic_ABC_rep = diagram.database.inverse_eqcircle[logic_ABC]
            logic_ABC_all = diagram.database.circles_circles[logic_ABC_rep]
            pA, pB, pC = sorted(list(logic_ABC_all.points.keys()))[:3]
        else:
            pA, pB, pC = sorted(from_names)
        new_from_names = "".join([pA, pB, pC])
        return new_from_names, to_names


class InCircle(Constructor):
    """Constructor for the incircle for a triangle.

    Total number of actions C_k^3.
    """
    from_names_len = 3
    to_names_len = 4

    @staticmethod
    def compute(
            froms: List[Point]) -> Tuple[Point, Circle, Point, Point, Point]:
        A, B, C = froms
        BC = C - B
        a = BC.norm()
        CA = A - C
        b = CA.norm()
        AB = B - A
        c = AB.norm()
        area = 0.5 * abs(AB.cross(CA))
        r = 2 * area / (a + b + c)
        center = (a * A + b * B + c * C) / (a + b + c)
        circle = Circle(center, r)
        D = intersection_of_line_circle(A, B, circle)[0]
        E = intersection_of_line_circle(A, C, circle)[0]
        F = intersection_of_line_circle(B, C, circle)[0]

        return center, circle, D, E, F

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        A, B, C = from_names
        center, circle, d, e, f = to

        Center, D, E, F = to_names

        diagram.add_point(Center, center)
        diagram.add_point(D, d)
        diagram.add_point(E, e)
        diagram.add_point(F, f)
        diagram.add_circle(Center + D, circle)

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + A)
        diagram.add_line_to_draw(C + B)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C = from_names
        I, D, E, F = to_names
        return [
            Predicate("eqcircle", [[I, D], [None, D, E, F]]),
            Predicate("eqline", [A, B, A, D]),
            Predicate("eqline", [A, C, A, E]),
            Predicate("eqline", [B, C, B, F]),
            Predicate("perp", [A, D, I]),
            Predicate("perp", [B, F, I]),
            Predicate("perp", [C, E, I]),
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
    def runtime_check(diagram: 'Diagram', to: Tuple) -> Tuple[bool, int]:
        center, circle, D, E, F = to
        if any(p is None for p in [D, E, F]):
            return False, CANNOT
        if circle in diagram.circle_dict.values():
            return False, EXIST
        if any(p in diagram.point_dict.values() for p in [center, D, E, F]):
            return False, EXIST
        left_end = center.x - circle.r
        right_end = center.x + circle.r
        down_end = center.y - circle.r
        up_end = center.y + circle.r
        if any(
                abs(val) >= 2000 for val in [
                    center.x, center.y, D.x, D.y, E.x, E.y, F.x, F.y, left_end,
                    right_end, up_end, down_end
                ]):
            diagram.OOB_terminal = True
            return False, CANNOT
        return True, RSUCCESS

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


class ExCircle(Constructor):
    """Constructor for the excircle for a triangle.

    Total number of actions 3 * C_k^3.
    """
    from_names_len = 3
    to_names_len = 4

    @staticmethod
    def compute(
            froms: List[Point]) -> Tuple[Point, Circle, Point, Point, Point]:
        A, B, C = froms
        BC = C - B
        a = BC.norm()
        CA = A - C
        b = CA.norm()
        AB = B - A
        c = AB.norm()
        area = 0.5 * abs(AB.cross(CA))
        r = 2 * area / (a - b + c)
        center = (a * A - b * B + c * C) / (a - b + c)
        circle = Circle(center, r)
        D = intersection_of_line_circle(A, B, circle)[0]
        E = intersection_of_line_circle(A, C, circle)[0]
        F = intersection_of_line_circle(B, C, circle)[0]
        return center, circle, D, E, F

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        A, B, C = from_names
        center, circle, d, e, f = to

        Center, D, E, F = to_names

        diagram.add_point(Center, center)
        diagram.add_point(D, d)
        diagram.add_point(E, e)
        diagram.add_point(F, f)
        diagram.add_circle(Center + D, circle)

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + A)
        diagram.add_line_to_draw(C + B)
        diagram.add_line_to_draw(D + A)
        diagram.add_line_to_draw(F + C)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        A, B, C = from_names
        I, D, E, F = to_names
        return [
            Predicate("eqcircle", [[I, D], [None, D, E, F]]),
            Predicate("eqline", [A, B, A, D]),
            Predicate("eqline", [A, C, A, E]),
            Predicate("eqline", [B, C, B, F]),
            Predicate("perp", [A, D, I]),
            Predicate("perp", [B, F, I]),
            Predicate("perp", [C, E, I]),
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
        A, B, C = froms
        BC = C - B
        a = BC.norm()
        CA = A - C
        b = CA.norm()
        AB = B - A
        c = AB.norm()
        if isclose(a - b + c, 0):
            return False, SAMELINE

        return True, SSUCCESS

    @staticmethod
    def runtime_check(diagram: 'Diagram', to: Tuple) -> Tuple[bool, int]:
        center, circle, D, E, F = to
        if any(p is None for p in [D, E, F]):
            return False, CANNOT
        if circle in diagram.circle_dict.values():
            return False, EXIST
        if any(p in diagram.point_dict.values() for p in [center, D, E, F]):
            return False, EXIST
        left_end = center.x - circle.r
        right_end = center.x + circle.r
        down_end = center.y - circle.r
        up_end = center.y + circle.r
        if any(
                abs(val) >= 2000 for val in [
                    center.x, center.y, D.x, D.y, E.x, E.y, F.x, F.y, left_end,
                    right_end, up_end, down_end
                ]):
            diagram.OOB_terminal = True
            return False, CANNOT
        return True, RSUCCESS

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
            new_to_names = to_names[0] + to_names[1:][::-1]
        else:
            new_from_names = from_names
            new_to_names = to_names
        return new_from_names, new_to_names

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


class CenterCircle(Constructor):
    """Constructor for a circle given the center and a point.

    Total number of actions A_k^2.
    """
    from_names_len = 2
    to_names_len = 0

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Circle]:
        O, A = froms
        circle = Circle(O, A)
        return (circle, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        Center, A = from_names
        circle = to[0]

        diagram.add_circle(Center + A, circle)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        return []

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B = from_names
        logic_AB = LogicCircle(A, [B])
        if logic_AB in diagram.database.inverse_eqcircle:
            logic_AB_rep = diagram.database.inverse_eqcircle[logic_AB]
            logic_AB_all = diagram.database.circles_circles[logic_AB_rep]
            pB = logic_AB_all.min_point
        else:
            pB = B
        if B != pB:
            return False, EXIST
        return True, SSUCCESS

    @staticmethod
    def runtime_check(diagram: 'Diagram', to: Tuple) -> Tuple[bool, int]:
        circle = to[0]
        if circle in diagram.circle_dict.values():
            diagram.full_dup = True
            return False, EXIST
        return True, RSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_two_points = []
        for combo in itertools.product(diagram.parent_points, new_points):
            list_of_two_points.append(combo)
        for combo in itertools.product(new_points, diagram.parent_points):
            list_of_two_points.append(combo)
        list_of_two_points += itertools.permutations(new_points, 2)
        return list_of_two_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B = from_names
        logic_AB = LogicCircle(A, [B])
        if logic_AB in diagram.database.inverse_eqcircle:
            logic_AB_rep = diagram.database.inverse_eqcircle[logic_AB]
            logic_AB_all = diagram.database.circles_circles[logic_AB_rep]
            pB = logic_AB_all.min_point
        else:
            pB = B
        new_from_names = A + pB
        return new_from_names, to_names
