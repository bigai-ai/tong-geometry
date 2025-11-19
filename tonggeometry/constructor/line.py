r"""Define constructors for lines."""

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

__all__ = ["Perpendicular", "Parallel"]


class Perpendicular(Constructor):
    """Constructor for a perpendicular line."""
    from_names_len = 3
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C = froms
        AB = B - A
        A_p = Point(-AB.y, AB.x) + A
        l_CB = Line(C, B)
        l_perpA = Line(A_p, A)
        P = intersection_of_lines(l_perpA, l_CB)
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names
        diagram.add_point(P, p)

        A, B, C = from_names

        diagram.add_line_to_draw(P + A)
        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + B)
        diagram.add_line_to_draw(P + B)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C = from_names
        P = to_names
        return [
            Predicate("perp", [P, A, B]),
            Predicate("eqline", [B, C, B, P])
        ]

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, C = from_names
        l = Segment(B, C)
        l_rep = diagram.database.inverse_eqline[l]
        it = iter(diagram.database.lines_points[l_rep])
        pC = next(it)
        if pC == B:
            pC = next(it)
        if all(p in diagram.parent_points for p in [A, B, pC]) or C != pC:
            return False, EXIST
        return True, SSUCCESS

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        froms = [diagram.point_dict[name] for name in from_names]
        A, B, C = froms
        if on_same_line(*froms) or perp(Line(B, A), Line(C, B)) or perp(
                Line(B, A), Line(C, A)):
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
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C = from_names
        l = Segment(B, C)
        l_rep = diagram.database.inverse_eqline[l]
        it = iter(diagram.database.lines_points[l_rep])
        pC = next(it)
        if pC == B:
            pC = next(it)
        new_from_names = A + B + pC
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


class Parallel(Constructor):
    """Constructor for a parallel line."""
    from_names_len = 4
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C, D = froms
        A_p = C - B + A
        l_DC = Line(D, C)
        l_paraA = Line(A_p, A)
        P = intersection_of_lines(l_paraA, l_DC)
        return (P, )

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output
        p = to[0]

        P = to_names
        diagram.add_point(P, p)

        A, B, C, D = from_names

        diagram.add_line_to_draw(P + A)
        diagram.add_line_to_draw(C + B)
        diagram.add_line_to_draw(D + C)
        diagram.add_line_to_draw(P + C)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[str]:
        A, B, C, D = from_names
        P = to_names
        return [
            Predicate("para", [A, P, B, C]),
            Predicate("eqline", [C, D, C, P])
        ]

    @staticmethod
    def good_to_rep(diagram: 'Diagram', from_names: str) -> bool:
        A, B, C, D = from_names
        para_l = Segment(B, C)
        para_l_rep = diagram.database.inverse_eqline[para_l]
        it = iter(diagram.database.lines_points[para_l_rep])
        pB = next(it)
        if pB == C:
            pB = next(it)
        its_l = Segment(C, D)
        its_l_rep = diagram.database.inverse_eqline[its_l]
        it = iter(diagram.database.lines_points[its_l_rep])
        pD = next(it)
        if pD == C:
            pD = next(it)
        if all(p in diagram.parent_points
               for p in [A, pB, C, pD]) or B != pB or D != pD:
            return False, EXIST
        return True, SSUCCESS

    @staticmethod
    def good_to_draw(diagram: 'Diagram', from_names: str,
                     draw_only: bool) -> Tuple[bool, int]:
        froms = [diagram.point_dict[name] for name in from_names]
        A, B, C, D = froms
        if on_same_line(A, C, D) or on_same_line(A, B, C) or on_same_line(
                B, C, D) or parallel(Line(A, D), Line(B, C)):
            return False, SAMELINE

        return True, SSUCCESS

    @staticmethod
    def new_from_names(diagram: 'Diagram',
                       new_points: str = "") -> List[Tuple]:
        list_of_four_points = []
        for num_new in range(1, 5):
            num_old = 4 - num_new
            from_old = itertools.combinations(diagram.parent_points, num_old)
            from_new = itertools.combinations(new_points, num_new)
            combs = itertools.product(from_old, from_new)
            for comb in combs:
                list_of_four_points += itertools.permutations(
                    comb[0] + comb[1], 4)
        return list_of_four_points

    @staticmethod
    def convert(diagram: 'Diagram', from_names: str,
                to_names: str) -> Tuple[str, str]:
        A, B, C, D = from_names
        para_l = Segment(B, C)
        para_l_rep = diagram.database.inverse_eqline[para_l]
        it = iter(diagram.database.lines_points[para_l_rep])
        pB = next(it)
        if pB == C:
            pB = next(it)
        its_l = Segment(C, D)
        its_l_rep = diagram.database.inverse_eqline[its_l]
        it = iter(diagram.database.lines_points[its_l_rep])
        pD = next(it)
        if pD == C:
            pD = next(it)
        new_from_names = A + pB + C + pD
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


# class Connect(Constructor):
#     """Connect two points with a line."""
#     from_names_len = 2
#     to_names_len = 0

#     @staticmethod
#     def construct(diagram: 'Diagram', from_names: str, to_names: str,
#                   output: Tuple) -> Tuple:
#         A, B = from_names
#         diagram.add_line_to_draw(B + A)
#         return ()

#     @staticmethod
#     def given_facts(from_names: str, to_names: str) -> List[Predicate]:
#         return []

#     @staticmethod
#     def new_from_names(diagram: 'Diagram', new_points: str = "") -> List[Tuple]:
#         list_of_two_points = []
#         list_of_two_points += itertools.combinations(new_points, 2)
#         list_of_two_points += itertools.product(diagram.parent_points,
#                                                 new_points)
#         return list_of_two_points

# class EqualLengthPlus(Constructor):
#     """Find point P on a line that is equal to a prespecified distance.

#     Pick the right point.
#     """

#     @staticmethod
#     def compute(froms: List) -> Point:
#         A, l, B, C = froms

#         dist_BC = ((B.vector - C.vector)**2).sum()
#         nx, ny, c = l.nx, l.ny, l.c

#         if isclose(ny, 0):
#             x = -c / nx
#             y = A.y + math.sqrt(dist_BC - (x - A.x)**2)
#             P = Point(x, y)
#         else:
#             eq_a = 1 + (nx / ny)**2
#             eq_b = 2 * nx / ny * (c / ny + A.y) - 2 * A.x
#             eq_c = A.x**2 + (A.y + c / ny)**2 - dist_BC

#             x = (-eq_b + math.sqrt(eq_b**2 - 4 * eq_a * eq_c)) / (2 * eq_a)
#             P = Point(x, (-c - nx * x) / ny)
#         return P

#     @staticmethod
#     def construct(diagram: 'Diagram', from_names: str, to_names: str):
#         valid, output, message_id = EqualLengthPlus.static_check(
#             diagram, from_names, to_names)
#         if not valid:
#             raise AssertionError(ErrorMessages[message_id])
#         to, froms = output
#         A, l, B, C = froms
#         to.name = to_names

#         diagram.add_line_to_draw(l)
#         diagram.add_point(to)
#         diagram.add_point_on_line(to, l)
#         diagram.get_line([to, A])
#         diagram.get_line([B, C])
#         diagram.lines_from_point(to)

#     @staticmethod
#     def given_facts(from_names: str, to_names: str) -> List[str]:
#         to = to_names.lower()
#         fro = from_names.lower()
#         return [
#             f"pointLiesOnLine({to},line({fro[1]},{fro[2]})).",
#             f"equals(lengthOf(line({fro[0]},{to})),lengthOf(line({fro[3]},{fro[4]}))).",
#         ]

#     @staticmethod
#     def static_check(
#             diagram: 'Diagram', from_names: str,
#             to_names: str) -> Tuple[bool, int]:

#         # check input
#         if len(from_names) != 5 or len(set(from_names[1:3])) != 2 or len(
#                 set(from_names[3:])) != 2:
#             return False, None, DISTINCT
#         if len(to_names) != 1:
#             return False, None, INVALID
#         if not all(p_name in diagram.point_dict
#                    for p_name in from_names):
#             return False, None, POINT
#         if to_names in diagram.point_dict:
#             return False, None, USED

#         # check action duplicate
#         for action in diagram.actions:
#             if action.constructor_name == "EqualLengthPlus":
#                 old_from_name = action.from_names
#                 if old_from_name[0] == from_names[0] and len(
#                         set.intersection(*[
#                             diagram.point_dict[p_name][1]
#                             for p_name in old_from_name[1:3] + from_names[1:3]
#                         ])) > 0 and (from_names[3:] == old_from_name[3:]
#                                      or from_names[3:][::-1]
#                                      == old_from_name[3:]):
#                     return False, None, DUPLICATE

#         # check if can be drawn
#         froms = [
#             diagram.get_point_by_name(from_names[0]),
#             diagram.get_line([
#                 diagram.get_point_by_name(from_names[1]),
#                 diagram.get_point_by_name(from_names[2])
#             ],
#                              draw=False),
#             diagram.get_point_by_name(from_names[3]),
#             diagram.get_point_by_name(from_names[4])
#         ]
#         A, l, B, C = froms

#         if A.name in diagram.line_has_points[l.name][1]:
#             return False, None, SAMELINE

#         dist_BC = ((B.vector - C.vector)**2).sum()
#         nx, ny, c = l.nx, l.ny, l.c
#         if isclose(ny, 0):
#             x = -c / nx
#             d = dist_BC - (x - A.x)**2
#             if d < 0:
#                 return False, None, CANNOT
#             y = A.y + math.sqrt(dist_BC - (x - A.x)**2)
#             to = Point(x, y)
#         else:
#             eq_a = 1 + (nx / ny)**2
#             eq_b = 2 * nx / ny * (c / ny + A.y) - 2 * A.x
#             eq_c = A.x**2 + (A.y + c / ny)**2 - dist_BC
#             d = eq_b**2 - 4 * eq_a * eq_c
#             if d < 0:
#                 return False, None, CANNOT
#             x = (-eq_b + math.sqrt(eq_b**2 - 4 * eq_a * eq_c)) / (2 * eq_a)
#             to = Point(x, (-c - nx * x) / ny)

#         # check existing facts
#         if to in diagram.point_dict.values():
#             return False, None, EXIST

#         return True, (to, froms), SUCCESS

#     @staticmethod
#     def new_from_names(
#         diagram: 'Diagram',
#         new_points: Optional[str] = None
#     ) -> List[Action]:
#         valid = []
#         if new_points is not None:
#             if not new_points:
#                 return []
#             list_of_five_points = []
#             segments_all = itertools.combinations(
#                 diagram.parent_points + list(new_points), 2)
#             segments_old = itertools.combinations(diagram.parent_points, 2)
#             for new_p in new_points:
#                 for two_segments in itertools.product(segments_all,
#                                                       segments_all):
#                     if new_p not in two_segments[0]:
#                         list_of_five_points.append((new_p, ) +
#                                                    two_segments[0] +
#                                                    two_segments[1])
#             for old_p in diagram.parent_points:
#                 for two_segments in itertools.product(
#                         segments_all, segments_all) - itertools.product(
#                             segments_old, segments_old):
#                     list_of_five_points.append((old_p, ) + two_segments[0] +
#                                                two_segments[1])
#         else:
#             list_of_five_points = []
#             segments_all = itertools.combinations(
#                 diagram.point_dict.keys(), 2)
#             for anchor in diagram.point_dict:
#                 for two_segments in itertools.product(segments_all,
#                                                       segments_all):
#                     if anchor not in two_segments[0]:
#                         list_of_five_points.append((anchor, ) +
#                                                    two_segments[0] +
#                                                    two_segments[1])
#         to_names = diagram.next_point_name()
#         for five_points in list_of_five_points:
#             from_names = "".join(five_points)
#             if EqualLengthPlus.static_check(diagram, from_names, to_names)[0]:
#                 valid.append(Action(EqualLengthPlus, from_names, to_names))
#         return valid
