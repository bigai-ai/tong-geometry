r"""Define constructors for parallelograms."""

import itertools
from typing import TYPE_CHECKING, List, Tuple

from tonggeometry.constructor.parent import SAMELINE, SSUCCESS, Constructor
from tonggeometry.constructor.primitives import Point, on_same_line
from tonggeometry.inference_engine.predicate import Predicate

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram

__all__ = ["Parallelogram"]


class Parallelogram(Constructor):
    """Construct a parallelogram.

    From [A, B, C] find point P such that PABC is a parallelogram
    vec(AP) = vec(BC).

    Totol number of actions 3 * C_k^3.
    """
    from_names_len = 3
    to_names_len = 1

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A, B, C = froms
        P = A + C - B
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
        diagram.add_line_to_draw(C + B)
        diagram.add_line_to_draw(P + A)
        diagram.add_line_to_draw(P + C)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C = from_names
        P = to_names
        return [
            Predicate("para", [A, B, C, P]),
            Predicate("cong", [A, B, C, P]),
            Predicate("para", [A, P, B, C]),
            Predicate("cong", [A, P, B, C])
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
