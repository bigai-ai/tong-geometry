r"""Define base constructors. Note that coll needs to be defined."""

import math
from typing import TYPE_CHECKING, List, Tuple

from tonggeometry.constructor.parent import Constructor
from tonggeometry.constructor.primitives import Circle, Point
from tonggeometry.inference_engine.predicate import Predicate

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram

__all__ = [
    "BaseAcuteTriangle",
    "BaseParallelogram",
    "BaseInscribedQuad",
    "BaseInscribedTri",
    "BaseHarmonicQuad",
]


class BaseAcuteTriangle(Constructor):
    """Base acute triangle constructor."""
    from_names_len = 0
    to_names_len = 3

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A = Point(math.cos(-40.0 / 180 * math.pi),
                  math.sin(-40.0 / 180 * math.pi)) * 100
        B = Point(math.cos(-140.0 / 180 * math.pi),
                  math.sin(-140.0 / 180 * math.pi)) * 100
        C = Point(math.cos(75.0 / 180 * math.pi), math.sin(
            75.0 / 180 * math.pi)) * 100
        return A, B, C

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        for p, name in zip(to, to_names):
            diagram.add_point(name, p)

        A, B, C = to_names

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + A)
        diagram.add_line_to_draw(C + B)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        return []


class BaseParallelogram(Constructor):
    """Base parallelogram constructor."""
    from_names_len = 0
    to_names_len = 4

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A = Point(3.6, 4.2) * 100
        B = Point(0, 0) * 100
        C = Point(12, 0) * 100
        D = Point(15.6, 4.2) * 100
        return A, B, C, D

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        for p, name in zip(to, to_names):
            diagram.add_point(name, p)

        A, B, C, D = to_names

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + B)
        diagram.add_line_to_draw(D + A)
        diagram.add_line_to_draw(D + C)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C, D = to_names
        return [
            Predicate("para", [A, B, C, D]),
            Predicate("cong", [A, B, C, D]),
            Predicate("para", [A, D, B, C]),
            Predicate("cong", [A, D, B, C]),
        ]


class BaseInscribedTri(Constructor):
    """Base inscribed triangle constructor."""
    from_names_len = 0
    to_names_len = 4

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A = Point(math.cos(-0.05 * 2 * math.pi), math.sin(
            -0.05 * 2 * math.pi)) * 100
        B = Point(math.cos(0.29 * 2 * math.pi), math.sin(
            0.29 * 2 * math.pi)) * 100
        C = Point(math.cos(-0.45 * 2 * math.pi), math.sin(
            -0.45 * 2 * math.pi)) * 100
        D = Point(0, 0)
        return A, B, C, D

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        for p, name in zip(to, to_names):
            diagram.add_point(name, p)

        A, B, C, D = to_names

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + A)
        diagram.add_line_to_draw(C + B)

        c = Circle(to[3], to[0])
        diagram.add_circle(D + A, c)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C, D = to_names
        return [Predicate("eqcircle", [[D, A], [None, A, B, C]])]


class BaseInscribedQuad(Constructor):
    """Base inscribed quadrilateral constructor."""
    from_names_len = 0
    to_names_len = 5

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A = Point(math.cos(0.1 * 2 * math.pi), math.sin(
            0.1 * 2 * math.pi)) * 100
        B = Point(math.cos(0.37 * 2 * math.pi), math.sin(
            0.37 * 2 * math.pi)) * 100
        C = Point(math.cos(0.57 * 2 * math.pi), math.sin(
            0.57 * 2 * math.pi)) * 100
        D = Point(math.cos(0.92 * 2 * math.pi), math.sin(
            0.92 * 2 * math.pi)) * 100
        E = Point(0, 0)
        return A, B, C, D, E

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        for p, name in zip(to, to_names):
            diagram.add_point(name, p)

        A, B, C, D, E = to_names

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + B)
        diagram.add_line_to_draw(D + A)
        diagram.add_line_to_draw(D + C)

        c = Circle(to[4], to[0])
        diagram.add_circle(E + A, c)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C, D, E = list(to_names)
        return [Predicate("eqcircle", [[E, A], [None, B, C, D]])]


class BaseHarmonicQuad(Constructor):
    """Base harmonic quadrilateral constructor."""
    from_names_len = 0
    to_names_len = 5

    @staticmethod
    def compute(froms: List[Point]) -> Tuple[Point]:
        A = Point(math.cos(0.37 * math.pi), math.sin(0.37 * math.pi)) * 100
        B = Point(-0.49521363128689805, 0.8687712353592538) * 100
        C = Point(-math.cos(0.1 * math.pi), -math.sin(0.1 * math.pi)) * 100
        D = Point(math.cos(0.1 * math.pi), -math.sin(0.1 * math.pi)) * 100
        E = Point(0, 0)
        return A, B, C, D, E

    @staticmethod
    def construct(diagram: 'Diagram', from_names: str, to_names: str,
                  output: Tuple):
        to = output

        for p, name in zip(to, to_names):
            diagram.add_point(name, p)

        A, B, C, D, E = to_names

        diagram.add_line_to_draw(B + A)
        diagram.add_line_to_draw(C + B)
        diagram.add_line_to_draw(D + A)
        diagram.add_line_to_draw(D + C)

        c = Circle(to[4], to[0])
        diagram.add_circle(E + A, c)

    @staticmethod
    def given_facts(from_names: str, to_names: str) -> List[Predicate]:
        A, B, C, D, E = to_names
        return [
            Predicate("eqratio", [A, B, A, D, B, C, D, C]),
            Predicate("eqcircle", [[E, A], [None, B, C, D]])
        ]
