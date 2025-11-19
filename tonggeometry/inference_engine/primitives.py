r"""Structured representation for primitives, based on points."""

from functools import cached_property
from typing import List

from tonggeometry.util import OrderedSet


class Segment:
    """Line representation from two points."""

    def __init__(self, p1: str, p2: str):
        self.p1 = p1
        self.p2 = p2

    @cached_property
    def key(self) -> str:
        """Sorted name."""
        return "".join(sorted([self.p1, self.p2]))

    @cached_property
    def dependency(self) -> str:
        """Dependent points."""
        return self.key

    def __eq__(self, other: 'Segment') -> bool:
        return isinstance(other, Segment) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return self.key

    def __lt__(self, other: 'Segment') -> bool:
        return self.key < other.key


class Angle:
    """Angle representation for full angle from three points, directional."""

    def __init__(self, p1: str, p2: str, p3: str):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.name = "".join([p1, p2, p3])
        self.s1 = Segment(p2, p1)
        self.s2 = Segment(p2, p3)

    @cached_property
    def key(self) -> str:
        """Sorted name."""
        left, right = sorted([self.p1, self.p3])
        return left + self.p2 + right

    @cached_property
    def dependency(self) -> str:
        """Dependent points."""
        return self.key

    def __eq__(self, other: 'Angle') -> bool:
        return isinstance(other, Angle) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return self.key

    def __lt__(self, other: 'Angle') -> bool:
        return self.key < other.key


class Ratio:
    """Ratio representation from two segments."""

    def __init__(self, s1: Segment, s2: Segment):
        self.s1 = s1
        self.s2 = s2
        self.name = ", ".join([str(s1), str(s2)])

    @cached_property
    def key(self) -> str:
        """Sorted name."""
        return ", ".join(sorted([str(self.s1), str(self.s2)]))

    @cached_property
    def dependency(self) -> str:
        """Dependent points."""
        dependency = OrderedSet.fromkeys(self.s1.dependency +
                                         self.s2.dependency)
        return "".join(dependency)

    def __eq__(self, other: 'Ratio') -> bool:
        return isinstance(other, Ratio) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return self.key

    def __lt__(self, other: 'Ratio') -> bool:
        return self.key < other.key


class Triangle:
    """Triangle representation from three points."""

    def __init__(self, p1: str, p2: str, p3: str):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.name = "".join([p1, p2, p3])

    @cached_property
    def key(self) -> str:
        """Sorted name."""
        return "".join(sorted([self.p1, self.p2, self.p3]))

    @cached_property
    def dependency(self) -> str:
        """Dependent points."""
        return self.key

    def __eq__(self, other: 'Triangle') -> bool:
        return isinstance(other, Triangle) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return self.key

    def __lt__(self, other: 'Triangle') -> bool:
        return self.key < other.key


class Circle:
    """Circle representation from center and points"""

    def __init__(self, center: str, points: List[str]):
        """IMPORTANT: when creating a new centered circle, make sure the first
        point in points is the minimum"""
        self.center = center
        self.points = OrderedSet.fromkeys(points)
        if len(points) == 0:
            self.min_point = None
        else:
            self.min_point = min(points)

    @property
    def key(self) -> str:
        """Sorted name."""
        points = "".join(sorted(self.points))
        return f"{self.center}, " + points

    @cached_property
    def dependency(self) -> str:
        """Dependent points."""
        p_idx = self.key.index(" ")
        points = self.key[p_idx + 1:]
        if self.center:
            return self.center + points
        return points

    def add_point(self, point: str):
        """Add a new point into self.points."""
        self.points[point] = None
        if self.min_point is None or point < self.min_point:
            self.min_point = point

    def __eq__(self, other: 'Circle') -> bool:
        return isinstance(other, Circle) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return self.key

    def __lt__(self, other: 'Circle') -> bool:
        if self.center and other.center or (not self.center
                                            and not other.center):
            return self.key < other.key
        if self.center:
            return True
        return False
