r"""Basic primitives, Point, Line, Circle, for geometry problems."""

import math
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np

from tonggeometry.util import isclose


class SamePointError(Exception):
    """Exception for same points."""

    def __init__(self, message="Two points are the same."):
        self.message = message
        super().__init__(self.message)


class Point:
    """The Point class."""
    # for numpy to respect right hand side operations
    __array_priority__ = 10000

    def __init__(self, *params, name: Optional[str] = None):
        if len(params) == 1:
            self.vector = np.array(params[0], dtype=np.float64)
            self.x = float(self.vector[0])
            self.y = float(self.vector[1])
        elif len(params) == 2:
            self.x, self.y = params
            self.vector = np.array([self.x, self.y], dtype=np.float64)
        else:
            raise ValueError("Invalid parameters for a point!")
        self.name = name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Point):
            return isclose(self.x, other.x, 1e-1) and isclose(
                self.y, other.y, 1e-1)
        return False

    def __add__(self, other: Any) -> 'Point':
        if isinstance(other, Point):
            return Point(self.vector + other.vector)
        if (isinstance(other, np.ndarray)
                and len(other.shape) == 1) or isinstance(other, (float, int)):
            return Point(self.vector + other)
        raise ValueError("Invalid argument for a point to add!")

    def __radd__(self, other: Any) -> 'Point':
        if (isinstance(other, np.ndarray)
                and len(other.shape) == 1) or isinstance(other, (float, int)):
            return Point(other + self.vector)
        raise ValueError("Invalid argument for a point to add!")

    def __sub__(self, other: Any) -> 'Point':
        if isinstance(other, Point):
            return Point(self.vector - other.vector)
        if (isinstance(other, np.ndarray)
                and len(other.shape) == 1) or isinstance(other, (float, int)):
            return Point(self.vector - other)
        raise ValueError("Invalid argument for a point to sub!")

    def __rsub__(self, other: Any) -> 'Point':
        if (isinstance(other, np.ndarray)
                and len(other.shape) == 1) or isinstance(other, (float, int)):
            return Point(other - self.vector)
        raise ValueError("Invalid argument for a point to sub!")

    def __mul__(self, other: Any) -> 'Point':
        if isinstance(other, Point):
            return Point(self.vector * other.vector)
        if isinstance(other, np.ndarray) and len(other.shape) == 2:
            return Point(np.dot(self.vector, other))
        if (isinstance(other, np.ndarray)
                and len(other.shape) == 1) or isinstance(other, (float, int)):
            return Point(self.vector * other)
        raise ValueError("Invalid argument for a point to mul!")

    def __rmul__(self, other: Any) -> 'Point':
        if isinstance(other, np.ndarray) and len(other.shape) == 2:
            return Point(np.dot(other, self.vector))
        if (isinstance(other, np.ndarray)
                and len(other.shape) == 1) or isinstance(other, (float, int)):
            return Point(other * self.vector)
        raise ValueError("Invalid argument for a point to mul!")

    def __truediv__(self, other: Any) -> 'Point':
        if isinstance(other, Point):
            return Point(self.vector / other.vector)
        if (isinstance(other, np.ndarray)
                and len(other.shape) == 1) or isinstance(other, (float, int)):
            return Point(self.vector / other)
        raise ValueError("Invalid argument for a point to div!")

    def __repr__(self) -> str:
        return f"Point {self.name}: ({self.x},{self.y})"

    def cross(self, other: 'Point') -> float:
        """Cross product.

        Cross product Ax_b = det([A;B]) = A.x * B.y - A.y * B.x =
        |A| * |B| * sin(t), t from A to B, in right-hand rule
        """
        # float(np.cross(self.vector, other.vector)) DeprecationWarning
        return float(self.x * other.y - self.y * other.x)

    def dot(self, other: 'Point') -> float:
        """Dot product."""
        return float(np.dot(self.vector, other.vector))

    def norm(self) -> float:
        """L_2 norm of the vector."""
        return float(np.linalg.norm(self.vector))

    def __lt__(self, other: 'Point') -> bool:
        """Sort points. Two points must be different."""
        if isclose(self.x, other.x):
            return self.y < other.y
        return self.x < other.x


class Line:
    """The Line class."""

    def __init__(self, *params, name: Optional[str] = None):
        if len(params) == 3:
            nx, ny, c = params
            self.nx, self.ny, self.c = self.normalize(nx, ny, c)
        elif len(params) == 2:
            p_a, p_b = params
            self.nx, self.ny, self.c = self.from_two_points(p_a, p_b)
            if name is None:
                name = f"{p_a.name}{p_b.name}"
        else:
            raise ValueError("Invalid parameters for a line!")
        self.name = name
        # The point on line closest to the origin
        self.point = np.array([-self.c * self.nx, -self.c * self.ny],
                              dtype=np.float64)
        # Normal vector of the line, pointing to the positive x orthant
        self.norm = np.array([self.nx, self.ny], dtype=np.float64)
        # Directional vector of the line
        self.vector = np.array([self.ny, -self.nx], dtype=np.float64)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Line):
            return isclose(self.nx, other.nx) and isclose(
                self.ny, other.ny) and isclose(self.c, other.c)
        return False

    def __repr__(self) -> str:
        return f"Line {self.name}: ({self.nx},{self.ny},{self.c})"

    def __contains__(self, point: Point) -> bool:
        return isclose(self.nx * point.x + self.ny * point.y + self.c, 0)

    def normalize(self, nx: float, ny: float,
                  c: float) -> Tuple[float, float, float]:
        """Normalize parameters of a line.

        Normalize the normal vector to have unit length and
        in the positive x orthant (excluding negative y).
        """
        norm = math.sqrt(nx**2 + ny**2)
        if norm == 0:
            raise SamePointError

        if nx < 0 or (nx == 0 and ny < 0):
            norm = -norm

        return nx / norm, ny / norm, c / norm

    def from_two_points(self, p_a: Point,
                        p_b: Point) -> Tuple[float, float, float]:
        """Initialize a line from two points.

        Use a normal vector to define a line: nx * (x - x_a) + ny * (y - y_a) = 0.
        Trivial soluiton: nx = y_b - y_a, ny = x_a - x_b, such that A, B on line.
        c = -x_a * nx - y_a * ny.
        If (nx, ny, c) normalized, (-c * nx, -c * ny) is the point on the line
        closest to the origin and |c| is the distance from origin to the line.
        """
        x_a, y_a, x_b, y_b = p_a.x, p_a.y, p_b.x, p_b.y
        nx, ny = y_b - y_a, x_a - x_b
        c = -x_a * nx - y_a * ny
        return self.normalize(nx, ny, c)

    def project(self, p: Union[np.ndarray, Point]) -> Point:
        """Project a point to the line."""
        if isinstance(p, Point):
            p = p.vector
        return Point(self.point + self.vector *
                     (p - self.point).dot(self.vector))


class Circle:
    """The Circle class."""

    def __init__(self, *params, name: Optional[str] = None):
        if len(params) == 3:
            self.x, self.y, self.r = params
            self.center = np.array([self.x, self.y], dtype=np.float64)
        elif len(params) == 2 and isinstance(params[1], (float, int)):
            center, self.r = params
            center_vec = center.vector if isinstance(center, Point) else center
            self.center = np.array(center_vec, dtype=np.float64)
            self.x = float(self.center[0])
            self.y = float(self.center[1])
        elif len(params) == 2 and isinstance(params[1], (Point, np.ndarray)):
            center, circum = params
            center_vec = center.vector if isinstance(center, Point) else center
            circum_vec = circum.vector if isinstance(circum, Point) else circum
            self.center = np.array(center_vec, dtype=np.float64)
            self.r = np.linalg.norm(self.center - circum_vec)
            self.x = float(self.center[0])
            self.y = float(self.center[1])
            if name is None and isinstance(center, Point) and isinstance(
                    circum, Point):
                name = f"{center.name}{circum.name}"
        else:
            raise ValueError("Invalid parameters for a circle!")
        self.name = name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Circle):
            return isclose(self.x, other.x) and isclose(
                self.y, other.y) and isclose(self.r, other.r)
        return False

    def __repr__(self) -> str:
        return f"Circle {self.name}: ({self.x},{self.y},{self.r})"

    def __contains__(self, point: Point) -> bool:
        return isclose(np.linalg.norm(self.center - point.vector), self.r)


def get_angle(A: Union[Point, np.ndarray],
              B: Union[Point, np.ndarray],
              acute: bool = False) -> float:
    """Get the angle between two vectors."""
    if isinstance(A, Point):
        A = A.vector
    if isinstance(B, Point):
        B = B.vector
    cos_value = np.clip(
        np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)), -1, 1)
    if acute:
        cos_value = abs(cos_value)
    return math.acos(cos_value)


def angle_type(A: Union[Point, np.ndarray], B: Union[Point,
                                                     np.ndarray]) -> int:
    """Angle type of AOB; Acute 1, Perp 0, Obtuse -1."""
    if isinstance(A, Point):
        A = A.vector
    if isinstance(B, Point):
        B = B.vector
    dot_value = np.dot(A, B)
    if isclose(dot_value, 0):
        return 0
    if dot_value > 0:
        return 1
    return -1


def on_same_line(A: Point, B: Point, C: Point, eps=1e-5) -> bool:
    """Check if three points are on the same line."""
    if A == B or B == C or A == C:
        return True
    vec_AB = B - A
    vec_AB = vec_AB / vec_AB.norm()
    vec_BC = C - B
    vec_BC = vec_BC / vec_BC.norm()
    return isclose(vec_AB.cross(vec_BC), 0, eps=eps)


def parallel(AB: Line, CD: Line) -> bool:
    """Check if two lines are parallel."""
    return isclose(AB.nx * CD.ny, AB.ny * CD.nx)


def perp(AB: Line, CD: Line) -> bool:
    """Check if two lines are perpendicular."""
    return isclose(AB.norm.dot(CD.norm), 0)


def ccw(A, B, C):
    """
    ccw: counterclockwise, check if C is on the left of AB
    ref: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
    """
    AC = C - A
    AC = AC / AC.norm()
    AB = B - A
    AB = AB / AB.norm()
    val = AB.cross(AC)
    return val


def intersect(A: Point,
              B: Point,
              C: Point,
              D: Point,
              touch_as_intersect=False) -> bool:
    """Check if two line segments intersect.

    [A, B, C, D], checks if AB intersects with CD.
    """

    if touch_as_intersect:
        if A in [C, D] or B in [C, D]:
            return True
        ACD = ccw(A, C, D)
        BCD = ccw(B, C, D)
        ABC = ccw(A, B, C)
        ABD = ccw(A, B, D)
        bool_val_CDAB = isclose(ACD, 0) or isclose(
            BCD, 0) or (ACD > 0 > BCD) or (ACD < 0 < BCD)
        bool_val_ABCD = isclose(ABC, 0) or isclose(
            ABD, 0) or (ABC > 0 > ABD) or (ABC < 0 < ABD)
        return bool_val_CDAB and bool_val_ABCD

    if A in [C, D] or B in [C, D]:
        return False
    ACD = ccw(A, C, D)
    BCD = ccw(B, C, D)
    ABC = ccw(A, B, C)
    ABD = ccw(A, B, D)
    bool_val_CDAB = (ACD > 0 > BCD) or (ACD < 0 < BCD)
    bool_val_ABCD = (ABC > 0 > ABD) or (ABC < 0 < ABD)
    return bool_val_CDAB and bool_val_ABCD


def intersection_of_lines(AB: Line, CD: Line) -> Point:
    """Compute the intersection of two lines.

    nx1 * x + ny1 * y + c1 = 0
    nx2 * x + ny2 * y + c2 = 0

    (nx1 * ny2 - nx2 * ny1) * x + (c1 * ny2 - c2 * ny1) = 0
    (ny1 * nx2 - ny2 * nx1) * y + (c1 * nx2 - c2 * nx1) = 0
    """
    nx_1, ny_1, c_1 = AB.nx, AB.ny, AB.c
    nx_2, ny_2, c_2 = CD.nx, CD.ny, CD.c

    x = -(c_1 * ny_2 - c_2 * ny_1) / (nx_1 * ny_2 - nx_2 * ny_1)
    y = -(c_1 * nx_2 - c_2 * nx_1) / (ny_1 * nx_2 - ny_2 * nx_1)

    return Point(x, y)


def intersection_of_circles(OA: Circle, OB: Circle) -> Optional[Tuple[Point]]:
    """Compute the intersections of two circles. From the center of OA to OB,
    the left intersection is the first."""
    x_1, y_1, r_1 = OA.x, OA.y, OA.r
    x_2, y_2, r_2 = OB.x, OB.y, OB.r

    d = np.linalg.norm(OA.center - OB.center)
    if isclose(d, 0):
        return [None]
    if isclose(d, r_1 + r_2) or isclose(d, abs(r_1 - r_2)):
        return [Point(OA.center + (OB.center - OA.center) * r_1 / d)]
    if d > r_1 + r_2:
        return [None]
    if d < abs(r_1 - r_2):
        return [None]
    # Law of Cosines
    a = (r_1**2 - r_2**2 + d**2) / (2 * d)
    h = math.sqrt(r_1**2 - a**2)
    x_mid = x_1 + a * (x_2 - x_1) / d
    y_mid = y_1 + a * (y_2 - y_1) / d
    x_3 = x_mid - h * (y_2 - y_1) / d
    y_3 = y_mid + h * (x_2 - x_1) / d
    x_4 = x_mid + h * (y_2 - y_1) / d
    y_4 = y_mid - h * (x_2 - x_1) / d
    p3, p4 = Point(x_3, y_3), Point(x_4, y_4)
    return [p3, p4]


def intersection_of_line_circle(A: Point, B: Point,
                                OC: Circle) -> Optional[Tuple[Point]]:
    """Compute the intersections of a line and a circle. The first intersection
    from A to B is returned first."""
    AB = B.vector - A.vector
    AO = OC.center - A.vector
    unit_AB = AB / np.linalg.norm(AB)
    p_proj = A.vector + unit_AB.dot(AO) * unit_AB
    d = np.linalg.norm(OC.center - p_proj)
    if isclose(d, OC.r):
        return [Point(p_proj)]
    if d > OC.r:
        return [None]
    h = math.sqrt(OC.r**2 - d**2)
    p1, p2 = Point(p_proj - h * unit_AB), Point(p_proj + h * unit_AB)
    return [p1, p2]


def same_dir(A: Point, B: Point, C: Point, D: Point, E: Point,
             F: Point) -> bool:
    """Check if from vec_AB to vec_BC is the same as vec_DE to vec_EF."""
    BA = A - B
    BA = BA / BA.norm()
    BC = C - B
    BC = BC / BC.norm()
    ED = D - E
    ED = ED / ED.norm()
    EF = F - E
    EF = EF / EF.norm()
    cross_BAC = BA.cross(BC)
    dot_BAC = BA.dot(BC)
    cross_EDF = ED.cross(EF)
    dot_EDF = ED.dot(EF)
    if isclose(cross_BAC, 0) and isclose(cross_EDF, 0):
        if isclose(dot_BAC, 0) or isclose(dot_EDF, 0):
            return 0
        if dot_BAC > 0 and dot_EDF > 0 or dot_BAC < 0 and dot_EDF < 0:
            return 1
        return -1
    if isclose(cross_BAC, 0) or isclose(cross_EDF, 0):
        return 0
    if cross_BAC > 0 and cross_EDF > 0 or cross_BAC < 0 and cross_EDF < 0:
        return 1
    return -1


def signed_area(points: Iterable[Point]):
    """The signed area formed by directed winding of the points.

    Area < 0 means counter-clockwise. Area > 0 clockwise. Area == 0, closed
    loop like an 8 shape.

    https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    """
    total_area = 0
    for i, p_i in enumerate(points):
        p_i1 = points[i - 1]
        total_area += (p_i.y + p_i1.y) * (p_i.x - p_i1.x)
    return total_area
