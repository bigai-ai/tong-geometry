r"""Test the primitives module."""

import numpy as np

from tonggeometry.constructor.primitives import (
    Circle, Line, Point, angle_type, get_angle, intersect,
    intersection_of_circles, intersection_of_line_circle,
    intersection_of_lines, on_same_line, parallel, perp, same_dir)
from tonggeometry.util import isclose


def unique(a_list):
    """Return a list of unique elements in the original order."""
    a_unique_list = []
    for item in a_list:
        if item not in a_unique_list:
            a_unique_list.append(item)
    return a_unique_list


def test_point_constructors():
    """Test the constructors of the Point class."""
    p_a = Point(1, 2, name="A")
    p_b = Point(np.array([0.5, 0.3]), name="B")
    print(p_a)
    print(p_b)


def test_point_op():
    """Test the operators of the Point class."""
    p_a = Point(1, 2, name="A")
    p_bp = Point(1, 2, name="B")
    p_bn = np.array([1, 2])

    assert p_a + p_bp == Point(2, 4)
    assert p_a + p_bn == Point(2, 4)
    assert p_bn + p_a == Point(2, 4)
    assert p_a + 1 == Point(2, 3)
    assert 1 + p_a == Point(2, 3)

    assert p_a - p_bp == Point(0, 0)
    assert p_a - p_bn == Point(0, 0)
    assert p_a - 1 == Point(0, 1)
    assert p_bn - p_a == Point(0, 0)
    assert 1 - p_a == Point(0, -1)

    assert p_bp * p_a == Point(1, 4)
    assert p_bn * p_a == Point(1, 4)
    assert np.array([[1, 1], [2, 2]]) * p_a == Point(3, 6)
    assert p_a * 2 == Point(2, 4)
    assert p_a * p_bp == Point(1, 4)
    assert p_a * p_bn == Point(1, 4)
    assert p_a * np.array([[1, 1], [2, 2]]) == Point(5, 5)
    assert 2 * p_a == Point(2, 4)

    assert p_a / p_bp == Point(1, 1)
    assert p_a / p_bn == Point(1, 1)
    assert p_a / 2 == Point(0.5, 1)

    assert p_a.cross(p_bp) == 0
    assert p_a.dot(p_bp) == 5
    assert p_a.norm() == np.sqrt(5)


def test_point_dup():
    """Test the duplication of the Point class."""
    p_a = Point(1, 2, name="A")
    p_b = Point(1, 2, name="B")
    p_c = Point(1.000000001, 2, name="C")
    p_d = Point(1, 2.0000000002, name="D")
    p_e = Point(1.0000000001, 2.0000000002, name="E")
    p_f = Point(np.array([0.5, 0.3]), name="F")
    p_g = Point(np.array([0.50000000000000000001, 0.300000000001]), name="G")

    for p_p in [p_b, p_c, p_d, p_e]:
        assert p_a == p_p

    assert p_f == p_g

    assert p_a in [p_b]
    assert p_a in [p_c]
    assert p_a in [p_d]
    assert p_a in [p_c, p_d, p_e]
    assert p_f in [p_f, p_g]

    assert len(unique([p_a, p_b, p_c, p_d, p_e, p_f, p_g])) == 2


def test_point_sort():
    """Test the sorting of the Point class."""
    p_a = Point(1, 2, name="A")
    p_b = Point(2, 4, name="B")
    p_c = Point(1.000000001, 3, name="C")
    p_d = Point(0, 3, name="D")
    assert sorted([p_a, p_b, p_c, p_d]) == [p_d, p_a, p_c, p_b]


def test_line_constructors():
    """Test the constructors of the Line class."""
    l_1 = Line(1, 2, 3, name="l1")
    p_a = Point(1, 2, name="A")
    p_b = Point(np.array([0.5, 0.3]), name="B")
    l_2 = Line(p_a, p_b, name="l2")
    print(l_1)
    print(l_2)


def test_line_dup():
    """Test the duplication of the Line class."""
    l_1 = Line(1, 2, 3, name="l1")
    l_2 = Line(1, 2, 3, name="l2")
    l_3 = Line(-1, -2, -3, name="l3")
    p_a = Point(1, 2, name="A")
    p_b = Point(np.array([0.5, 0.3]), name="B")
    l_4 = Line(p_a, p_b, name="l4")
    l_5 = Line(p_a, p_b, name="l5")

    assert l_1 == l_2
    assert l_1 == l_3
    assert l_4 == l_5


def test_line_contains():
    """Test the line contains function."""
    l_1 = Line(1, 2, 3, name="l1")
    p_1 = Point(-1, -1, name="A")
    p_2 = Point(1, 1, name="B")

    assert p_1 in l_1
    assert p_2 not in l_1


def test_line_proj():
    """Test the projection function."""
    p_1 = Point(1, 0, name="A")
    p_2 = Point(-1, 0, name="B")
    p_3 = Point(0, 1, name="C")
    p_4 = Point(0, -1, name="D")

    l_1 = Line(1, 0, 0, name="l1")
    assert l_1.project(p_1) == Point(0, 0)
    assert l_1.project(p_2) == Point(0, 0)
    assert l_1.project(p_3) == p_3
    assert l_1.project(p_4) == p_4

    l_2 = Line(1, -1, 1, name="l2")
    assert l_2.project(p_1) == p_3
    assert l_2.project(p_2) == p_2
    assert l_2.project(p_3) == p_3
    assert l_2.project(p_4) == p_2


def test_circle_constructors():
    """Test the constructors of the Circle class."""
    c_1 = Circle(1, 2, 3, name="c1")
    c_2 = Circle(np.array([1, 2]), 3, name="c2")
    print(c_1)
    print(c_2)


def test_circle_dup():
    """Test the duplication of the Circle class."""
    c_1 = Circle(1, 2, 3, name="c1")
    c_2 = Circle(1, 2, 3, name="c2")
    c_3 = Circle(np.array([1, 2]), 3, name="c3")
    c_4 = Circle(Point(0, 0), Point(0, 1), name="c4")

    assert c_1 == c_2
    assert c_1 == c_3
    assert c_1 in [c_2, c_3]
    assert len(unique([c_1, c_2, c_3])) == 1
    assert c_4.x == 0 and c_4.y == 0 and c_4.r == 1


def test_circle_contains():
    """Test the circle contains function."""
    c_1 = Circle(1, 2, 1, name="c1")
    p_1 = Point(1, 2, name="A")
    p_2 = Point(2, 2, name="B")
    p_3 = Point(1, 3, name="C")
    p_4 = Point(1 + 3**0.5 / 2, 2 + 1 / 2, name="D")
    assert p_1 not in c_1
    assert p_2 in c_1
    assert p_3 in c_1
    assert p_4 in c_1


def test_angle():
    """Test the get_angle function."""
    p_1 = Point(1, 1, name="A")
    p_2 = Point(1, 0, name="B")
    p_3 = Point(-1, 0, name="C")
    assert isclose(get_angle(p_1, p_2), np.pi / 4)
    assert isclose(get_angle(p_1, p_3), np.pi * 3 / 4)
    assert isclose(get_angle(p_1.vector, p_2), np.pi / 4)
    assert isclose(get_angle(p_1.vector, p_3, acute=True), np.pi / 4)


def test_angle_type():
    """Test the angle_type function."""
    p_1 = Point(1, 1, name="A")
    p_2 = Point(1, 0, name="B")
    p_3 = Point(-1, 0, name="C")
    p_4 = Point(-1, 1, name="D")
    assert angle_type(p_1, p_2) == 1
    assert angle_type(p_1, p_4) == 0
    assert angle_type(p_1, p_3) == -1


def test_on_same_line():
    """Test the on_same_line function."""
    p_1 = Point(0, 0, name="A")
    p_2 = Point(1, 1, name="B")
    p_3 = Point(2, 2, name="C")
    p_4 = Point(2, 3, name="D")
    assert on_same_line(p_1, p_2, p_3)
    assert not on_same_line(p_1, p_2, p_4)
    assert not on_same_line(p_2, p_3, p_4)
    assert on_same_line(p_1, p_2, p_1)
    assert on_same_line(p_3, p_3, p_3)


def test_perp():
    """Test the perpendicular function."""
    l_1 = Line(1, 2, 3, name="l1")
    l_2 = Line(4, 8, 3, name="l1")
    l_3 = Line(-8, 4, 6, name="l3")
    assert not perp(l_1, l_2)
    assert perp(l_1, l_3)
    assert perp(l_2, l_3)


def test_parallel():
    """Test the parallel function."""
    l_1 = Line(1, 2, 3, name="l1")
    l_2 = Line(4, 8, 3, name="l1")
    l_3 = Line(5, 6, 6, name="l3")
    assert parallel(l_1, l_2)
    assert not parallel(l_1, l_3)


def test_intersect():
    """Test the intersect function."""
    p_1 = Point(0, 0, name="A")
    p_2 = Point(1, 1, name="B")
    p_3 = Point(1, 0, name="C")
    p_4 = Point(0, 1, name="D")
    p_5 = Point(0.5, 0.5, name="E")
    p_6 = Point(2, 2, name="F")
    assert intersect(p_1, p_2, p_3, p_4)
    assert not intersect(p_1, p_2, p_3, p_5)
    assert intersect(p_1, p_2, p_3, p_5, touch_as_intersect=True)
    assert not intersect(p_1, p_2, p_3, p_6)


def test_intersection_of_lines():
    """Test the intersection of lines function."""
    p_1 = Point(0, 0, name="A")
    p_2 = Point(1, 1, name="B")
    p_3 = Point(1, 0, name="C")
    p_4 = Point(0, 1, name="D")
    p_5 = Point(0.5, 0.5, name="E")
    p_6 = Point(2, 2, name="F")
    assert intersection_of_lines(Line(p_1, p_2), Line(p_3,
                                                      p_4)) == Point(0.5, 0.5)
    assert intersection_of_lines(Line(p_1, p_2), Line(p_3,
                                                      p_5)) == Point(0.5, 0.5)
    assert intersection_of_lines(Line(p_1, p_2), Line(p_3, p_6)) == p_6


def test_intersection_of_circles():
    """Test the intersection of circle function."""
    c_1 = Circle(0, 0, 1, name="c1")
    c_2 = Circle(1, 0, 1, name="c2")
    c_3 = Circle(2, 0, 1, name="c3")
    c_4 = Circle(3, 0, 1, name="c4")
    c_5 = Circle(0.5, 0, 0.5, name="c5")
    assert Point(0.5, 0.866025) in intersection_of_circles(c_1, c_2)
    assert Point(0.5, -0.866025) in intersection_of_circles(c_1, c_2)
    assert Point(1, 0) in intersection_of_circles(c_1, c_3)
    assert intersection_of_circles(c_1, c_4)[0] is None
    assert intersection_of_circles(c_1, c_1)[0] is None
    assert Point(1, 0) in intersection_of_circles(c_1, c_5)
    print(intersection_of_circles(c_1, c_5))


def test_intersection_of_line_circle():
    """Test the intersection of line and circle function."""
    c_1 = Circle(0, 0, 1, name="c1")
    A = Point(0, 0)
    B = Point(1, 1)
    C = Point(-1, -1)
    D = Point(-1, 0)
    E = Point(-1, -1)
    F = Point(0, -1)
    G = Point(-1, -1)
    H = Point(0, -2)
    assert Point(0.707107, 0.707107) in intersection_of_line_circle(A, B, c_1)
    assert Point(-0.707107,
                 -0.707107) in intersection_of_line_circle(A, B, c_1)
    assert Point(-1, 0) in intersection_of_line_circle(C, D, c_1)
    assert Point(0, -1) in intersection_of_line_circle(E, F, c_1)
    assert intersection_of_line_circle(G, H, c_1)[0] is None


def test_same_dir():
    """Test the same direction function."""
    B = Point(0, 0, name="A")
    A = Point(1, 1, name="B")
    C = Point(1, 0, name="C")
    E = Point(0, 0, name="D")
    D = Point(-1, 1, name="E")
    F = Point(-1, 0, name="F")
    assert same_dir(A, B, C, D, E, F) == -1  # pylint: disable=arguments-out-of-order
    assert same_dir(A, B, C, F, E, D) == 1  # pylint: disable=arguments-out-of-order
    assert same_dir(F, B, C, F, B, F) == -1
    assert same_dir(F, B, C, C, B, F) == 1
