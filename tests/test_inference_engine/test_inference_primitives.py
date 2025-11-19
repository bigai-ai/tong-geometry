r"""Test the primitives in the inference engine."""

from tonggeometry.inference_engine.primitives import (Angle, Circle, Ratio,
                                                      Segment, Triangle)


def test_line():
    """Test the line primitive. Segment is Segment."""
    s1 = Segment("A", "B")
    s2 = Segment("B", "A")
    s3 = Segment("C", "B")
    assert s1.dependency == "AB"
    assert s1 == s2
    assert str(s1) == str(s2) == "AB"
    assert str(s3) == "BC"
    assert len(set([s1, s2, s3])) == 2
    assert sorted([s1, s2, s3]) == [s1, s2, s3]


def test_angle():
    """Test the angle primitive."""
    a1 = Angle("D", "E", "F")
    a2 = Angle("F", "E", "D")
    a3 = Angle("C", "A", "B")
    assert a1.dependency == "DEF"
    assert a1 == a2
    assert str(a1) == str(a2) == "DEF"
    assert str(a3) == "BAC"
    assert len(set([a1, a2, a3])) == 2
    assert sorted([a1, a2, a3]) == [a3, a1, a2]


def test_ratio():
    """Test the ratio primitive."""
    s1 = Segment("D", "E")
    s2 = Segment("F", "E")
    s3 = Segment("C", "A")
    s4 = Segment("B", "A")
    ratio1 = Ratio(s1, s2)
    ratio2 = Ratio(s2, s1)  # pylint: disable=arguments-out-of-order
    ratio3 = Ratio(s3, s4)
    assert ratio1.dependency == "DEF"
    assert ratio1 == ratio2
    assert str(ratio1) == str(ratio2) == "DE, EF"
    assert str(ratio3) == "AB, AC"
    assert len(set([ratio1, ratio2, ratio3])) == 2
    assert sorted([ratio1, ratio2, ratio3]) == [ratio3, ratio1, ratio2]


def test_triangle():
    """Test the triangle primitive."""
    t1 = Triangle("B", "A", "C")
    t2 = Triangle("A", "C", "B")
    t3 = Triangle("D", "F", "E")
    assert t1.dependency == "ABC"
    assert t1 == t2
    assert str(t1) == str(t2) == "ABC"
    assert str(t3) == "DEF"
    assert len(set([t1, t2, t3])) == 2
    assert sorted([t1, t2, t3]) == [t1, t2, t3]


def test_circle():
    """Test the circle primitive."""
    c1 = Circle("A", ["B", "C", "E", "D", "Y", "X"])
    c2 = Circle(None, ["C", "D", "Y", "X", "E", "B"])
    c3 = Circle("A", ["C", "D", "Y", "X", "E", "B"])
    c4 = Circle("U", ["M", "O", "N"])
    c5 = Circle(None, ["C", "D", "Y", "X", "E", "B", "A"])
    assert c1.dependency == "ABCDEXY"
    assert c2.dependency == "BCDEXY"
    assert c1 != c2
    assert c1 == c3
    assert str(c1) == str(c3) == "A, BCDEXY"
    assert str(c2) == "None, BCDEXY"
    assert str(c4) == "U, MNO"
    assert len(set([c1, c2, c3, c4, c5])) == 4
    assert sorted([c1, c2, c3, c4, c5]) == [c1, c3, c4, c5, c2]
