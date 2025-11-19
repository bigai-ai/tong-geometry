r"""Test the predicate class."""

import random

from tonggeometry.inference_engine.predicate import OrderedFact, Predicate
from tonggeometry.inference_engine.primitives import (Angle, Circle, Ratio,
                                                      Segment, Triangle)


def test_eqline():
    """Test the coll predicate."""
    f1 = Predicate("eqline", ["A", "B", "B", "C"])
    f2 = Predicate("eqline", ["B", "C", "A", "B"])
    f3 = Predicate("eqline", ["X", "B", "U", "A"])
    assert f1 == f2
    assert str(f1) == str(f2) == "eqline (AB, BC)"
    assert str(f3) == "eqline (AU, BX)"
    assert len(set([f1, f2, f3])) == 2


def test_eqcircle():
    """Test the circle predicate."""
    f1 = Predicate("eqcircle", [["O", "A"], ["O", "C"]])
    f2 = Predicate("eqcircle", [["O", "C"], ["O", "A"]])
    f3 = Predicate("eqcircle",
                   [Circle(None, ["A", "B", "C"]),
                    Circle("O", ["A"])])
    f4 = Predicate("eqcircle", [["M", "X"], ["B", "U"]])
    assert f1 == f2
    assert str(f1) == str(f2) == "eqcircle (O, A, O, C)"
    assert str(f3) == "eqcircle (O, A, None, ABC)"
    assert str(f4) == "eqcircle (B, U, M, X)"
    assert len(set([f1, f2, f3, f4])) == 3


def test_cong():
    """Test the cong predicate."""
    f1 = Predicate("cong", [Segment("A", "B"), Segment("C", "D")])
    f2 = Predicate("cong", ["D", "C", "B", "A"])
    f3 = Predicate("cong", ["X", "B", "U", "D"])
    assert f1 == f2
    assert str(f1) == str(f2) == "cong (AB, CD)"
    assert str(f3) == "cong (BX, DU)"
    assert len(set([f1, f2, f3])) == 2


def test_midp():
    """Test the midp predicate."""
    f1 = Predicate("midp", ["A", Segment("B", "C")])
    f2 = Predicate("midp", ["A", Segment("C", "B")])
    f3 = Predicate("midp", ["X", Segment("B", "U")])
    f4 = Predicate("midp", ["X", "U", "B"])
    assert f1 == f2
    assert f3 == f4
    assert str(f1) == str(f2) == "midp (A, BC)"
    assert str(f3) == str(f4) == "midp (X, BU)"
    assert len(set([f1, f2, f3, f4])) == 2


def test_para():
    """Test the para predicate."""
    f1 = Predicate("para", ["A", "B", "C", "D"])
    f2 = Predicate("para", ["B", "A", "D", "C"])
    f3 = Predicate("para", ["C", "A", "B", "D"])
    f4 = Predicate("para", [Segment("D", "F"), Segment("C", "X")])
    f5 = Predicate("para", [Segment("X", "C"), Segment("F", "D")])
    f6 = Predicate("para", [Segment("C", "F"), Segment("D", "X")])
    assert f1 == f2
    assert f4 == f5
    assert str(f1) == str(f2) == "para (AB, CD)"
    assert str(f3) == "para (AC, BD)"
    assert str(f4) == str(f5) == "para (CX, DF)"
    assert len(set([f1, f2, f3, f4, f5, f6])) == 4


def test_perp():
    """Test the perp predicate."""
    f1 = Predicate("perp", [Angle("A", "B", "C")])
    f2 = Predicate("perp", [Angle("C", "B", "A")])
    f3 = Predicate("perp", ["C", "A", "D"])
    assert f1 == f2
    assert str(f1) == str(f2) == "perp (ABC)"
    assert str(f3) == "perp (CAD)"
    assert len(set([f1, f2, f3])) == 2


def test_eqangle():
    """Test the eqangle predicate."""
    f1 = Predicate("eqangle", ["A", "B", "C", "D", "E", "F"])
    f2 = Predicate("eqangle", [Angle(*"CBA"), Angle(*"FED")])
    f3 = Predicate("eqangle", ["C", "B", "A", "F", "E", "D"])
    f4 = Predicate("eqangle", ["U", "V", "W", "X", "Y", "Z"])
    assert f1 == f2 == f3
    assert str(f1) == str(f2) == str(f3) == "eqangle (ABC, DEF)"
    assert str(f4) == "eqangle (UVW, XYZ)"
    assert len(set([f1, f2, f3, f4])) == 2


def test_eqratio():
    """Test the eqratio predicate."""
    f1 = Predicate("eqratio", ["A", "B", "C", "D", "E", "F", "G", "H"])
    f2 = Predicate("eqratio", [
        Segment("A", "B"),
        Segment("C", "D"),
        Segment("F", "E"),
        Segment("H", "G")
    ])
    f3 = Predicate("eqratio", [
        Ratio(Segment("F", "E"), Segment("H", "G")),
        Ratio(Segment("A", "B"), Segment("C", "D"))
    ])
    f4 = Predicate("eqratio", ["U", "V", "W", "X", "Y", "Z", "A", "B"])
    f5 = Predicate("eqratio", [
        Segment("U", "V"),
        Segment("W", "X"),
        Segment("Z", "Y"),
        Segment("B", "A")
    ])
    f6 = Predicate("eqratio", [
        Ratio(Segment("Y", "Z"), Segment("B", "A")),
        Ratio(Segment("V", "U"), Segment("X", "W"))
    ])
    assert f1 == f2 == f3
    assert f4 == f5 == f6
    assert str(f1) == str(f2) == str(f3) == "eqratio (AB, CD, EF, GH)"
    assert str(f4) == str(f5) == str(f6) == "eqratio (AB, YZ, WX, UV)"
    assert len(set([f1, f2, f3, f4, f5, f6])) == 2


def test_simtri():
    """Test the simtri predicate."""
    f1 = Predicate("simtri", ["A", "B", "C", "D", "E", "F"])
    f2 = Predicate("simtri", ["B", "C", "A", "E", "F", "D"])
    f3 = Predicate("simtri",
                   [Triangle("D", "F", "E"),
                    Triangle("A", "C", "B")])
    f4 = Predicate("simtri", ["U", "V", "W", "X", "Y", "Z"])
    f5 = Predicate("simtri", ["W", "U", "V", "Z", "X", "Y"])
    f6 = Predicate("simtri",
                   [Triangle("Y", "X", "Z"),
                    Triangle("V", "U", "W")])
    assert str(f1) == str(f2) == str(f3) == "simtri (ABC, DEF)"
    assert str(f4) == str(f5) == str(f6) == "simtri (UVW, XYZ)"
    assert len(set([f1, f2, f3, f4])) == 2


def test_contri():
    """Test the contri predicate."""
    f1 = Predicate("contri", ["A", "B", "C", "D", "E", "F"])
    f2 = Predicate("contri", ["B", "C", "A", "E", "F", "D"])
    f3 = Predicate("contri",
                   [Triangle("D", "F", "E"),
                    Triangle("A", "C", "B")])
    f4 = Predicate("contri", ["U", "V", "W", "X", "Y", "Z"])
    f5 = Predicate("contri", ["W", "U", "V", "Z", "X", "Y"])
    f6 = Predicate("contri",
                   [Triangle("Y", "X", "Z"),
                    Triangle("V", "U", "W")])
    assert str(f1) == str(f2) == str(f3) == "contri (ABC, DEF)"
    assert str(f4) == str(f5) == str(f6) == "contri (UVW, XYZ)"
    assert len(set([f1, f2, f3, f4])) == 2


def test_sort():
    """Test predicate sorting."""
    f1 = Predicate("eqline", ["A", "B", "B", "C"])
    f2 = Predicate("eqline", ["X", "B", "B", "U"])
    f3 = Predicate("eqcircle", [[None, "X", "B", "U"], ["O", "C"]])
    f4 = Predicate("eqcircle", [["O", "A", "B", "C"], ["O", "A"]])
    f5 = Predicate("cong", [Segment("A", "B"), Segment("C", "D")])
    f6 = Predicate("cong", ["X", "B", "U", "D"])
    f7 = Predicate("midp", ["A", Segment("A", "B")])
    f8 = Predicate("midp", ["X", Segment("B", "U")])
    f9 = Predicate("para", ["C", "A", "B", "D"])
    f10 = Predicate("para", [Segment("D", "F"), Segment("C", "X")])
    f11 = Predicate("perp", ["C", "A", "B"])
    f12 = Predicate("perp", [Angle("D", "F", "E")])
    f13 = Predicate("eqangle", [Angle(*"CBA"), Angle(*"DEF")])
    f14 = Predicate("eqangle", ["U", "V", "W", "X", "Y", "Z"])
    f15 = Predicate("eqratio", [
        Ratio(Segment("F", "E"), Segment("H", "G")),
        Ratio(Segment("C", "D"), Segment("A", "B"))
    ])
    f16 = Predicate("eqratio", ["U", "V", "W", "X", "Y", "Z", "A", "B"])
    f17 = Predicate("simtri",
                    [Triangle("D", "F", "E"),
                     Triangle("A", "C", "B")])
    f18 = Predicate("simtri", ["U", "V", "W", "X", "Y", "Z"])
    f19 = Predicate("contri",
                    [Triangle("D", "F", "E"),
                     Triangle("A", "C", "B")])
    f20 = Predicate("contri", ["U", "V", "W", "X", "Y", "Z"])

    sorted_predicates = [
        f1, f2, f4, f3, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16,
        f17, f18, f19, f20
    ]
    shuffled_predicates = sorted_predicates[:]
    random.shuffle(shuffled_predicates)
    sorted_shuffled_predicates = sorted(shuffled_predicates)
    assert sorted_predicates == sorted_shuffled_predicates


def test_orderedfact():
    """Test the OrderedFact class."""
    f1 = Predicate("eqline", ["A", "B", "B", "C"])
    f2 = Predicate("eqline", ["X", "B", "B", "U"])
    f3 = Predicate("eqcircle", [[None, "X", "B", "U"], ["O", "C"]])
    f4 = Predicate("eqcircle", [["O", "A", "B", "C"], ["O", "A"]])
    f5 = Predicate("cong", [Segment("A", "B"), Segment("C", "D")])
    f6 = Predicate("cong", ["X", "B", "U", "D"])
    f7 = Predicate("midp", ["A", Segment("A", "B")])
    f8 = Predicate("midp", ["X", Segment("B", "U")])
    f9 = Predicate("para", ["C", "A", "B", "D"])
    f10 = Predicate("para", [Segment("D", "F"), Segment("C", "X")])
    f11 = Predicate("perp", ["C", "A", "B"])
    f12 = Predicate("perp", [Angle("D", "F", "E")])
    f13 = Predicate("eqangle", [Angle(*"CBA"), Angle(*"DEF")])
    f14 = Predicate("eqangle", ["U", "V", "W", "X", "Y", "Z"])
    f15 = Predicate("eqratio", [
        Ratio(Segment("F", "E"), Segment("H", "G")),
        Ratio(Segment("C", "D"), Segment("A", "B"))
    ])
    f16 = Predicate("eqratio", ["U", "V", "W", "X", "Y", "Z", "A", "B"])
    f17 = Predicate("simtri",
                    [Triangle("D", "F", "E"),
                     Triangle("A", "C", "B")])
    f18 = Predicate("simtri", ["U", "V", "W", "X", "Y", "Z"])
    f19 = Predicate("contri",
                    [Triangle("D", "F", "E"),
                     Triangle("A", "C", "B")])
    f20 = Predicate("contri", ["U", "V", "W", "X", "Y", "Z"])

    predicates = [
        f1, f2, f4, f3, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16,
        f17, f18, f19, f20
    ]
    types = [
        "eqline", "eqline", "eqcircle", "eqcircle", "cong", "cong", "midp",
        "midp", "para", "para", "perp", "perp", "eqangle", "eqangle",
        "eqratio", "eqratio", "simtri", "simtri", "contri", "contri"
    ]
    random.shuffle(predicates)
    of = OrderedFact()
    counter = 0
    for fact in predicates:
        counter += 1
        of.add(fact)
        assert of.num == counter

    while not of.is_empty():
        f, _ = of.first()
        t = types.pop(0)
        counter -= 1
        assert counter == of.num
        assert f.type == t
        assert f not in of
