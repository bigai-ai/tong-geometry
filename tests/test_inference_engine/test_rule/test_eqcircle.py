r"""Test the eqcircle module in forward_chainer."""

from itertools import combinations

from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import (Angle, Circle, Ratio,
                                                      Segment)
from tonggeometry.inference_engine.rule.eqcircle import (
    cong_and_ax_to_radax, cong_and_cong_and_ax_to_radax, cong_to_eqcircle,
    eqangle_and_eqcircle_to_eqangle_cen, eqangle_and_eqcircle_to_eqangle_cir,
    eqangle_and_eqcircle_to_eqangle_half, eqangle_and_eqcircle_to_perp,
    eqangle_and_radax_to_perp, eqangle_and_radax_to_radax, eqangle_to_eqcircle,
    eqcircle_and_eqangle_to_eqangle_cen, eqcircle_and_eqangle_to_eqangle_cir,
    eqcircle_and_eqangle_to_eqangle_half, eqcircle_and_eqangle_to_perp,
    eqcircle_and_eqcircle_to_radax, eqcircle_and_eqline_to_ax,
    eqcircle_and_eqline_to_perp, eqcircle_and_perp_to_ax,
    eqcircle_and_perp_to_eqangle, eqcircle_and_perp_to_eqline,
    eqcircle_to_cong, eqcircle_to_eqangle, eqcircle_to_eqcircle,
    eqline_and_eqcircle_to_ax, eqline_and_eqcircle_to_perp,
    eqline_and_radax_to_perp, eqline_and_radax_to_radax,
    eqratio_and_ax_to_radax, perp_and_eqcircle_to_ax,
    perp_and_eqcircle_to_eqangle, perp_and_eqcircle_to_eqline, radax)


def test_eqcircle_to_eqcircle():
    """Test eqcircle_to_eqcircle"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDEFGMXYZK", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("A", ["B"]), Circle("A", ["D"])]))
    f = Fact("eqcircle", [Circle("A", ["B"]), Circle("A", ["C"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_to_eqcircle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact(
        "eqcircle",
        [Circle("A", ["B"]), Circle(None, [*"BCD"])])
    f = Fact("eqcircle",
             [Circle(None, ["A", "B", "E"]),
              Circle(None, ["E", "F", "G"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_to_eqcircle(diagram, f)
    assert len(new_facts) == 8
    f = Fact("eqcircle", [Circle("M", ["A"]), Circle(None, ["A", "Y", "Z"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_to_eqcircle(diagram, f)
    assert len(new_facts) == 2
    f = Fact("eqcircle", [Circle(None, ["A", "Y", "Z"]), Circle("K", ["B"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_to_eqcircle(diagram, f)
    assert len(new_facts) == 6


def test_eqcircle_to_eqangle():
    """Test eqcircle_to_eqangle"""
    diagram = Diagram()
    A, B, C, D = "ABCD"
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    f = Fact("eqcircle", [Circle("X", ["A"]), Circle(None, ["B", "C", "D"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_to_eqangle(diagram, f)
    assert len(new_facts) == 6
    assert Fact("eqangle", [Angle(B, A, C), Angle(B, D, C)]) in new_facts
    assert Fact("eqangle", [Angle(A, B, D), Angle(A, C, D)]) in new_facts
    assert Fact("eqangle", [Angle(C, A, D), Angle(C, B, D)]) in new_facts
    assert Fact("eqangle", [Angle(A, C, B), Angle(A, D, B)]) in new_facts
    assert Fact("eqangle", [Angle(A, B, C), Angle(A, D, C)]) in new_facts
    assert Fact("eqangle", [Angle(B, A, D), Angle(B, C, D)]) in new_facts


def test_eqcircle_to_cong():
    """Test eqcircle_to_cong"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDE", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    f = Fact("eqcircle", [Circle("A", ["B"]), Circle(None, ["C", "D", "E"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_to_cong(diagram, f)
    assert len(new_facts) == 3
    assert Fact("cong", [Segment(*"AB"), Segment(*"AC")]) in new_facts
    assert Fact("cong", [Segment(*"AB"), Segment(*"AD")]) in new_facts
    assert Fact("cong", [Segment(*"AB"), Segment(*"AE")]) in new_facts


def test_cong_to_eqcircle():
    """Test cong_to_eqcircle"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDE", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    f = Fact("cong", [Segment("A", "B"), Segment("B", "C")])
    diagram.database.add_fact(f)
    new_facts = cong_to_eqcircle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact(
        "eqcircle",
        [Circle("B", ["A"]), Circle("B", ["C"])])


def test_eqangle_to_eqcircle():
    """Test eqangle_to_eqcircle"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(0, 0)
    diagram.point_dict['B'] = Point(1, 0)
    diagram.point_dict['C'] = Point(0, 1)
    for p1, p2 in combinations("ABCDE", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"ADC")])
    diagram.database.add_fact(f)
    new_facts = eqangle_to_eqcircle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact(
        "eqcircle",
        [Circle(None, ["A", "B", "C"]),
         Circle(None, ["A", "D", "C"])])


def test_eqline_and_eqcircle_to_perp():
    """Test eqline_and_eqcircle_to_perp"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDE", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("A", ["B"]), Circle("A", ["C"])]))
    diagram.database.add_fact(
        Fact("eqcircle", [Circle("A", ["B"]),
                          Circle(None, ["B", "C", "D"])]))
    diagram.database.add_fact(
        Fact("eqcircle", [Circle("A", ["B"]),
                          Circle(None, ["B", "C", "E"])]))
    f = Fact("eqline", [Segment(*"AB"), Segment(*"AC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqcircle_to_perp(diagram, f)
    assert len(new_facts) == 2
    assert Fact("perp", [Angle(*"BDC")]) in new_facts
    assert Fact("perp", [Angle(*"BEC")]) in new_facts


def test_eqcircle_and_eqline_to_perp():
    """Test eqcircle_and_eqline_to_perp"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDE", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"AC")]))
    f = Fact("eqcircle", [Circle("A", ["B"]), Circle(None, ["C", "D", "E"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_and_eqline_to_perp(diagram, f)
    assert len(new_facts) == 2
    assert Fact("perp", [Angle(*"BDC")]) in new_facts
    assert Fact("perp", [Angle(*"BEC")]) in new_facts


def test_eqcircle_and_eqangle_to_perp():
    """Test eqcircle_and_eqangle_to_perp"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABC"), Angle(*"ACD")]))
    f = Fact("eqcircle", [Circle("O", ["C"]), Circle(None, [*"ABC"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_and_eqangle_to_perp(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("perp", [Angle(*"OCD")])


def test_eqangle_and_eqcircle_to_perp():
    """Test eqangle_and_eqcircle_to_perp"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("O", ["A"]), Circle(None, [*"ABC"])]))
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"ACD")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqcircle_to_perp(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("perp", [Angle(*"OCD")])


def test_eqcircle_and_perp_to_eqangle():
    """Test eqcircle_and_perp_to_eqangle"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("perp", [Angle(*"OCD")]))
    f = Fact("eqcircle", [Circle("O", ["C"]), Circle(None, [*"ABC"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_and_perp_to_eqangle(diagram, f)
    assert len(new_facts) == 2
    assert Fact("eqangle", [Angle(*"ABC"), Angle(*"ACD")]) in new_facts
    assert Fact("eqangle", [Angle(*"BAC"), Angle(*"BCD")]) in new_facts


def test_perp_and_eqcircle_to_eqangle():
    """Test perp_and_eqcircle_to_eqangle"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("O", ["C"]), Circle(None, [*"ABC"])]))
    f = Fact("perp", [Angle(*"OCD")])
    diagram.database.add_fact(f)
    new_facts = perp_and_eqcircle_to_eqangle(diagram, f)
    assert len(new_facts) == 2
    assert Fact("eqangle", [Angle(*"ABC"), Angle(*"ACD")]) in new_facts
    assert Fact("eqangle", [Angle(*"BAC"), Angle(*"BCD")]) in new_facts


def test_eqcircle_and_perp_to_eqline():
    """Test eqcircle_and_perp_to_eqline"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("perp", [Angle(*"ABC")]))
    f = Fact("eqcircle", [Circle("D", ["A"]), Circle(None, [*"ABC"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_and_perp_to_eqline(diagram, f)
    assert len(new_facts) == 1
    assert Fact("eqline", [Segment(*"DA"), Segment(*"DC")]) in new_facts


def test_perp_and_eqcircle_to_eqline():
    """Test perp_and_eqcircle_to_eqline"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("D", ["A"]), Circle(None, [*"ABC"])]))
    f = Fact("perp", [Angle(*"ABC")])
    diagram.database.add_fact(f)
    new_facts = perp_and_eqcircle_to_eqline(diagram, f)
    assert len(new_facts) == 1
    assert Fact("eqline", [Segment(*"DA"), Segment(*"DC")]) in new_facts


def test_eqangle_and_eqcircle_to_eqangle_cen():
    """Test eqangle_and_eqcircle_to_eqangle_cen"""
    diagram = Diagram()
    diagram.point_dict['O'] = Point(0, 0)
    diagram.point_dict['A'] = Point(1, -1)
    diagram.point_dict['B'] = Point(-1, -1)
    diagram.point_dict['C'] = Point(1, 1)
    diagram.point_dict['M'] = Point(0, 0)
    diagram.point_dict['D'] = Point(1, -1)
    diagram.point_dict['E'] = Point(-1, -1)
    diagram.point_dict['F'] = Point(1, 1)
    for p1, p2 in combinations("ABCDEFGHMN", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("O", ["A"]), Circle(None, [*"ABC"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("O", ["A"]), Circle("O", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["D"]), Circle(None, [*"DEF"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["D"]), Circle("M", ["E"])]))
    f = Fact("eqangle", [Angle(*"AOB"), Angle(*"DME")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqcircle_to_eqangle_cen(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"ACB"), Angle(*"DFE")])


def test_eqcircle_and_eqangle_to_eqangle_cen():
    """Test eqcircle_and_eqangle_to_eqangle_cen"""
    diagram = Diagram()
    diagram.point_dict['O'] = Point(0, 0)
    diagram.point_dict['A'] = Point(1, -1)
    diagram.point_dict['B'] = Point(-1, -1)
    diagram.point_dict['C'] = Point(1, 1)
    diagram.point_dict['M'] = Point(0, 0)
    diagram.point_dict['D'] = Point(1, -1)
    diagram.point_dict['E'] = Point(-1, -1)
    diagram.point_dict['F'] = Point(1, 1)
    for p1, p2 in combinations("ABCDEFGHMN", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("O", ["A"]), Circle("O", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["D"]), Circle(None, [*"DEF"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["D"]), Circle("M", ["E"])]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"AOB"), Angle(*"DME")]))
    f = Fact("eqcircle", [Circle("O", ["A"]), Circle(None, [*"ABC"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_and_eqangle_to_eqangle_cen(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"ACB"), Angle(*"DFE")])


def test_eqangle_and_eqcircle_to_eqangle_cir():
    """Test eqangle_and_eqcircle_to_eqangle_cir"""
    diagram = Diagram()
    diagram.point_dict['O'] = Point(0, 0)
    diagram.point_dict['A'] = Point(1, -1)
    diagram.point_dict['B'] = Point(-1, -1)
    diagram.point_dict['C'] = Point(1, 1)
    diagram.point_dict['M'] = Point(0, 0)
    diagram.point_dict['D'] = Point(1, -1)
    diagram.point_dict['E'] = Point(-1, -1)
    diagram.point_dict['F'] = Point(1, 1)
    for p1, p2 in combinations("ABCDEFGHMN", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("O", ["A"]), Circle(None, [*"ABC"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("O", ["A"]), Circle("O", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["D"]), Circle(None, [*"DEF"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["D"]), Circle("M", ["E"])]))
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqcircle_to_eqangle_cir(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"AOC"), Angle(*"DMF")])


def test_eqcircle_and_eqangle_to_eqangle_cir():
    """Test eqcircle_and_eqangle_to_eqangle_cir"""
    diagram = Diagram()
    diagram.point_dict['O'] = Point(0, 0)
    diagram.point_dict['A'] = Point(1, -1)
    diagram.point_dict['B'] = Point(-1, -1)
    diagram.point_dict['C'] = Point(1, 1)
    diagram.point_dict['M'] = Point(0, 0)
    diagram.point_dict['D'] = Point(1, -1)
    diagram.point_dict['E'] = Point(-1, -1)
    diagram.point_dict['F'] = Point(1, 1)
    for p1, p2 in combinations("ABCDEFGHMN", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("O", ["A"]), Circle("O", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["D"]), Circle(None, [*"DEF"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["D"]), Circle("M", ["E"])]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")]))
    f = Fact("eqcircle", [Circle("O", ["A"]), Circle(None, [*"ABC"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_and_eqangle_to_eqangle_cir(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"AOC"), Angle(*"DMF")])


def test_eqangle_and_eqcircle_to_eqangle_half():
    """Test eqangle_and_eqcircle_to_eqangle_half"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-1, 0)
    diagram.point_dict['B'] = Point(0, 1)
    diagram.point_dict['C'] = Point(1, 0)
    diagram.point_dict['D'] = Point(0, -1)
    diagram.point_dict['M'] = Point(0, 0)
    for p1, p2 in combinations("ABCDM", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["A"]), Circle(None, [*"ABC"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["A"]), Circle("M", [*"C"])]))
    f = Fact("eqangle", [Angle(*"AMD"), Angle(*"DMC")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqcircle_to_eqangle_half(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"AMD"), Angle(*"ABC")])


def test_eqcircle_and_eqangle_to_eqangle_half():
    """Test eqcircle_and_eqangle_to_eqangle_half"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-1, 0)
    diagram.point_dict['B'] = Point(0, 1)
    diagram.point_dict['C'] = Point(1, 0)
    diagram.point_dict['D'] = Point(0, -1)
    diagram.point_dict['M'] = Point(0, 0)
    for p1, p2 in combinations("ABCDM", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("M", ["A"]), Circle("M", [*"C"])]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"AMD"), Angle(*"DMC")]))
    f = Fact("eqcircle", [Circle("M", ["A"]), Circle(None, [*"ABC"])])
    diagram.database.add_fact(f)
    new_facts = eqcircle_and_eqangle_to_eqangle_half(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"AMD"), Angle(*"ABC")])


def test_cong_and_ax_to_radax():
    """Test cong_and_ax_to_radax"""
    diagram = Diagram()
    diagram.point_dict["X"] = Point(0, 0)
    diagram.point_dict["Y"] = Point(0, 0)
    diagram.point_dict["D"] = Point(-1, 0)
    diagram.point_dict["E"] = Point(1, 0)
    diagram.point_dict["A"] = Point(0, 1)
    diagram.point_dict["C"] = Point(1, 1)
    diagram.point_dict["B"] = Point(2, 0)
    diagram.database.points_axes["A"] = {}
    diagram.database.points_axes["A"][Segment(*"AX")] = (False, ["D"])
    diagram.database.points_axes["A"][Segment(*"AY")] = (False, ["E"])
    for p1, p2 in combinations("ABCDEXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("D", ["X"]), Circle("D", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("E", ["Y"]), Circle("E", ["C"])]))
    f = Fact("cong", [Segment(*"XA"), Segment(*"YA")])
    diagram.database.add_fact(f)
    cong_and_ax_to_radax(diagram, f)
    assert "A" in diagram.database.points_radaxes
    assert Circle("D", ["B"]) in diagram.database.points_radaxes["A"]
    assert Circle("E", ["C"]) in diagram.database.points_radaxes["A"]
    assert Circle("D", ["B"]) in diagram.database.radaxes
    assert Circle("E", ["C"]) in diagram.database.radaxes
    assert "A" in diagram.database.radaxes[Circle("D",
                                                  ["B"])][Circle("E", ["C"])]


def test_perp_and_eqcircle_to_ax():
    """Test perp_and_eqcircle_to_ax"""
    diagram = Diagram()
    diagram.point_dict["X"] = Point(0, 0)
    diagram.point_dict["Y"] = Point(0, 0)
    diagram.point_dict["D"] = Point(-1, 0)
    diagram.point_dict["E"] = Point(1, 0)
    diagram.point_dict["A"] = Point(0, 1)
    diagram.point_dict["C"] = Point(1, 1)
    diagram.point_dict["B"] = Point(2, 0)
    for p1, p2 in combinations("ABCDEXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("D", ["X"]), Circle("D", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("E", ["Y"]), Circle("E", ["C"])]))
    diagram.database.add_fact(Fact("cong", [Segment(*"XA"), Segment(*"YA")]))
    f = Fact("perp", [Angle(*"AXD")])
    diagram.database.add_fact(f)
    perp_and_eqcircle_to_ax(diagram, f)
    assert "A" in diagram.database.points_axes
    assert diagram.database.points_axes["A"][Segment(*"AX")][1][0] == "D"
    f = Fact("perp", [Angle(*"AYE")])
    diagram.database.add_fact(f)
    perp_and_eqcircle_to_ax(diagram, f)
    assert diagram.database.points_axes["A"][Segment(*"AY")][1][0] == "E"
    assert len(diagram.database.axes) == 2
    assert len(diagram.database.points_axes) == 1
    assert len(diagram.database.points_axes["A"]) == 2
    assert "A" in diagram.database.points_radaxes
    assert Circle("D", ["B"]) in diagram.database.points_radaxes["A"]
    assert Circle("E", ["C"]) in diagram.database.points_radaxes["A"]
    assert Circle("D", ["B"]) in diagram.database.radaxes
    assert Circle("E", ["C"]) in diagram.database.radaxes
    assert "A" in diagram.database.radaxes[Circle("D",
                                                  ["B"])][Circle("E", ["C"])]


def test_eqcircle_and_perp_to_ax():
    """Test eqcircle_and_perp_to_ax"""
    diagram = Diagram()
    diagram.point_dict["X"] = Point(0, 0)
    diagram.point_dict["Y"] = Point(0, 0)
    diagram.point_dict["D"] = Point(-1, 0)
    diagram.point_dict["E"] = Point(1, 0)
    diagram.point_dict["A"] = Point(0, 1)
    diagram.point_dict["C"] = Point(1, 1)
    diagram.point_dict["B"] = Point(2, 0)
    for p1, p2 in combinations("ABCDEXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("perp", [Angle(*"AXD")]))
    diagram.database.add_fact(Fact("perp", [Angle(*"AYE")]))
    diagram.database.add_fact(Fact("cong", [Segment(*"XA"), Segment(*"YA")]))
    f = Fact("eqcircle", [Circle("D", ["X"]), Circle("D", ["B"])])
    diagram.database.add_fact(f)
    eqcircle_and_perp_to_ax(diagram, f)
    assert "A" in diagram.database.points_axes
    assert diagram.database.points_axes["A"][Segment(*"AX")][1][0] == "D"
    f = Fact("eqcircle", [Circle("E", ["Y"]), Circle("E", ["C"])])
    diagram.database.add_fact(f)
    eqcircle_and_perp_to_ax(diagram, f)
    assert diagram.database.points_axes["A"][Segment(*"AY")][1][0] == "E"
    assert len(diagram.database.axes) == 2
    assert len(diagram.database.points_axes) == 1
    assert len(diagram.database.points_axes["A"]) == 2
    assert "A" in diagram.database.points_radaxes
    assert Circle("D", ["B"]) in diagram.database.points_radaxes["A"]
    assert Circle("E", ["C"]) in diagram.database.points_radaxes["A"]
    assert Circle("D", ["B"]) in diagram.database.radaxes
    assert Circle("E", ["C"]) in diagram.database.radaxes
    assert "A" in diagram.database.radaxes[Circle("D",
                                                  ["B"])][Circle("E", ["C"])]


def test_eqratio_and_ax_to_radax():
    """Test eqratio_and_ax_to_radax"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(0, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 0)
    diagram.point_dict["F"] = Point(1, 0)
    diagram.point_dict["P"] = Point(0, 1)
    diagram.database.points_axes["P"] = {}
    diagram.database.points_axes["P"][Segment(*"AB")] = (False, ["E"])
    diagram.database.points_axes["P"][Segment(*"CD")] = (False, ["F"])
    for p1, p2 in combinations("ABCDEFP", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("E", ["A"]), Circle("E", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("F", ["C"]), Circle("F", ["D"])]))
    f = Fact("eqratio", [
        Ratio(Segment(*"PA"), Segment(*"PC")),
        Ratio(Segment(*"PD"), Segment(*"PB"))
    ])
    diagram.database.add_fact(f)
    eqratio_and_ax_to_radax(diagram, f)
    assert "P" in diagram.database.points_radaxes
    assert Circle("E", ["A"]) in diagram.database.points_radaxes["P"]
    assert Circle("F", ["C"]) in diagram.database.points_radaxes["P"]
    assert Circle("E", ["A"]) in diagram.database.radaxes
    assert Circle("F", ["C"]) in diagram.database.radaxes
    assert "P" in diagram.database.radaxes[Circle("E",
                                                  ["A"])][Circle("F", ["C"])]
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 0)
    diagram.point_dict["F"] = Point(1, 0)
    diagram.point_dict["P"] = Point(0, 1)
    diagram.database.points_axes["P"] = {}
    diagram.database.points_axes["P"][Segment(*"AB")] = (False, ["E"])
    diagram.database.points_axes["P"][Segment(*"PC")] = (False, ["F"])
    for p1, p2 in combinations("ABCDEFP", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("E", ["A"]), Circle("E", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("F", ["C"]), Circle("F", ["D"])]))
    f = Fact("eqratio", [
        Ratio(Segment(*"PC"), Segment(*"PA")),
        Ratio(Segment(*"PB"), Segment(*"PC"))
    ])
    diagram.database.add_fact(f)
    eqratio_and_ax_to_radax(diagram, f)
    assert "P" in diagram.database.points_radaxes
    assert Circle("E", ["A"]) in diagram.database.points_radaxes["P"]
    assert Circle("F", ["C"]) in diagram.database.points_radaxes["P"]
    assert Circle("E", ["A"]) in diagram.database.radaxes
    assert Circle("F", ["C"]) in diagram.database.radaxes
    assert "P" in diagram.database.radaxes[Circle("E",
                                                  ["A"])][Circle("F", ["C"])]


def test_cong_and_cong_and_ax_to_radax():
    """Test cong_and_cong_and_ax_to_radax"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 0)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 0)
    diagram.point_dict["F"] = Point(1, 0)
    diagram.point_dict["P"] = Point(0, 1)
    diagram.database.points_axes["P"] = {}
    diagram.database.points_axes["P"][Segment(*"AB")] = (False, ["E"])
    diagram.database.points_axes["P"][Segment(*"CD")] = (False, ["F"])
    for p1, p2 in combinations("ABCDEFP", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("E", ["A"]), Circle("E", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("F", ["C"]), Circle("F", ["D"])]))
    diagram.database.add_fact(Fact("cong", [Segment(*"PB"), Segment(*"PD")]))
    f = Fact("cong", [Segment(*"PA"), Segment(*"PC")])
    diagram.database.add_fact(f)
    cong_and_cong_and_ax_to_radax(diagram, f)
    assert "P" in diagram.database.points_radaxes
    assert Circle("E", ["A"]) in diagram.database.points_radaxes["P"]
    assert Circle("F", ["C"]) in diagram.database.points_radaxes["P"]
    assert Circle("E", ["A"]) in diagram.database.radaxes
    assert Circle("F", ["C"]) in diagram.database.radaxes
    assert "P" in diagram.database.radaxes[Circle("E",
                                                  ["A"])][Circle("F", ["C"])]


def test_eqline_and_eqcircle_to_ax():
    """Test eqline_and_eqcircle_to_ax"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(0, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 0)
    diagram.point_dict["F"] = Point(1, 0)
    diagram.point_dict["P"] = Point(0, 1)
    diagram.database.points_axes["P"] = {}
    diagram.database.points_axes["P"][Segment(*"CD")] = (False, ["F"])
    for p1, p2 in combinations("ABCDEFP", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("E", ["A"]), Circle("E", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("F", ["C"]), Circle("F", ["D"])]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment(*"PA"), Segment(*"PC")),
            Ratio(Segment(*"PD"), Segment(*"PB"))
        ]))
    f = Fact("eqline", [Segment(*"PA"), Segment(*"PB")])
    diagram.database.add_fact(f)
    eqline_and_eqcircle_to_ax(diagram, f)
    assert "P" in diagram.database.points_radaxes
    assert Circle("E", ["A"]) in diagram.database.points_radaxes["P"]
    assert Circle("F", ["C"]) in diagram.database.points_radaxes["P"]
    assert Circle("E", ["A"]) in diagram.database.radaxes
    assert Circle("F", ["C"]) in diagram.database.radaxes
    assert "P" in diagram.database.radaxes[Circle("E",
                                                  ["A"])][Circle("F", ["C"])]


def test_eqcircle_and_eqline_to_ax():
    """Test eqcircle_and_eqline_to_ax"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(0, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 0)
    diagram.point_dict["F"] = Point(1, 0)
    diagram.point_dict["P"] = Point(0, 1)
    diagram.database.points_axes["P"] = {}
    diagram.database.points_axes["P"][Segment(*"PC")] = (False, ["F"])
    for p1, p2 in combinations("ABCDEFP", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"PA"), Segment(*"PB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"PA"), Segment(*"AB")]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("F", ["C"]), Circle("F", ["D"])]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment(*"PC"), Segment(*"PA")),
            Ratio(Segment(*"PB"), Segment(*"PC"))
        ]))
    f = Fact("eqcircle", [Circle("E", ["A"]), Circle("E", ["B"])])
    diagram.database.add_fact(f)
    eqcircle_and_eqline_to_ax(diagram, f)
    assert "P" in diagram.database.points_radaxes
    assert Circle("E", ["A"]) in diagram.database.points_radaxes["P"]
    assert Circle("F", ["C"]) in diagram.database.points_radaxes["P"]
    assert Circle("E", ["A"]) in diagram.database.radaxes
    assert Circle("F", ["C"]) in diagram.database.radaxes
    assert "P" in diagram.database.radaxes[Circle("E",
                                                  ["A"])][Circle("F", ["C"])]


def test_eqcircle_and_eqcircle_to_radax():
    """Test eqcircle_and_eqcircle_to_radax"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 0)
    diagram.point_dict["F"] = Point(1, 0)
    for p1, p2 in combinations("ABCEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("F", ["C"]), Circle("F", ["B"])]))
    f = Fact("eqcircle", [Circle("E", ["A"]), Circle("E", ["B"])])
    diagram.database.add_fact(f)
    eqcircle_and_eqcircle_to_radax(diagram, f)
    assert "B" in diagram.database.points_radaxes
    assert Circle("E", ["A"]) in diagram.database.points_radaxes["B"]
    assert Circle("F", ["B"]) in diagram.database.points_radaxes["B"]
    assert Circle("E", ["A"]) in diagram.database.radaxes
    assert Circle("F", ["B"]) in diagram.database.radaxes
    assert "B" in diagram.database.radaxes[Circle("E",
                                                  ["A"])][Circle("F", ["B"])]


def test_eqline_and_radax_to_radax():
    """Test eqline_and_radax_to_radax"""
    diagram = Diagram()
    diagram.point_dict["D"] = Point(-1, 0)
    diagram.point_dict["E"] = Point(1, 0)
    diagram.point_dict["P"] = Point(0, 1)
    diagram.point_dict["C"] = Point(1, 1)
    diagram.point_dict["B"] = Point(2, 0)
    diagram.point_dict["X"] = Point(0, 0)
    diagram.point_dict["Y"] = Point(0, 0)
    diagram.database.points_radaxes["A"] = {
        Circle("D", ["X"]): {
            Circle("E", ["Y"]): None
        },
        Circle("E", ["Y"]): {
            Circle("D", ["X"]): None
        }
    }
    diagram.database.points_radaxes["B"] = {
        Circle("D", ["X"]): {
            Circle("E", ["Y"]): None
        },
        Circle("E", ["Y"]): {
            Circle("D", ["X"]): None
        }
    }
    diagram.database.radaxes[Circle("E", ["Y"])] = {
        Circle("D", ["X"]): {
            "A": ([], Circle("E", ["Y"]), Circle("D", ["X"])),
            "B": ([], Circle("E", ["Y"]), Circle("D", ["X"]))
        }
    }
    diagram.database.radaxes[Circle("D", ["X"])] = {
        Circle("E", ["Y"]): {
            "A": ([], Circle("D", ["X"]), Circle("E", ["Y"])),
            "B": ([], Circle("D", ["X"]), Circle("E", ["Y"]))
        }
    }
    for p1, p2 in combinations("ABCDEPXYZ", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("D", ["X"]), Circle("D", ["Z"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("E", ["Y"]), Circle("E", ["Z"])]))
    f = Fact("eqline", [Segment(*"PA"), Segment(*"PB")])
    diagram.database.add_fact(f)
    eqline_and_radax_to_radax(diagram, f)
    assert "P" in diagram.database.points_radaxes
    assert Circle("D", ["X"]) in diagram.database.points_radaxes["P"]
    assert Circle("E", ["Y"]) in diagram.database.points_radaxes["P"]
    assert "P" in diagram.database.radaxes[Circle("D",
                                                  ["X"])][Circle("E", ["Y"])]


def test_eqline_and_radax_to_perp():
    """Test eqline_and_radax_to_perp"""
    diagram = Diagram()
    diagram.point_dict["D"] = Point(-1, 0)
    diagram.point_dict["E"] = Point(1, 0)
    diagram.point_dict["P"] = Point(0, 1)
    diagram.point_dict["C"] = Point(1, 1)
    diagram.point_dict["B"] = Point(2, 0)
    diagram.point_dict["X"] = Point(0, 0)
    diagram.point_dict["Y"] = Point(0, 0)
    diagram.database.points_radaxes["A"] = {
        Circle("D", ["X"]): {
            Circle("E", ["Y"]): None
        },
        Circle("E", ["Y"]): {
            Circle("D", ["X"]): None
        }
    }
    diagram.database.points_radaxes["B"] = {
        Circle("D", ["X"]): {
            Circle("E", ["Y"]): None
        },
        Circle("E", ["Y"]): {
            Circle("D", ["X"]): None
        }
    }
    diagram.database.radaxes[Circle("E", ["Y"])] = {
        Circle("D", ["X"]): {
            "A": ([], Circle("E", ["Y"]), Circle("D", ["X"])),
            "B": ([], Circle("E", ["Y"]), Circle("D", ["X"]))
        }
    }
    diagram.database.radaxes[Circle("D", ["X"])] = {
        Circle("E", ["Y"]): {
            "A": ([], Circle("D", ["X"]), Circle("E", ["Y"])),
            "B": ([], Circle("D", ["X"]), Circle("E", ["Y"]))
        }
    }
    for p1, p2 in combinations("ABCDEPXYZ", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("D", ["X"]), Circle("D", ["Z"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("E", ["Y"]), Circle("E", ["Z"])]))
    f = Fact("eqline", [Segment(*"DB"), Segment(*"EB")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_radax_to_perp(diagram, f)
    assert len(new_facts) == 1
    assert Fact("perp", [Angle(*"DBA")]) in new_facts


def test_eqangle_and_radax_to_radax():
    """Test eqangle_and_radax_to_radax"""
    diagram = Diagram()
    diagram.point_dict["D"] = Point(-1, 0)
    diagram.point_dict["E"] = Point(1, 0)
    diagram.point_dict["P"] = Point(0, 1)
    diagram.point_dict["C"] = Point(1, 1)
    diagram.point_dict["B"] = Point(2, 0)
    diagram.database.points_radaxes["A"] = {
        Circle("D", ["B"]): {
            Circle("E", ["C"]): None
        },
        Circle("E", ["C"]): {
            Circle("D", ["B"]): None
        }
    }
    diagram.database.radaxes[Circle("E", ["C"])] = {
        Circle("D", ["B"]): {
            "A": ([], Circle("E", ["C"]), Circle("D", ["B"]))
        }
    }
    diagram.database.radaxes[Circle("D", ["B"])] = {
        Circle("E", ["C"]): {
            "A": ([], Circle("D", ["B"]), Circle("E", ["C"]))
        }
    }
    for p1, p2 in combinations("ABCDEPXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("D", ["X"]), Circle("D", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("E", ["Y"]), Circle("E", ["C"])]))
    diagram.database.add_fact(Fact("perp", [Angle(*"DAP")]))
    f = Fact("eqangle", [Angle(*"DAP"), Angle(*"EAP")])
    diagram.database.add_fact(f)
    eqangle_and_radax_to_radax(diagram, f)
    assert "P" in diagram.database.points_radaxes
    assert Circle("D", ["B"]) in diagram.database.points_radaxes["P"]
    assert Circle("E", ["C"]) in diagram.database.points_radaxes["P"]
    assert "P" in diagram.database.radaxes[Circle("D",
                                                  ["B"])][Circle("E", ["C"])]


def test_eqangle_and_radax_to_perp():
    """Test eqangle_and_radax_to_perp"""
    diagram = Diagram()
    diagram.database.points_radaxes["A"] = {
        Circle("D", ["B"]): {
            Circle("E", ["C"]): None
        },
        Circle("E", ["C"]): {
            Circle("D", ["B"]): None
        }
    }
    diagram.database.points_radaxes["B"] = {
        Circle("D", ["B"]): {
            Circle("E", ["C"]): None
        },
        Circle("E", ["C"]): {
            Circle("D", ["B"]): None
        }
    }
    diagram.database.radaxes[Circle("E", ["C"])] = {
        Circle("D", ["B"]): {
            "A": ([], Circle("E", ["C"]), Circle("D", ["B"])),
            "B": ([], Circle("E", ["C"]), Circle("D", ["B"]))
        }
    }
    diagram.database.radaxes[Circle("D", ["B"])] = {
        Circle("E", ["C"]): {
            "A": ([], Circle("D", ["B"]), Circle("E", ["C"])),
            "B": ([], Circle("E", ["C"]), Circle("D", ["B"]))
        }
    }
    for p1, p2 in combinations("ABCDEPXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("perp", [Angle(*"DHA")]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("D", ["X"]), Circle("D", ["B"])]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("E", ["Y"]), Circle("E", ["C"])]))
    f = Fact("eqangle", [Angle(*"DHA"), Angle(*"BHD")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_radax_to_perp(diagram, f)
    assert len(new_facts) == 2
    assert Fact("perp", [Angle(*"EHA")]) in new_facts
    assert Fact("perp", [Angle(*"EHB")]) in new_facts


def test_radax():
    """Test radax"""

    def init():
        diagram = Diagram()
        diagram.point_dict["D"] = Point(-1, 0)
        diagram.point_dict["E"] = Point(1, 0)
        diagram.point_dict["A"] = Point(0, 0)
        diagram.point_dict["C"] = Point(10, 1)
        diagram.point_dict["G"] = Point(100, 1)
        diagram.point_dict["C"] = Point(1, 1)
        diagram.point_dict["B"] = Point(2, 0)
        diagram.database.points_radaxes["A"] = {
            Circle("D", ["B"]): {
                Circle("E", ["C"]): None
            },
            Circle("E", ["C"]): {
                Circle("D", ["B"]): None
            }
        }
        diagram.database.radaxes[Circle("E", ["C"])] = {
            Circle("D", ["B"]): {
                "A": ([], Circle("E", ["Y"]), Circle("D", ["X"])),
                "Z": ([], Circle("E", ["Y"]), Circle("D", ["X"])),
            }
        }
        diagram.database.radaxes[Circle("D", ["B"])] = {
            Circle("E", ["C"]): {
                "A": ([], Circle("D", ["X"]), Circle("E", ["Y"])),
                "Z": ([], Circle("E", ["Y"]), Circle("D", ["X"])),
            }
        }
        for p1, p2 in combinations("ABCDEFQXYZ", 2):
            diagram.database.add_fact(
                Fact("eqline",
                     [Segment(p1, p2), Segment(p1, p2)]))
        diagram.database.add_fact(
            Fact("eqcircle",
                 [Circle("D", ["X"]), Circle("D", ["B"])]))
        diagram.database.add_fact(
            Fact("eqcircle",
                 [Circle("E", ["Y"]), Circle("E", ["C"])]))
        return diagram

    c1 = Circle("D", ["B"])
    c2 = Circle("E", ["C"])

    diagram = init()
    c1_ax1 = Segment("A", "H")
    c1_ax2 = Segment("J", "K")
    c2_ax1 = Segment("A", "I")
    c2_ax2 = Segment("L", "M")
    p = "A"
    diagram.database.axes[c1] = {(c1_ax1, p): None, (c1_ax2, p): None}
    diagram.database.axes[c2] = {(c2_ax1, p): None, (c2_ax2, p): None}
    diagram.database.points_axes[p] = {
        c1_ax1: (False, []),
        c2_ax1: (False, []),
        c1_ax2: (False, []),
        c2_ax2: (False, [])
    }
    new_facts = radax(diagram, c1, c2, "A")
    assert len(new_facts) == 4
    assert Fact("cong", [c1_ax1, c2_ax1]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"AH"), Segment(*"AL")),
        Ratio(Segment(*"AM"), Segment(*"AH"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"AJ"), Segment(*"AI")),
        Ratio(Segment(*"AI"), Segment(*"AK"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"AJ"), Segment(*"AL")),
        Ratio(Segment(*"AM"), Segment(*"AK"))
    ]) in new_facts

    diagram = init()
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("D", ["A"]), Circle("D", ["X"])]))
    new_facts = radax(diagram, c1, c2, "A")
    assert len(new_facts) == 1
    assert new_facts[0] == Fact(
        "eqcircle",
        [Circle("E", ["Y"]), Circle("E", ["A"])])

    diagram = init()
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "A"), Segment("E", "A")]))
    new_facts = radax(diagram, c1, c2, "A")
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("perp", [Angle(*"ZAD")])

    diagram = init()
    diagram.database.add_fact(Fact("perp", [Angle(*"DAZ")]))
    new_facts = radax(diagram, c1, c2, "A")
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqline", [Segment(*"DA"), Segment(*"EA")])

    diagram = init()
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "Z"), Segment("E", "Z")]))
    new_facts = radax(diagram, c1, c2, "A")
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("perp", [Angle(*"AZD")])

    diagram = init()
    diagram.database.add_fact(Fact("perp", [Angle("D", "Z", "A")]))
    new_facts = radax(diagram, c1, c2, "A")
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqline", [Segment(*"DZ"), Segment(*"EZ")])

    diagram = init()
    diagram.database.radaxes[Circle("E", ["C"])][Circle(
        "D", ["B"])]["W"] = ([], Circle("E", ["Y"]), Circle("D", ["X"]))
    diagram.database.radaxes[Circle("D", ["B"])][Circle(
        "E", ["C"])]["W"] = ([], Circle("D", ["X"]), Circle("E", ["Y"]))
    new_facts = radax(diagram, c1, c2, "A")
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqline", [Segment(*"AZ"), Segment(*"AW")])

    diagram = init()
    diagram.point_dict["A"] = Point(0, 1)
    diagram.point_dict["Q"] = Point(0, 0)
    diagram.database.add_fact(Fact("perp", [Angle(*"DQA")]))
    diagram.database.add_fact(Fact("perp", [Angle(*"EQA")]))
    new_facts = radax(diagram, c1, c2, "A")
    assert len(new_facts) == 2
    assert Fact("eqline", [Segment(*"AQ"), Segment(*"ZQ")]) in new_facts
    assert Fact("eqline", [Segment(*"DQ"), Segment(*"EQ")]) in new_facts

    diagram = init()
    diagram.point_dict["Q"] = Point(1, 0)
    diagram.database.add_fact(Fact("perp", [Angle(*"DAQ")]))
    diagram.database.add_fact(Fact("perp", [Angle(*"EAQ")]))
    new_facts = radax(diagram, c1, c2, "A")
    assert len(new_facts) == 2
    assert Fact("eqline", [Segment(*"AQ"), Segment(*"ZQ")]) in new_facts
    assert Fact("eqline", [Segment(*"AE"), Segment(*"AD")]) in new_facts

    diagram = init()
    diagram.point_dict["F"] = Point(-2, 0)
    diagram.database.points_radaxes["A"] = {
        Circle("D", ["B"]): {
            Circle("E", ["C"]): None,
            Circle("F", ["G"]): None
        },
        Circle("E", ["C"]): {
            Circle("D", ["B"]): None
        },
        Circle("F", ["G"]): {
            Circle("D", ["B"]): None
        }
    }
    diagram.database.radaxes[Circle("D", ["B"])] = {
        Circle("E", ["C"]): {
            "A": ([], Circle("D", ["X"]), Circle("E", ["Y"])),
            "Z": ([], Circle("E", ["Y"]), Circle("D", ["X"])),
        },
        Circle("F", ["G"]): {
            "A": ([], Circle("D", ["B"]), Circle("F", ["G"]))
        }
    }
    diagram.database.radaxes[Circle("F", ["G"])] = {
        Circle("D", ["B"]): {
            "A": ([], Circle("F", ["G"]), Circle("D", ["B"]))
        }
    }
    for p1, p2 in combinations("ABCDEFGHPXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqcircle",
             [Circle("F", ["G"]), Circle("F", ["H"])]))
    new_facts = radax(diagram, c1, c2, "A")
    assert len(new_facts) == 0
    assert Circle("F", ["G"]) in diagram.database.points_radaxes["A"][Circle(
        "E", ["C"])]
    assert Circle("E", ["C"]) in diagram.database.points_radaxes["A"][Circle(
        "F", ["G"])]
