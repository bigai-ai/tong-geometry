r"""Test the contri module in forward_chainer."""

from itertools import combinations

from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Segment, Triangle
from tonggeometry.inference_engine.rule.contri import (
    cong_and_cong_and_cong_to_contri, cong_and_cong_and_eqangle_to_contri,
    contri_to_cong, contri_to_eqangle, eqangle_and_cong_and_cong_to_contri)


def test_cong_and_cong_and_cong_to_contri():
    """Test cong_and_cong_to_contri"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    diagram.database.add_fact(
        Fact("cong", [Segment("A", "B"), Segment("D", "E")]))
    diagram.database.add_fact(
        Fact("cong", [Segment("B", "C"), Segment("E", "F")]))
    f = Fact("cong", [Segment("A", "C"), Segment("D", "F")])
    diagram.database.add_fact(f)
    new_facts = cong_and_cong_and_cong_to_contri(diagram, f)
    assert len(new_facts) == 2
    assert new_facts[0] == new_facts[1]
    assert new_facts[0] == Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_cong_and_cong_and_eqangle_to_contri_SAS():
    """Test cong_and_cong_and_eqangle_to_contri"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABC"), Angle(*"FED")]))
    diagram.database.add_fact(
        Fact("cong", [Segment("A", "B"), Segment("D", "E")]))
    f = Fact("cong", [Segment("B", "C"), Segment("E", "F")])
    diagram.database.add_fact(f)
    new_facts = cong_and_cong_and_eqangle_to_contri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_cong_and_cong_and_eqangle_to_contri_SSA():
    """Test cong_and_cong_and_eqangle_to_contri"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("perp", [Angle(*"ACB")]))
    diagram.database.add_fact(Fact("perp", [Angle(*"DFE")]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ACB"), Angle(*"EFD")]))
    diagram.database.add_fact(
        Fact("cong", [Segment("A", "B"), Segment("D", "E")]))
    f = Fact("cong", [Segment("A", "C"), Segment("D", "F")])
    diagram.database.add_fact(f)
    new_facts = cong_and_cong_and_eqangle_to_contri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_cong_and_cong_and_eqangle_to_contri_SSA_h():
    """Test cong_and_cong_and_eqangle_to_contri"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("perp", [Angle(*"ACB")]))
    diagram.database.add_fact(Fact("perp", [Angle(*"EFD")]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ACB"), Angle(*"EFD")]))
    diagram.database.add_fact(
        Fact("cong", [Segment("A", "C"), Segment("D", "F")]))
    f = Fact("cong", [Segment("A", "B"), Segment("D", "E")])
    diagram.database.add_fact(f)
    new_facts = cong_and_cong_and_eqangle_to_contri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_eqangle_and_cong_and_cong_to_contri_SAS():
    """Test eqangle_and_cong_and_cong_to_contri"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("cong", [Segment("A", "B"), Segment("D", "E")]))
    diagram.database.add_fact(
        Fact("cong", [Segment("B", "C"), Segment("E", "F")]))
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"FED")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_cong_and_cong_to_contri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_eqangle_and_cong_and_cong_to_contri_SSA():
    """Test eqangle_and_cong_and_cong_to_contri"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("perp", [Angle("A", "C", "B")]))
    diagram.database.add_fact(Fact("perp", [Angle("E", "F", "D")]))
    diagram.database.add_fact(
        Fact("cong", [Segment("A", "C"), Segment("D", "F")]))
    diagram.database.add_fact(
        Fact("cong", [Segment("A", "B"), Segment("E", "D")]))
    f = Fact("eqangle", [Angle(*"ACB"), Angle(*"EFD")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_cong_and_cong_to_contri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_contri_to_cong():
    """Test contri_to_cong"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    f = Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])
    diagram.database.add_fact(f)
    new_facts = contri_to_cong(diagram, f)
    assert len(new_facts) == 3
    assert Fact("cong", [Segment("A", "B"), Segment("D", "E")]) in new_facts
    assert Fact("cong", [Segment("B", "C"), Segment("E", "F")]) in new_facts
    assert Fact("cong", [Segment("A", "C"), Segment("D", "F")]) in new_facts


def test_contri_to_eqangle():
    """Test contri_to_eqangle"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    f = Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])
    diagram.database.add_fact(f)
    new_facts = contri_to_eqangle(diagram, f)
    assert len(new_facts) == 3
    assert Fact("eqangle", [Angle(*"ABC"), Angle(*"FED")]) in new_facts
    assert Fact("eqangle", [Angle(*"BCA"), Angle(*"DFE")]) in new_facts
    assert Fact("eqangle", [Angle(*"CAB"), Angle(*"EDF")]) in new_facts
