r"""Test the perp module in forward_chainer."""

from itertools import combinations

from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Segment
from tonggeometry.inference_engine.rule.perp import (
    cong_and_eqline_and_perp_to_cong, eqline_and_cong_and_perp_to_cong,
    eqline_and_perp_to_perp, midp_and_perp_to_cong,
    perp_and_eqline_and_cong_to_cong, perp_and_eqline_to_perp,
    perp_and_midp_to_cong, perp_and_perp_to_eqangle)


def test_perp_and_perp_to_eqangle():
    """Test perp_and_perp_to_eqangle"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("perp", [Angle(*"ABC")]))
    f = Fact("perp", [Angle(*"ADB")])
    diagram.database.add_fact(f)
    new_facts = perp_and_perp_to_eqangle(diagram, f)
    assert len(new_facts) == 3
    assert Fact("eqangle", [Angle(*"ABC"), Angle(*"ADB")]) in new_facts
    assert Fact("eqangle", [Angle(*"ABC"), Angle(*"BDA")]) in new_facts
    assert Fact("eqangle", [Angle(*"BAD"), Angle(*"CBD")]) in new_facts
    diagram = Diagram()
    diagram.database.add_fact(Fact("perp", [Angle(*"ABC")]))
    f = Fact("perp", [Angle(*"ACD")])
    diagram.database.add_fact(f)
    new_facts = perp_and_perp_to_eqangle(diagram, f)
    assert len(new_facts) == 3
    assert Fact("eqangle", [Angle(*"ABC"), Angle(*"ACD")]) in new_facts
    assert Fact("eqangle", [Angle(*"ABC"), Angle(*"DCA")]) in new_facts
    assert Fact("eqangle", [Angle(*"BAC"), Angle(*"BCD")]) in new_facts


def test_eqline_and_perp_to_perp():
    """Test eqline_and_perp_to_perp"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"AB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BC"), Segment(*"BC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BD"), Segment(*"BD")]))
    diagram.database.add_fact(Fact("perp", [Angle(*"ABC")]))
    f = Fact("eqline", [Segment(*"BC"), Segment(*"BD")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_perp_to_perp(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("perp", [Angle(*"ABD")])


def test_perp_and_eqline_to_perp():
    """Test perp_and_eqline_to_perp"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"AB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BC"), Segment(*"BC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BD"), Segment(*"BD")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BE"), Segment(*"BE")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"BC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BD"), Segment(*"BE")]))
    f = Fact("perp", [Angle(*"ABD")])
    diagram.database.add_fact(f)
    new_facts = perp_and_eqline_to_perp(diagram, f)
    assert len(new_facts) == 2
    assert Fact("perp", [Angle(*"ABE")]) in new_facts
    assert Fact("perp", [Angle(*"CBD")]) in new_facts


def test_perp_and_midp_to_cong():
    """Test righttri_and_midp_to_cong"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("midp", ["M", Segment("A", "B")]))
    f = Fact("perp", [Angle(*"AHB")])
    diagram.database.add_fact(f)
    new_facts = perp_and_midp_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert list(new_facts)[0] == Fact(
        "cong", [Segment("H", "M"), Segment("M", "A")])


def test_midp_and_perp_to_cong():
    """Test midp_and_perp_to_cong"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("perp", [Angle(*"AHB")]))
    f = Fact("midp", ["M", Segment("A", "B")])
    new_facts = midp_and_perp_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert list(new_facts)[0] == Fact(
        "cong", [Segment("H", "M"), Segment("M", "A")])


def test_perp_and_eqline_and_cong_to_cong():
    """Test perp_and_eqline_and_cong_to_cong"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AC"), Segment(*"DA")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AC"), Segment(*"DC")]))
    diagram.database.add_fact(Fact("cong", [Segment(*"DB"), Segment(*"DC")]))
    f = Fact("perp", [Angle(*"ABC")])
    diagram.database.add_fact(f)
    new_facts = perp_and_eqline_and_cong_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("cong", [Segment(*"DA"), Segment(*"DC")])


def test_cong_and_eqline_and_perp_to_cong():
    """Test test_cong_and_eqline_and_perp_to_cong"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 0)
    diagram.point_dict["B"] = Point(0, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AC"), Segment(*"DA")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AC"), Segment(*"DC")]))
    diagram.database.add_fact(Fact("perp", [Angle(*"ABC")]))
    f = Fact("cong", [Segment(*"DB"), Segment(*"DC")])
    new_facts = cong_and_eqline_and_perp_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("cong", [Segment(*"DA"), Segment(*"DC")])


def test_eqline_and_cong_and_perp_to_cong():
    """Test eqline_and_cong_and_perp_to_cong"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("cong", [Segment(*"DB"), Segment(*"DC")]))
    diagram.database.add_fact(Fact("perp", [Angle(*"ABC")]))
    f = Fact("eqline", [Segment(*"DA"), Segment(*"DC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_cong_and_perp_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("cong", [Segment(*"DA"), Segment(*"DC")])
