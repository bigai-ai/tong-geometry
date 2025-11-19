r"""Test the para module in forward_chainer."""

from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Segment
from tonggeometry.inference_engine.rule.para import (eqline_and_para_to_para,
                                                     para_and_eqline_to_para,
                                                     para_to_eqangle,
                                                     para_to_eqline)


def test_para_to_eqangle():
    """Test para_to_eqangle"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(0, 0)
    diagram.point_dict['B'] = Point(1, 0)
    diagram.point_dict['C'] = Point(0, 1)
    diagram.point_dict['D'] = Point(1, 1)
    s1 = Segment("A", "B")
    s2 = Segment("C", "D")
    f = Fact("para", [s1, s2])
    diagram.database.add_fact(f)
    new_facts = para_to_eqangle(diagram, f)
    assert len(new_facts) == 4
    assert Fact(
        "eqangle",
        [Angle("C", "A", "B"), Angle("A", "C", "D")]) in new_facts
    assert Fact(
        "eqangle",
        [Angle("D", "A", "B"), Angle("A", "D", "C")]) in new_facts
    assert Fact(
        "eqangle",
        [Angle("C", "B", "A"), Angle("B", "C", "D")]) in new_facts
    assert Fact(
        "eqangle",
        [Angle("D", "B", "A"), Angle("B", "D", "C")]) in new_facts


def test_para_to_eqline():
    """Test para_to_eqline"""
    diagram = Diagram()
    s1 = Segment("A", "B")
    s2 = Segment("A", "D")
    f = Fact("para", [s1, s2])
    diagram.database.add_fact(f)
    new_facts = para_to_eqline(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqline", [s1, s2])


def test_para_and_eqline_to_para():
    """Test para_and_eqline_to_para"""
    diagram = Diagram()
    s1 = Segment("A", "B")
    s2 = Segment("C", "D")
    s3 = Segment("E", "F")
    s4 = Segment("G", "H")
    diagram.point_dict['A'] = Point(0, 0)
    diagram.point_dict['B'] = Point(1, 0)
    diagram.point_dict['E'] = Point(0, 1)
    diagram.point_dict['F'] = Point(1, 1)
    for s in [s1, s2, s3, s4]:
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("eqline", [s1, s2]))
    diagram.database.add_fact(Fact("eqline", [s3, s4]))
    f = Fact("para", [s1, s3])
    diagram.database.add_fact(f)
    new_facts = para_and_eqline_to_para(diagram, f)
    assert len(new_facts) == 2
    assert Fact("para", [s1, s4]) in new_facts
    assert Fact("para", [s2, s3]) in new_facts


def test_eqline_and_para_to_para():
    """Test eqline_and_para_to_para"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(0, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(1, 1)
    diagram.point_dict["E"] = Point(2, 0)
    diagram.point_dict["F"] = Point(2, 1)
    diagram.point_dict["G"] = Point(3, 0)
    diagram.point_dict["H"] = Point(3, 1)
    s1 = Segment("A", "B")
    s2 = Segment("C", "D")
    s3 = Segment("E", "F")
    s4 = Segment("G", "H")
    for s in [s1, s2, s3, s4]:
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("para", [s1, s2]))
    diagram.database.add_fact(Fact("para", [s3, s4]))
    f = Fact("eqline", [s1, s3])
    diagram.database.add_fact(f)
    new_facts = eqline_and_para_to_para(diagram, f)
    assert len(new_facts) == 2
    assert Fact("para", [s1, s4]) in new_facts
    assert Fact("para", [s2, s3]) in new_facts
