r"""Test the simtri module in forward_chainer."""

from itertools import combinations

from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import (Angle, Ratio, Segment,
                                                      Triangle)
from tonggeometry.inference_engine.rule.simtri import (
    cong_and_simtri_to_contri, eqangle_and_eqangle_and_eqangle_to_simtri,
    eqangle_and_eqratio_to_simtri, eqratio_and_eqangle_to_simtri,
    eqratio_and_eqratio_to_simtri, simtri_and_cong_to_contri,
    simtri_to_eqangle, simtri_to_eqratio)


def test_eqratio_and_eqratio_to_simtri():
    """Test eqratio_and_eqratio_to_simtri"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment(*"AB"), Segment(*"DE")),
            Ratio(Segment(*"BC"), Segment(*"EF"))
        ]))
    f = Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"DE")),
        Ratio(Segment(*"AC"), Segment(*"DF"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_eqratio_to_simtri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("simtri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_eqangle_and_eqangle_and_eqangle_to_simtri():
    """Test eqangle_and_eqangle_and_eqangle_to_simtri"""
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
    f = Fact("eqangle", [Angle(*"BAC"), Angle(*"FDE")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqangle_and_eqangle_to_simtri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("simtri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_eqratio_and_eqangle_to_simtri_SAS():
    """Test eqratio_and_eqangle_to_simtri"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABC"), Angle(*"FED")]))
    f = Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"DE")),
        Ratio(Segment(*"BC"), Segment(*"EF"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_eqangle_to_simtri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("simtri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_eqratio_and_eqangle_to_simtri_SSA():
    """Test eqratio_and_eqangle_to_simtri"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    diagram.database.add_fact(Fact("perp", [Angle(*"BAC")]))
    diagram.database.add_fact(Fact("perp", [Angle(*"EDF")]))
    f = Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"DE")),
        Ratio(Segment(*"BC"), Segment(*"EF"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_eqangle_to_simtri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("simtri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_eqangle_and_eqratio_to_simtri_SAS():
    """Test eqangle_and_eqratio_to_simtri"""
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
        Fact("eqratio", [
            Ratio(Segment(*"AB"), Segment(*"DE")),
            Ratio(Segment(*"BC"), Segment(*"EF"))
        ]))
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"FED")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqratio_to_simtri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("simtri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_eqangle_and_eqratio_to_simtri_SSA():
    """Test eqangle_and_eqratio_to_simtri"""
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
    diagram.database.add_fact(Fact("perp", [Angle(*"BCA")]))
    diagram.database.add_fact(Fact("perp", [Angle(*"DFE")]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment(*"AB"), Segment(*"ED")),
            Ratio(Segment(*"BC"), Segment(*"EF"))
        ]))
    f = Fact("eqangle", [Angle(*"BCA"), Angle(*"DFE")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqratio_to_simtri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("simtri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_simtri_to_eqratio():
    """Test simtri_to_eqratio"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    f = Fact("simtri", [Triangle("A", "B", "C"), Triangle("D", "E", "F")])
    diagram.database.add_fact(f)
    new_facts = simtri_to_eqratio(diagram, f)
    assert len(new_facts) == 3
    assert Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"DE")),
        Ratio(Segment(*"AC"), Segment(*"DF"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"DE")),
        Ratio(Segment(*"BC"), Segment(*"EF"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"AC"), Segment(*"DF")),
        Ratio(Segment(*"BC"), Segment(*"EF"))
    ]) in new_facts


def test_simtri_to_eqangle():
    """Test simtri_to_eqangle"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    f = Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])
    diagram.database.add_fact(f)
    new_facts = simtri_to_eqangle(diagram, f)
    assert len(new_facts) == 3
    assert Fact("eqangle", [Angle(*"ABC"), Angle(*"FED")]) in new_facts
    assert Fact("eqangle", [Angle(*"BCA"), Angle(*"DFE")]) in new_facts
    assert Fact("eqangle", [Angle(*"CAB"), Angle(*"EDF")]) in new_facts


def test_simtri_and_cong_to_contri():
    """Tesst simtri_and_cong_to_contri"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 1)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0, 0)
    diagram.point_dict["E"] = Point(-1, 1)
    diagram.point_dict["F"] = Point(-1, 0)
    diagram.database.add_fact(
        Fact("cong", [Segment("A", "B"), Segment("E", "D")]))
    f = Fact("simtri", [Triangle("A", "B", "C"), Triangle("D", "E", "F")])
    diagram.database.add_fact(f)
    new_facts = simtri_and_cong_to_contri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])


def test_cong_and_simtri_to_contri():
    """Test cong_and_simtri_to_contri"""
    diagram = Diagram()
    diagram.database.add_fact(
        Fact("simtri", [Triangle("A", "B", "C"),
                        Triangle("D", "E", "F")]))
    f = Fact("cong", [Segment("A", "B"), Segment("E", "D")])
    diagram.database.add_fact(f)
    new_facts = cong_and_simtri_to_contri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])
