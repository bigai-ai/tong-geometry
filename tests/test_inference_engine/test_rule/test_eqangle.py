r"""Test the eqangle module in forward_chainer."""

from itertools import combinations

from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Ratio, Segment
from tonggeometry.inference_engine.rule.eqangle import (
    eqangle_and_eqangle_and_eqangle_to_eqangle_isoquadtri,
    eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad,
    eqangle_and_eqangle_to_eqangle, eqangle_and_eqangle_to_eqangle_bisector,
    eqangle_and_eqline_to_eqangle, eqangle_and_eqline_to_eqline,
    eqangle_and_eqline_to_eqratio, eqangle_and_eqline_to_para,
    eqangle_and_para_to_para, eqangle_and_perp_to_eqangle,
    eqangle_and_perp_to_perp, eqangle_to_eqangle, eqangle_to_eqline_or_perp,
    eqline_and_eqangle_to_eqangle, eqline_and_eqangle_to_eqline,
    eqline_and_eqangle_to_eqratio, eqline_and_eqangle_to_para,
    eqline_and_eqratio_to_eqangle, eqratio_and_eqline_to_eqangle,
    para_and_eqangle_to_para, perp_and_eqangle_to_eqangle,
    perp_and_eqangle_to_perp)


def test_eqangle_to_eqangle():
    """Test eqangle_to_eqangle"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDE", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"BC")]))
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DBE")])
    diagram.database.add_fact(f)
    new_facts = eqangle_to_eqangle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"ABD"), Angle(*"CBE")])


def test_eqangle_and_eqline_to_eqline():
    """Test eqangle_and_eqline_to_eqline"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDEF", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"BC")]))
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqline_to_eqline(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqline", [Segment(*"DE"), Segment(*"EF")])


def test_eqline_and_eqangle_to_eqline():
    """Test eqline_and_eqangle_to_eqline"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDEF", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")]))
    f = Fact("eqline", [Segment(*"AB"), Segment(*"BC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqangle_to_eqline(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqline", [Segment(*"DE"), Segment(*"EF")])


def test_eqangle_and_perp_to_perp():
    """Test eqangle_and_perp_to_perp"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("perp", [Angle(*"ABC")]))
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_perp_to_perp(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("perp", [Angle(*"DEF")])


def test_perp_and_eqangle_to_perp():
    """Test perp_and_eqangle_to_perp"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")]))
    f = Fact("perp", [Angle(*"ABC")])
    diagram.database.add_fact(f)
    new_facts = perp_and_eqangle_to_perp(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("perp", [Angle(*"DEF")])


def test_eqangle_and_para_to_para():
    """Test eqangle_and_para_to_para"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("para", [Segment(*"AB"), Segment(*"DE")]))
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_para_to_para(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("para", [Segment(*"BC"), Segment(*"EF")])


def test_para_and_eqangle_to_para():
    """Test para_and_eqangle_to_para"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(1, 0)
    diagram.point_dict['B'] = Point(0, 0)
    diagram.point_dict['C'] = Point(1, 1)
    diagram.point_dict['D'] = Point(1, 1)
    diagram.point_dict['E'] = Point(0, 1)
    diagram.point_dict['F'] = Point(1, 2)
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")]))
    f = Fact("para", [Segment(*"AB"), Segment(*"DE")])
    diagram.database.add_fact(f)
    new_facts = para_and_eqangle_to_para(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("para", [Segment(*"BC"), Segment(*"EF")])


def test_eqangle_and_eqline_to_para():
    """Test eqangle_and_eqline_to_para"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"AB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"DE"), Segment(*"DE")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"DE")]))
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqline_to_para(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("para", [Segment(*"BC"), Segment(*"EF")])


def test_eqline_and_eqangle_to_para():
    """Test eqline_and_eqangle_to_para"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"AB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"DE"), Segment(*"DE")]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")]))
    f = Fact("eqline", [Segment(*"AB"), Segment(*"DE")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqangle_to_para(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("para", [Segment(*"BC"), Segment(*"EF")])


def test_eqangle_to_eqline_or_perp_eqline():
    """Test eqangle_to_eqline_or_perp"""
    diagram = Diagram()
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"CBA")])
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(0, 1)
    diagram.point_dict["C"] = Point(0, 2)
    assert len(diagram.database.add_fact(f)) == 0
    new_facts = eqangle_to_eqline_or_perp(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact(
        "eqline", [Segment("A", "B"), Segment("B", "C")])


def test_eqangle_to_eqline_or_perp_perp():
    """Test eqangle_to_eqline_or_perp"""
    diagram = Diagram()
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"CBA")])
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(0, 1)
    diagram.point_dict["C"] = Point(1, 1)
    assert len(diagram.database.add_fact(f)) == 0
    new_facts = eqangle_to_eqline_or_perp(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("perp", [Angle(*"ABC")])


def test_eqline_and_eqangle_to_eqangle():
    """Test eqline_and_eqangle_to_eqangle"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"AB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BC"), Segment(*"BC")]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABD"), Angle(*"DEF")]))
    f = Fact("eqline", [Segment(*"AB"), Segment(*"BC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqangle_to_eqangle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"ABD"), Angle(*"CBD")])


def test_eqangle_and_eqline_to_eqangle():
    """Test eqangle_and_eqline_to_eqangle"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"AB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BC"), Segment(*"BC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BD"), Segment(*"BD")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"DE"), Segment(*"DE")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EF"), Segment(*"EF")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"BC")]))
    f = Fact("eqangle", [Angle(*"ABD"), Angle(*"DEF")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqline_to_eqangle(diagram, f)
    assert len(new_facts) == 2
    print(new_facts)
    assert Fact("eqangle", [Angle(*"ABD"), Angle(*"CBD")]) in new_facts
    assert Fact("eqangle", [Angle(*"CBD"), Angle(*"DEF")]) in new_facts


def test_eqangle_and_eqangle_to_eqangle():
    """Test eqangle_and_eqangle_to_eqangle"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(1, 0)
    diagram.point_dict['F'] = Point(0, 0)
    diagram.point_dict['C'] = Point(1, 1)
    diagram.database.add_fact(Fact("eqangle", [Angle(*"FAC"), Angle(*"CED")]))
    f = Fact("eqangle", [Angle(*"ACF"), Angle(*"DEA")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqangle_to_eqangle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"AFC"), Angle(*"CEA")])


def test_eqangle_and_eqangle_to_eqangle_bisector():
    """Test eqangle_and_eqangle_to_eqangle_bisector"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(1, 0)
    diagram.point_dict['C'] = Point(0, 0)
    diagram.point_dict['I'] = Point(1, 1)
    diagram.database.add_fact(Fact("eqangle", [Angle(*"BAI"), Angle(*"IAC")]))
    f = Fact("eqangle", [Angle(*"ACI"), Angle(*"ICB")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqangle_to_eqangle_bisector(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"ABI"), Angle(*"IBC")])


def test_eqangle_and_eqline_to_eqratio():
    """Test eqangle_and_eqline_to_eqratio"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-1, 0)
    diagram.point_dict['B'] = Point(0, 1)
    diagram.point_dict['C'] = Point(1, 0)
    diagram.point_dict['D'] = Point(0, 0)
    for p1, p2 in combinations("ABCD", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AD"), Segment(*"DC")]))
    f = Fact("eqangle", [Angle(*"ABD"), Angle(*"DBC")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqline_to_eqratio(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment(*"AD"), Segment(*"AB")),
        Ratio(Segment(*"DC"), Segment(*"BC"))
    ])


def test_eqline_and_eqangle_to_eqratio():
    """Test eqline_and_eqangle_to_eqratio"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-1, 0)
    diagram.point_dict['B'] = Point(0, 1)
    diagram.point_dict['C'] = Point(1, 0)
    diagram.point_dict['D'] = Point(0, 0)
    for p1, p2 in combinations("ABCD", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABD"), Angle(*"DBC")]))
    f = Fact("eqline", [Segment(*"AD"), Segment(*"DC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqangle_to_eqratio(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment(*"AD"), Segment(*"AB")),
        Ratio(Segment(*"DC"), Segment(*"BC"))
    ])


def test_eqratio_and_eqline_to_eqangle():
    """Test eqratio_and_eqline_to_eqangle"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-1, 0)
    diagram.point_dict['B'] = Point(0, 1)
    diagram.point_dict['C'] = Point(1, 0)
    diagram.point_dict['D'] = Point(0, 0)
    for p1, p2 in combinations("ABCD", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AD"), Segment(*"DC")]))
    f = Fact("eqratio", [
        Ratio(Segment(*"AD"), Segment(*"AB")),
        Ratio(Segment(*"DC"), Segment(*"BC"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_eqline_to_eqangle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"ABD"), Angle(*"DBC")])


def test_eqline_and_eqratio_to_eqangle():
    """Test eqline_and_eqratio_to_eqangle"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-1, 0)
    diagram.point_dict['B'] = Point(0, 1)
    diagram.point_dict['C'] = Point(1, 0)
    diagram.point_dict['D'] = Point(0, 0)
    for p1, p2 in combinations("ABCD", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment(*"AD"), Segment(*"AB")),
            Ratio(Segment(*"DC"), Segment(*"BC"))
        ]))
    f = Fact("eqline", [Segment(*"AD"), Segment(*"DC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqratio_to_eqangle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"ABD"), Angle(*"DBC")])


def test_eqangle_and_perp_to_eqangle():
    """Test eqangle_and_perp_to_eqangle"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-1, 1)
    diagram.point_dict['D'] = Point(0, 0)
    diagram.point_dict['B'] = Point(-1, 0)
    diagram.point_dict['C'] = Point(-1, -1)
    for p1, p2 in combinations("ABCDE", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("perp", [Angle(*"BDE")]))
    f = Fact("eqangle", [Angle(*"BDA"), Angle(*"CDB")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_perp_to_eqangle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"ADE"), Angle(*"EDC")])


def test_perp_and_eqangle_to_eqangle():
    """Test perp_and_eqangle_to_eqangle"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-1, 1)
    diagram.point_dict['D'] = Point(0, 0)
    diagram.point_dict['B'] = Point(-1, 0)
    diagram.point_dict['C'] = Point(-1, -1)
    diagram.point_dict['E'] = Point(0, 1)
    diagram.point_dict['F'] = Point(-1, 1)
    diagram.point_dict['H'] = Point(1, 1)
    for p1, p2 in combinations("ABCDEFH", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"BDC"), Angle(*"ADB")]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"FDE"), Angle(*"EDH")]))
    f = Fact("perp", [Angle(*"BDE")])
    diagram.database.add_fact(f)
    new_facts = perp_and_eqangle_to_eqangle(diagram, f)
    assert len(new_facts) == 2
    assert Fact("eqangle", [Angle(*"ADE"), Angle(*"EDC")]) in new_facts
    assert Fact("eqangle", [Angle(*"FDB"), Angle(*"BDH")]) in new_facts


def test_eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad():
    """Test eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-1, 1)
    diagram.point_dict['B'] = Point(-1, -1)
    diagram.point_dict['C'] = Point(1, -1)
    diagram.point_dict['D'] = Point(1, 1)
    diagram.point_dict['P'] = Point(0, 0.5)
    diagram.point_dict['Q'] = Point(0, 0.5)
    for p1, p2 in combinations("ABCDPQ", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    for p1, p2 in combinations("ABCPQ", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"APB"), Angle(*"DPC")]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABP"), Angle(*"QBD")]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"CDP"), Angle(*"QDA")]))
    f = Fact("eqangle", [Angle(*"DAP"), Angle(*"QAB")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqangle_and_eqangle_to_eqangle_isotriquad(
        diagram, f)
    assert len(new_facts) == 4
    assert Fact("eqangle", [Angle(*"BDP"), Angle(*"QDA")]) in new_facts
    assert Fact("eqangle", [Angle(*"BCP"), Angle(*"QCD")]) in new_facts
    assert Fact("eqangle", [Angle(*"ABP"), Angle(*"QBC")]) in new_facts
    assert Fact("eqangle", [Angle(*"AQB"), Angle(*"DQC")]) in new_facts


def test_eqangle_and_eqangle_and_eqangle_to_eqangle_isoquadtri():
    """Test eqangle_and_eqangle_and_eqangle_to_eqangle_isoquadtri"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-1, 1)
    diagram.point_dict['B'] = Point(-1, -1)
    diagram.point_dict['C'] = Point(1, -1)
    diagram.point_dict['D'] = Point(1, 1)
    diagram.point_dict['P'] = Point(0, 0.5)
    diagram.point_dict['Q'] = Point(0, 0.5)
    for p1, p2 in combinations("ABCDPQ", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"DAP"), Angle(*"QAB")]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"ABP"), Angle(*"QBC")]))
    f = Fact("eqangle", [Angle(*"APB"), Angle(*"DPC")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_eqangle_and_eqangle_to_eqangle_isoquadtri(
        diagram, f)
    assert len(new_facts) == 3
    assert Fact("eqangle", [Angle(*"AQB"), Angle(*"DQC")]) in new_facts
    assert Fact("eqangle", [Angle(*"BCP"), Angle(*"QCD")]) in new_facts
    assert Fact("eqangle", [Angle(*"CDP"), Angle(*"QDA")]) in new_facts
