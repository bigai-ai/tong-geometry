r"""Test the cong module in forward_chainer."""

from itertools import combinations

from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Ratio, Segment
from tonggeometry.inference_engine.rule.cong import (
    cong_and_eqline_to_cong, cong_and_eqline_to_l, cong_and_eqline_to_midp,
    cong_to_eqangle, cong_to_eqratio, cong_to_ll_radius, cong_to_ll_stick,
    eqangle_to_cong, eqline_and_cong_to_cong, eqline_and_cong_to_l,
    eqline_and_cong_to_midp, eqratio_to_ll)


def test_cong_to_eqangle():
    """Test cong_to_eqangle"""
    diagram = Diagram()
    diagram.point_dict["B"] = Point(0, 1)
    diagram.point_dict["A"] = Point(-1, 0)
    diagram.point_dict["C"] = Point(1, 0)
    f = Fact("cong", [Segment(*"AB"), Segment(*"BC")])
    diagram.database.add_fact(f)
    new_facts = cong_to_eqangle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"BAC"), Angle(*"ACB")])


def test_eqangle_to_cong():
    """Test eqangle_to_cong"""
    diagram = Diagram()
    diagram.point_dict["B"] = Point(0, 1)
    diagram.point_dict["A"] = Point(-1, 0)
    diagram.point_dict["C"] = Point(1, 0)
    for p1, p2 in combinations("ABC", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    f = Fact("eqangle", [Angle(*"CAB"), Angle(*"BCA")])
    diagram.database.add_fact(f)
    new_facts = eqangle_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("cong", [Segment(*"AB"), Segment(*"BC")])


def test_cong_and_eqline_to_midp():
    """Test cong_and_eqline_to_midp"""
    diagram = Diagram()
    for p1, p2 in combinations("ABC", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"BC")]))
    f = Fact("cong", [Segment(*"AB"), Segment(*"BC")])
    diagram.database.add_fact(f)
    new_facts = cong_and_eqline_to_midp(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("midp", ["B", Segment(*"AC")])


def test_eqline_and_cong_to_midp():
    """Test eqline_and_cong_to_midp"""
    diagram = Diagram()
    for p1, p2 in combinations("ABC", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("cong", [Segment(*"AB"), Segment(*"BC")]))
    f = Fact("eqline", [Segment(*"AB"), Segment(*"BC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_cong_to_midp(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("midp", ["B", Segment(*"AC")])


def test_cong_to_eqratio():
    """Test cong_to_eqratio"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(0, 1)
    diagram.point_dict["C"] = Point(0, 2)
    diagram.point_dict["D"] = Point(0, 3)
    diagram.point_dict["E"] = Point(0, 2)
    diagram.point_dict["F"] = Point(0, 4)
    diagram.point_dict["G"] = Point(0, 6)
    diagram.point_dict["H"] = Point(0, 8)
    diagram.database.add_fact(Fact("cong", [Segment(*"AB"), Segment(*"CD")]))
    f = Fact("cong", [Segment(*"EF"), Segment(*"GH")])
    diagram.database.add_fact(f)
    new_facts = cong_to_eqratio(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"EF")),
        Ratio(Segment(*"AB"), Segment(*"GH"))
    ])


def test_cong_and_eqline_to_cong():
    """Test cong_and_eqline_to_cong"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 1)
    diagram.point_dict["B"] = Point(-2, 2)
    diagram.point_dict["C"] = Point(1, -1)
    diagram.point_dict["D"] = Point(2, -2)
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AB"), Segment(*"CD")]))
    f = Fact("cong", [Segment(*"AB"), Segment(*"CD")])
    diagram.database.add_fact(f)
    new_facts = cong_and_eqline_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("cong", [Segment(*"BC"), Segment(*"AD")])


def test_eqline_and_cong_to_cong():
    """Test eqline_and_cong_to_cong"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 1)
    diagram.point_dict["B"] = Point(-2, 2)
    diagram.point_dict["C"] = Point(1, -1)
    diagram.point_dict["D"] = Point(2, -2)
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("cong", [Segment(*"AB"), Segment(*"CD")]))
    f = Fact("eqline", [Segment(*"AB"), Segment(*"CD")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_cong_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("cong", [Segment(*"BC"), Segment(*"AD")])


def test_cong_and_eqline_to_l():
    """Test cong_and_eqline_to_l"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 0)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0.5, 1)
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BC"), Segment(*"AB")]))
    f = Fact("cong", [Segment(*"DB"), Segment(*"DC")])
    diagram.database.add_fact(f)
    new_facts = cong_and_eqline_to_l(diagram, f)
    assert len(new_facts) == 0
    assert len(diagram.database.l) == 1
    assert diagram.database.l["DABC"] == "out"
    assert len(diagram.database.l_stick) == 1
    assert diagram.database.l_stick["DA"]["BC"] is None
    assert len(diagram.database.l_radius) == 1
    assert diagram.database.l_radius["DB"]["AC"] is None
    assert len(diagram.database.l_ratio) == 1
    assert diagram.database.l_ratio["ABC"]["D"] is None


def test_eqline_and_cong_to_l():
    """Test eqline_and_cong_to_l"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 0)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0.5, 1)
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(Fact("cong", [Segment(*"DB"), Segment(*"DC")]))
    f = Fact("eqline", [Segment(*"AB"), Segment(*"AC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_cong_to_l(diagram, f)
    assert len(new_facts) == 0
    assert len(diagram.database.l) == 1
    assert diagram.database.l["DABC"] == "out"
    assert len(diagram.database.l_stick) == 1
    assert diagram.database.l_stick["DA"]["BC"] is None
    assert len(diagram.database.l_radius) == 1
    assert diagram.database.l_radius["DB"]["AC"] is None
    assert len(diagram.database.l_ratio) == 1
    assert diagram.database.l_ratio["ABC"]["D"] is None


def test_l_and_cong_to_ll_stick():
    """Test l_and_cong_to_ll_stick"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 0)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0.5, 1)
    for p1, p2 in combinations("ABCDEFGH", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.l["ADXY"] = "out"
    diagram.database.l["FEGH"] = "out"
    diagram.database.l_stick["AD"] = {"XY": None}
    diagram.database.l_stick["FE"] = {"GH": None}
    diagram.database.add_fact(Fact("cong", [Segment(*"DB"), Segment(*"DC")]))
    diagram.database.add_fact(Fact("cong", [Segment(*"AD"), Segment(*"EF")]))
    diagram.database.add_fact(Fact("cong", [Segment(*"DB"), Segment(*"AX")]))
    diagram.database.add_fact(Fact("cong", [Segment(*"AX"), Segment(*"AY")]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment(*"AB"), Segment(*"EG")),
            Ratio(Segment(*"EH"), Segment(*"AC"))
        ]))
    f = Fact("eqline", [Segment(*"AB"), Segment(*"AC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_cong_to_l(diagram, f)
    assert len(new_facts) == 2
    assert Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"DX")),
        Ratio(Segment(*"DY"), Segment(*"AC"))
    ]) in new_facts
    assert Fact("cong", [Segment(*"FG"), Segment(*"DB")]) in new_facts


def test_l_and_cong_to_ll_radius():
    """Test l_and_cong_to_ll_radius"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 0)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(1, 0)
    diagram.point_dict["D"] = Point(0.5, 1)
    for p1, p2 in combinations("ABCDEFGH", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.l["BYDX"] = "out"
    diagram.database.l["FEGH"] = "out"
    diagram.database.l_radius["BD"] = {"YX": None}
    diagram.database.l_radius["FG"] = {"EH": None}
    diagram.database.add_fact(Fact("cong", [Segment(*"DB"), Segment(*"DC")]))
    diagram.database.add_fact(Fact("cong", [Segment(*"BD"), Segment(*"BY")]))
    diagram.database.add_fact(Fact("cong", [Segment(*"DB"), Segment(*"FG")]))
    diagram.database.add_fact(Fact("cong", [Segment(*"AD"), Segment(*"BY")]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment(*"AB"), Segment(*"EG")),
            Ratio(Segment(*"EH"), Segment(*"AC"))
        ]))
    f = Fact("eqline", [Segment(*"AB"), Segment(*"AC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_cong_to_l(diagram, f)
    assert len(new_facts) == 2
    assert Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"YD")),
        Ratio(Segment(*"YX"), Segment(*"AC"))
    ]) in new_facts
    assert Fact("cong", [Segment(*"EF"), Segment(*"AD")]) in new_facts


def test_cong_to_ll_stick():
    """Test cong_to_ll_stick"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDEFGH", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.l["DABC"] = "out"
    diagram.database.l["FEGH"] = "out"
    diagram.database.l["EFIJ"] = "out"
    diagram.database.l_stick["DA"] = {"BC": None}
    diagram.database.l_stick["FE"] = {"GH": None}
    diagram.database.l_stick["EF"] = {"IJ": None}
    diagram.database.add_fact(Fact("cong", [Segment(*"BD"), Segment(*"FG")]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment(*"AB"), Segment(*"FI")),
            Ratio(Segment(*"FJ"), Segment(*"AC"))
        ]))
    f = Fact("cong", [Segment(*"DA"), Segment(*"EF")])
    diagram.database.add_fact(f)
    new_facts = cong_to_ll_stick(diagram, f)
    assert len(new_facts) == 2
    assert Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"EG")),
        Ratio(Segment(*"EH"), Segment(*"AC"))
    ]) in new_facts
    assert Fact("cong", [Segment(*"BD"), Segment(*"EI")]) in new_facts


def test_cong_to_ll_radius():
    """Test cong_to_ll_radius"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDEFGH", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.l["DABC"] = "out"
    diagram.database.l["FEGH"] = "out"
    diagram.database.l["GIFJ"] = "out"
    diagram.database.l_radius["DB"] = {"AC": None}
    diagram.database.l_radius["FG"] = {"EH": None}
    diagram.database.l_radius["GF"] = {"IJ": None}
    diagram.database.add_fact(Fact("cong", [Segment(*"AD"), Segment(*"EF")]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment(*"AB"), Segment(*"FI")),
            Ratio(Segment(*"IJ"), Segment(*"AC"))
        ]))
    f = Fact("cong", [Segment(*"BD"), Segment(*"FG")])
    diagram.database.add_fact(f)
    new_facts = cong_to_ll_radius(diagram, f)
    assert len(new_facts) == 2
    assert Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"EG")),
        Ratio(Segment(*"EH"), Segment(*"AC"))
    ]) in new_facts
    assert Fact("cong", [Segment(*"AD"), Segment(*"IG")]) in new_facts


def test_eqratio_to_ll():
    """Test eqratio_to_ll"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCDEFGH", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.l["DABC"] = "out"
    diagram.database.l["FEGH"] = "out"
    diagram.database.l["IEGH"] = "out"
    diagram.database.l_ratio["ABC"] = {"D": None}
    diagram.database.l_ratio["EGH"] = {"F": None, "I": None}
    diagram.database.add_fact(Fact("cong", [Segment(*"AD"), Segment(*"EF")]))
    diagram.database.add_fact(Fact("cong", [Segment(*"BD"), Segment(*"GI")]))
    f = Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"EG")),
        Ratio(Segment(*"EH"), Segment(*"AC"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_to_ll(diagram, f)
    assert len(new_facts) == 2
    assert Fact("cong", [Segment(*"BD"), Segment(*"GF")]) in new_facts
    assert Fact("cong", [Segment(*"AD"), Segment(*"EI")]) in new_facts
