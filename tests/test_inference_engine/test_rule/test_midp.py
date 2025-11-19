r"""Test the midp module in forward_chainer."""

from itertools import combinations

from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Ratio, Segment
from tonggeometry.inference_engine.rule.midp import (
    eqline_and_midp_to_centri, eqline_to_centri, midp_and_eqline_to_centri,
    midp_and_midp_to_eqratio, midp_to_centri, midp_to_cong, midp_to_eqline,
    midp_to_eqratio)


def test_midp_to_eqline():
    """Test midp_to_eqline"""
    diagram = Diagram()
    f = Fact("midp", ["A", Segment("B", "C")])
    diagram.database.add_fact(f)
    new_facts = midp_to_eqline(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact(
        "eqline", [Segment("A", "B"), Segment("A", "C")])


def test_midp_to_cong():
    """Test midp_to_cong"""
    diagram = Diagram()
    f = Fact("midp", ["A", Segment("B", "C")])
    diagram.database.add_fact(f)
    new_facts = midp_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("cong", [Segment("A", "B"), Segment("A", "C")])


def test_midp_to_eqratio():
    """Test midp_to_eqratio"""
    diagram = Diagram()
    f = Fact("midp", ["A", Segment("B", "C")])
    diagram.database.add_fact(f)
    new_facts = midp_to_eqratio(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("B", "C")),
        Ratio(Segment("A", "C"), Segment("B", "C"))
    ])


def test_midp_and_midp_to_eqratio():
    """Test midp_and_midp_to_eqratio"""
    diagram = Diagram()
    diagram.database.add_fact(Fact("midp", ["A", Segment("B", "C")]))
    f = Fact("midp", ["A", Segment("X", "Y")])
    diagram.database.add_fact(f)
    new_facts = midp_and_midp_to_eqratio(diagram, f)
    assert len(new_facts) == 4
    assert Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("B", "C")),
        Ratio(Segment("A", "X"), Segment("X", "Y"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment("A", "C"), Segment("B", "C")),
        Ratio(Segment("A", "X"), Segment("X", "Y"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("B", "C")),
        Ratio(Segment("A", "Y"), Segment("X", "Y"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment("A", "C"), Segment("B", "C")),
        Ratio(Segment("A", "Y"), Segment("X", "Y"))
    ]) in new_facts


def test_midp_and_eqline_to_centri_midp():
    """Test midp_and_eqline_to_centri"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-2, 0)
    diagram.point_dict['C'] = Point(2, 0)
    diagram.point_dict['B'] = Point(0, 2)
    diagram.point_dict['F'] = Point(-1, 1)
    diagram.point_dict['D'] = Point(1, 1)
    diagram.point_dict['G'] = Point(0, 1)
    diagram.point_dict['E'] = Point(0, 0)
    for p1, p2 in combinations("ABCDEFG", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("midp", ["F", Segment(*"AB")]))
    diagram.database.add_fact(Fact("midp", ["D", Segment(*"BC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"GC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"FC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"GB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"EB")]))
    f = Fact("midp", ["E", Segment(*"AC")])
    diagram.database.add_fact(f)
    new_facts = midp_and_eqline_to_centri(diagram, f)
    assert len(new_facts) == 3
    assert Fact("eqline", [Segment(*"AG"), Segment(*"GD")]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"GB"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"BG"))
    ]) in new_facts


def test_midp_and_eqline_to_centri_eqline():
    """Test midp_and_eqline_to_centri"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-2, 0)
    diagram.point_dict['C'] = Point(2, 0)
    diagram.point_dict['B'] = Point(0, 2)
    diagram.point_dict['F'] = Point(-1, 1)
    diagram.point_dict['D'] = Point(1, 1)
    diagram.point_dict['G'] = Point(0, 1)
    diagram.point_dict['E'] = Point(0, 0)
    for p1, p2 in combinations("ABCDEFG", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("midp", ["F", Segment(*"AB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"GC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"FC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"GC"), Segment(*"FC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"GB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"EB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"GB"), Segment(*"EB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"CG"), Segment(*"CF")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"CG"), Segment(*"GF")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"CF"), Segment(*"GF")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BD"), Segment(*"DC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BD"), Segment(*"BC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"DC"), Segment(*"BC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AG"), Segment(*"GD")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AD"), Segment(*"GD")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AD"), Segment(*"AG")]))
    f = Fact("midp", ["E", Segment(*"AC")])
    diagram.database.add_fact(f)
    new_facts = midp_and_eqline_to_centri(diagram, f)
    assert len(new_facts) == 3
    assert Fact("midp", ["D", Segment(*"BC")]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"GB"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"BG"))
    ]) in new_facts


def test_eqline_and_midp_to_centri_midp():
    """Test eqline_and_midp_to_centri"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-2, 0)
    diagram.point_dict['C'] = Point(2, 0)
    diagram.point_dict['B'] = Point(0, 2)
    diagram.point_dict['F'] = Point(-1, 1)
    diagram.point_dict['D'] = Point(1, 1)
    diagram.point_dict['G'] = Point(0, 1)
    diagram.point_dict['E'] = Point(0, 0)
    for p1, p2 in combinations("ABCDEFG", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("midp", ["F", Segment(*"AB")]))
    diagram.database.add_fact(Fact("midp", ["D", Segment(*"BC")]))
    diagram.database.add_fact(Fact("midp", ["E", Segment(*"AC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"FC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"GB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"EB")]))
    f = Fact("eqline", [Segment(*"FG"), Segment(*"GC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_midp_to_centri(diagram, f)
    assert len(new_facts) == 3
    assert Fact("eqline", [Segment(*"AG"), Segment(*"GD")]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"GB"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"BG"))
    ]) in new_facts


def test_eqline_and_midp_to_centri_eqline():
    """Test eqline_and_midp_to_centri"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-2, 0)
    diagram.point_dict['C'] = Point(2, 0)
    diagram.point_dict['B'] = Point(0, 2)
    diagram.point_dict['F'] = Point(-1, 1)
    diagram.point_dict['D'] = Point(1, 1)
    diagram.point_dict['G'] = Point(0, 1)
    diagram.point_dict['E'] = Point(0, 0)
    for p1, p2 in combinations("ABCDEFG", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("midp", ["F", Segment(*"AB")]))
    diagram.database.add_fact(Fact("midp", ["E", Segment(*"AC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"GC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"FC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"GC"), Segment(*"FC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"GB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"EB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"GB"), Segment(*"EB")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"CG"), Segment(*"CF")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"CG"), Segment(*"GF")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"CF"), Segment(*"GF")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BD"), Segment(*"DC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"BD"), Segment(*"BC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"DC"), Segment(*"BC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AG"), Segment(*"GD")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AD"), Segment(*"GD")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"AD"), Segment(*"AG")]))
    f = Fact("eqline", [Segment(*"FG"), Segment(*"GC")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_midp_to_centri(diagram, f)
    assert len(new_facts) == 3
    assert Fact("midp", ["D", Segment(*"BC")]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"GB"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"BG"))
    ]) in new_facts


def test_midp_to_centri():
    """Test midp_to_centri"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-2, 0)
    diagram.point_dict['C'] = Point(2, 0)
    diagram.point_dict['B'] = Point(0, 2)
    diagram.point_dict['F'] = Point(-1, 1)
    diagram.point_dict['D'] = Point(1, 1)
    diagram.point_dict['G'] = Point(0, 1)
    diagram.point_dict['E'] = Point(0, 0)
    for p1, p2 in combinations("ABCDEFG", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("midp", ["F", Segment(*"AB")]))
    diagram.database.add_fact(Fact("midp", ["E", Segment(*"AC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"GC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"FC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"EB")]))
    f = Fact("eqline", [Segment(*"EG"), Segment(*"GB")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_midp_to_centri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"GB"))
    ])
    f = Fact("midp", ["D", Segment(*"BC")])
    diagram.database.add_fact(f)
    new_facts = midp_to_centri(diagram, f)
    assert len(new_facts) == 2
    assert Fact("eqline", [Segment(*"AG"), Segment(*"GD")]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"DG"), Segment(*"AG")),
        Ratio(Segment(*"EG"), Segment(*"BG"))
    ]) in new_facts


def test_eqline_to_centri_h():
    """Test eqline_to_centri"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-2, 0)
    diagram.point_dict['C'] = Point(2, 0)
    diagram.point_dict['B'] = Point(0, 2)
    diagram.point_dict['F'] = Point(-1, 1)
    diagram.point_dict['D'] = Point(1, 1)
    diagram.point_dict['G'] = Point(0, 1)
    diagram.point_dict['E'] = Point(0, 0)
    for p1, p2 in combinations("ABCDEFG", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("midp", ["F", Segment(*"AB")]))
    diagram.database.add_fact(Fact("midp", ["E", Segment(*"AC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"GC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"FC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"EB")]))
    f = Fact("eqline", [Segment(*"EG"), Segment(*"GB")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_midp_to_centri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"GB"))
    ])
    diagram.database.add_fact(Fact("eqline", [Segment(*"AG"), Segment(*"GD")]))
    f = Fact("eqline", [Segment(*"BD"), Segment(*"CD")])
    diagram.database.add_fact(f)
    new_facts = eqline_to_centri(diagram, f)
    assert len(new_facts) == 2
    assert Fact("midp", ["D", Segment(*"BC")]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"DG"), Segment(*"AG")),
        Ratio(Segment(*"EG"), Segment(*"BG"))
    ]) in new_facts


def test_eqline_to_centri_G():
    """Test eqline_to_centri"""
    diagram = Diagram()
    diagram.point_dict['A'] = Point(-2, 0)
    diagram.point_dict['C'] = Point(2, 0)
    diagram.point_dict['B'] = Point(0, 2)
    diagram.point_dict['F'] = Point(-1, 1)
    diagram.point_dict['D'] = Point(1, 1)
    diagram.point_dict['G'] = Point(0, 1)
    diagram.point_dict['E'] = Point(0, 0)
    for p1, p2 in combinations("ABCDEFG", 2):
        s = Segment(p1, p2)
        diagram.database.add_fact(Fact("eqline", [s, s]))
    diagram.database.add_fact(Fact("midp", ["F", Segment(*"AB")]))
    diagram.database.add_fact(Fact("midp", ["E", Segment(*"AC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"GC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"FG"), Segment(*"FC")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"EG"), Segment(*"EB")]))
    f = Fact("eqline", [Segment(*"EG"), Segment(*"GB")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_midp_to_centri(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment(*"FG"), Segment(*"GC")),
        Ratio(Segment(*"EG"), Segment(*"GB"))
    ])
    f = Fact("eqline", [Segment(*"AG"), Segment(*"GD")])
    diagram.database.add_fact(Fact("eqline", [Segment(*"BD"), Segment(*"CD")]))
    f = Fact("eqline", [Segment(*"AG"), Segment(*"GD")])
    diagram.database.add_fact(f)
    new_facts = eqline_to_centri(diagram, f)
    assert len(new_facts) == 2
    assert Fact("midp", ["D", Segment(*"BC")]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment(*"DG"), Segment(*"AG")),
        Ratio(Segment(*"EG"), Segment(*"BG"))
    ]) in new_facts
