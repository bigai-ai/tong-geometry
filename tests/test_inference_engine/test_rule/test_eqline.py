r"""Test the eqline module in forward_chainer."""

from itertools import combinations

from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Angle, Ratio, Segment
from tonggeometry.inference_engine.rule.eqline import (
    eqangle_and_harmonic_to_perp, eqline_and_desagues_to_eqline,
    eqline_and_eqline_to_x, eqline_and_eqratio_to_harmonic,
    eqline_and_pappus_to_eqline, eqline_to_cevian_middle,
    eqline_to_cevian_side, eqline_to_eqangle, eqline_to_eqline,
    eqratio_and_eqline_to_harmonic, perp_and_harmonic_to_eqangle)


def test_eqline_to_eqline():
    """Test eqline to eqline"""
    diagram = Diagram()
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    f = Fact("eqline", [Segment("A", "B"), Segment("C", "D")])
    diagram.database.add_fact(f)
    new_facts = eqline_to_eqline(diagram, f)
    assert len(new_facts) == 14
    segs = combinations("ABCD", 2)
    for s1, s2 in combinations(segs, 2):
        s1 = Segment(*s1)
        s2 = Segment(*s2)
        fact = Fact('eqline', [s1, s2])
        if fact != f:
            assert Fact('eqline', [s1, s2]) in new_facts


def test_eqline_to_eqangle():
    """Test eqline_to_eqangles to eqline"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(0, 1)
    diagram.point_dict["C"] = Point(2, 0)
    diagram.point_dict["D"] = Point(0, 2)
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    f = Fact("eqline", [Segment("A", "B"), Segment("B", "D")])
    diagram.database.add_fact(f)
    new_facts = eqline_to_eqangle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"CBA"), Angle(*"CBD")])


def test_eqline_and_eqline_to_x():
    """Test eqline_and_eqline_to_x"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 1)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(1, -1)
    diagram.point_dict["D"] = Point(1, 1)
    diagram.point_dict["E"] = Point(2, 2)
    for p1, p2 in combinations("ABCDE", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "B"), Segment("C", "A")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "B"), Segment("C", "A")]))
    f = Fact("eqline", [Segment("E", "B"), Segment("B", "D")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqline_to_x(diagram, f)
    assert len(new_facts) == 0
    assert diagram.database.inverse_x == {"AECD": "B", "ADCE": "B"}
    assert diagram.database.x == {
        Segment(*"AD"): {
            "AECDB": None
        },
        Segment(*"AE"): {
            "ADCEB": None
        },
        Segment(*"CE"): {
            "AECDB": None
        },
        Segment(*"CD"): {
            "ADCEB": None
        }
    }


def test_x_and_x_to_pappus():
    """Test x_and_x_to_pappus"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 1)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(1, -1)
    diagram.point_dict["D"] = Point(1, 1)
    diagram.point_dict["E"] = Point(-1, -1)
    diagram.point_dict["F"] = Point(2, 0)
    diagram.point_dict["G"] = Point(3, 1)
    diagram.point_dict["H"] = Point(3, -1)
    diagram.point_dict["I"] = Point(1, 0)
    for p1, p2 in combinations("ABCDEFGHI", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "B"), Segment("C", "A")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "B"), Segment("C", "A")]))
    f = Fact("eqline", [Segment("E", "B"), Segment("B", "D")])
    diagram.database.add_fact(f)
    eqline_and_eqline_to_x(diagram, f)
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "F"), Segment("F", "H")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "H"), Segment("F", "H")]))
    f = Fact("eqline", [Segment("G", "F"), Segment("F", "C")])
    diagram.database.add_fact(f)
    eqline_and_eqline_to_x(diagram, f)
    assert diagram.database.inverse_x == {
        'AECD': 'B',
        'ADCE': 'B',
        'CHGD': 'F',
        'CDGH': 'F'
    }
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("G", "F"), Segment("G", "C")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "D"), Segment("D", "G")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "D"), Segment("A", "G")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("E", "C"), Segment("C", "H")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("E", "C"), Segment("E", "H")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "I"), Segment("I", "H")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "I"), Segment("A", "H")]))
    f = Fact("eqline", [Segment(*"EI"), Segment(*"GI")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqline_to_x(diagram, f)
    assert len(new_facts) == 2
    assert Fact("eqline", [Segment(*"BF"), Segment(*"BI")]) in new_facts
    assert Fact("eqline", [Segment(*"BF"), Segment(*"IF")]) in new_facts


def test_eqline_and_pappus_to_eqline():
    """Test eqline_and_pappus_to_eqline"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(-1, 1)
    diagram.point_dict["B"] = Point(0, 0)
    diagram.point_dict["C"] = Point(1, -1)
    diagram.point_dict["D"] = Point(1, 1)
    diagram.point_dict["E"] = Point(-1, -1)
    diagram.point_dict["F"] = Point(2, 0)
    diagram.point_dict["G"] = Point(3, 1)
    diagram.point_dict["H"] = Point(3, -1)
    diagram.point_dict["I"] = Point(1, 0)
    for p1, p2 in combinations("ABCDEFGHI", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "B"), Segment("C", "A")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "B"), Segment("C", "A")]))
    f = Fact("eqline", [Segment("E", "B"), Segment("B", "D")])
    diagram.database.add_fact(f)
    eqline_and_eqline_to_x(diagram, f)
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "F"), Segment("F", "H")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "H"), Segment("F", "H")]))
    f = Fact("eqline", [Segment("G", "F"), Segment("F", "C")])
    diagram.database.add_fact(f)
    eqline_and_eqline_to_x(diagram, f)
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("G", "F"), Segment("G", "C")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "D"), Segment("D", "G")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "D"), Segment("A", "G")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "I"), Segment("I", "H")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "I"), Segment("A", "H")]))
    f = Fact("eqline", [Segment(*"EI"), Segment(*"GI")])
    diagram.database.add_fact(f)
    eqline_and_eqline_to_x(diagram, f)
    f = Fact("eqline", [Segment(*"EC"), Segment(*"EH")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_pappus_to_eqline(diagram, f)
    assert len(new_facts) == 2
    assert Fact("eqline", [Segment(*"BF"), Segment(*"BI")]) in new_facts
    assert Fact("eqline", [Segment(*"BF"), Segment(*"IF")]) in new_facts


def test_x_and_x_and_x_to_desargues():
    """Test x_and_x_and_x_to_desargues"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 1)
    diagram.point_dict["B"] = Point(1, 0)
    diagram.point_dict["C"] = Point(0, -1)
    diagram.point_dict["D"] = Point(2, 0.5)
    diagram.point_dict["E"] = Point(3, 0)
    diagram.point_dict["F"] = Point(1.5, -0.5)
    diagram.point_dict["G"] = Point(3, 0)
    diagram.point_dict["H"] = Point(3, -1)
    diagram.point_dict["I"] = Point(2, -1)
    diagram.point_dict["J"] = Point(1, -1)
    for p1, p2 in combinations("ABCDEFGHIJ", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    # ADG
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "G"), Segment("D", "G")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "D"), Segment("A", "G")]))
    # BEG
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "G"), Segment("E", "G")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "E"), Segment("B", "G")]))
    # # CFG
    # diagram.database.add_fact(
    #     Fact("eqline",
    #          [Segment("C", "G"), Segment("F", "G")]))
    # diagram.database.add_fact(
    #     Fact("eqline",
    #          [Segment("C", "F"), Segment("C", "G")]))
    # ABH
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "H"), Segment("B", "H")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "B"), Segment("A", "H")]))
    # DEH
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "H"), Segment("E", "H")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "E"), Segment("D", "H")]))
    # ACI
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "I"), Segment("C", "I")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "C"), Segment("A", "I")]))
    # DFI
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "I"), Segment("F", "I")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "F"), Segment("D", "I")]))
    # BCJ
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "J"), Segment("C", "J")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "C"), Segment("B", "J")]))
    # # EFJ
    # diagram.database.add_fact(
    #     Fact("eqline",
    #          [Segment("E", "J"), Segment("F", "J")]))
    # diagram.database.add_fact(
    #     Fact("eqline",
    #          [Segment("E", "F"), Segment("E", "J")]))
    diagram.database.add_fact(Fact("eqline", [Segment(*"JH"), Segment(*"JI")]))
    f = Fact("eqline", [Segment(*"EJ"), Segment(*"FJ")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqline_to_x(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqline", [Segment(*"CG"), Segment(*"FG")])


def test_eqline_and_desagues_to_eqline():
    """Test eqline_and_desagues_to_eqline"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 1)
    diagram.point_dict["B"] = Point(1, 0)
    diagram.point_dict["C"] = Point(0, -1)
    diagram.point_dict["D"] = Point(2, 0.5)
    diagram.point_dict["E"] = Point(3, 0)
    diagram.point_dict["F"] = Point(1.5, -0.5)
    diagram.point_dict["G"] = Point(3, 0)
    diagram.point_dict["H"] = Point(3, -1)
    diagram.point_dict["I"] = Point(2, -1)
    diagram.point_dict["J"] = Point(1, -1)
    for p1, p2 in combinations("ABCDEFGHIJ", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    # ADG
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "G"), Segment("D", "G")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "D"), Segment("A", "G")]))
    # BEG
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "G"), Segment("E", "G")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "E"), Segment("B", "G")]))
    # # CFG
    # diagram.database.add_fact(
    #     Fact("eqline",
    #          [Segment("C", "G"), Segment("F", "G")]))
    # diagram.database.add_fact(
    #     Fact("eqline",
    #          [Segment("C", "F"), Segment("C", "G")]))
    # ABH
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "H"), Segment("B", "H")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "B"), Segment("A", "H")]))
    # DEH
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "H"), Segment("E", "H")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "E"), Segment("D", "H")]))
    # ACI
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "I"), Segment("C", "I")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "C"), Segment("A", "I")]))
    # DFI
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "I"), Segment("F", "I")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "F"), Segment("D", "I")]))
    # BCJ
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "J"), Segment("C", "J")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "C"), Segment("B", "J")]))
    # # EFJ
    # diagram.database.add_fact(
    #     Fact("eqline",
    #          [Segment("E", "J"), Segment("F", "J")]))
    # diagram.database.add_fact(
    #     Fact("eqline",
    #          [Segment("E", "F"), Segment("E", "J")]))
    f = Fact("eqline", [Segment(*"EJ"), Segment(*"FJ")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqline_to_x(diagram, f)
    assert len(new_facts) == 0
    f = Fact("eqline", [Segment(*"JH"), Segment(*"JI")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_desagues_to_eqline(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqline", [Segment(*"CG"), Segment(*"FG")])


def test_eqratio_and_eqline_to_harmonic():
    """Test eqratio_and_eqline_to_harmonic"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(2, 0)
    diagram.point_dict["C"] = Point(3, 0)
    diagram.point_dict["D"] = Point(6, 0)
    diagram.point_dict["P"] = Point(3, 3)
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "C"), Segment("B", "D")]))
    diagram.database.add_fact(Fact("eqangle", [Angle(*"BPC"), Angle(*"CPD")]))
    f = Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("B", "C")),
        Ratio(Segment("A", "D"), Segment("D", "C"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_eqline_to_harmonic(diagram, f)
    assert len(diagram.database.harmonic) == 1
    assert "ACBD" in diagram.database.harmonic
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("perp", [Angle(*"APC")])


def test_eqline_and_eqratio_to_harmonic():
    """Test eqline_and_eqratio_to_harmonic"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(2, 0)
    diagram.point_dict["C"] = Point(3, 0)
    diagram.point_dict["D"] = Point(6, 0)
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment("A", "B"), Segment("B", "C")),
            Ratio(Segment("A", "D"), Segment("D", "C"))
        ]))
    diagram.database.add_fact(Fact("perp", [Angle(*"BPD")]))
    f = Fact("eqline", [Segment("A", "C"), Segment("B", "D")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqratio_to_harmonic(diagram, f)
    assert len(diagram.database.harmonic) == 1
    assert "ACBD" in diagram.database.harmonic
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"APB"), Angle(*"BPC")])


# def test_eqratio_and_eqcircle_to_harmonic():
#     """Test eqratio_and_eqcircle_to_harmonic"""
#     diagram = Diagram()
#     diagram.point_dict["A"] = Point(0, 0)
#     diagram.point_dict["B"] = Point(2, 1)
#     diagram.point_dict["C"] = Point(3, 0)
#     diagram.point_dict["D"] = Point(6, -2)
#     for p1, p2 in combinations("ABCD", 2):
#         diagram.database.add_fact(
#             Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
#     diagram.database.add_fact(Fact("perp", [Angle(*"BPD")]))
#     diagram.database.add_fact(
#         Fact("eqcircle", [Circle(None, [*"ABC"]),
#                           Circle(None, [*"BCD"])]))
#     f = Fact("eqratio", [
#         Ratio(Segment("A", "B"), Segment("B", "C")),
#         Ratio(Segment("A", "D"), Segment("D", "C"))
#     ])
#     diagram.database.add_fact(f)
#     new_facts = eqratio_and_eqcircle_to_harmonic(diagram, f)
#     assert len(diagram.database.harmonic) == 1
#     assert "ACBD" in diagram.database.harmonic
#     assert len(new_facts) == 1
#     assert new_facts[0] == Fact("eqangle", [Angle(*"APB"), Angle(*"BPC")])

# def test_eqcircle_and_eqratio_to_harmonic():
#     """Test eqcircle_and_eqratio_to_harmonic"""
#     diagram = Diagram()
#     diagram.point_dict["A"] = Point(0, 0)
#     diagram.point_dict["B"] = Point(2, 0.1)
#     diagram.point_dict["C"] = Point(3, 0)
#     diagram.point_dict["D"] = Point(6, -100)
#     diagram.point_dict["P"] = Point(3, 3)
#     for p1, p2 in combinations("ABCD", 2):
#         diagram.database.add_fact(
#             Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
#     diagram.database.add_fact(
#         Fact("eqratio", [
#             Ratio(Segment("A", "B"), Segment("B", "C")),
#             Ratio(Segment("A", "D"), Segment("D", "C"))
#         ]))
#     diagram.database.add_fact(Fact("eqangle", [Angle(*"APB"), Angle(*"BPC")]))
#     f = Fact("eqcircle", [Circle(None, [*"ABC"]), Circle(None, [*"BCD"])])
#     diagram.database.add_fact(f)
#     new_facts = eqcircle_and_eqratio_to_harmonic(diagram, f)
#     assert len(diagram.database.harmonic) == 1
#     assert "ACBD" in diagram.database.harmonic
#     assert len(new_facts) == 1
#     assert new_facts[0] == Fact("perp", [Angle(*"BPD")])


def test_perp_and_harmonic_to_eqangle():
    """Test perp_and_harmonic_to_eqangle"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(2, 0.1)
    diagram.point_dict["C"] = Point(3, 0)
    diagram.point_dict["D"] = Point(6, -100)
    for p1, p2 in combinations("ABCD", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment("A", "B"), Segment("B", "C")),
            Ratio(Segment("A", "D"), Segment("D", "C"))
        ]))
    f = Fact("eqline", [Segment("A", "C"), Segment("B", "D")])
    diagram.database.add_fact(f)
    eqline_and_eqratio_to_harmonic(diagram, f)
    f = Fact("perp", [Angle(*"APC")])
    diagram.database.add_fact(f)
    new_facts = perp_and_harmonic_to_eqangle(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqangle", [Angle(*"BPA"), Angle(*"APD")])


def test_eqangle_and_harmonic_to_perp():
    """Test eqangle_and_harmonic_to_perp"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(2, 0.1)
    diagram.point_dict["C"] = Point(3, 0)
    diagram.point_dict["D"] = Point(6, -100)
    diagram.point_dict["P"] = Point(3, 3)
    for p1, p2 in combinations("ABCDP", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment("A", "B"), Segment("B", "C")),
            Ratio(Segment("A", "D"), Segment("D", "C"))
        ]))
    f = Fact("eqline", [Segment("A", "C"), Segment("B", "D")])
    diagram.database.add_fact(f)
    eqline_and_eqratio_to_harmonic(diagram, f)
    f = Fact("eqangle", [Angle(*"APD"), Angle(*"DPC")])
    diagram.database.add_fact(f)
    new_facts = eqangle_and_harmonic_to_perp(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("perp", [Angle(*"BPD")])


def test_eqline_to_cevian_middle():
    """Test eqline_to_cevian_middle"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 1)
    diagram.point_dict["B"] = Point(-1, -1)
    diagram.point_dict["C"] = Point(1, -1)
    diagram.point_dict["D"] = Point(0, -1)
    diagram.point_dict["E"] = Point(0.5, 0)
    diagram.point_dict["F"] = Point(-0.5, 0)
    diagram.point_dict["O"] = Point(0, 0)
    for p1, p2 in combinations("ABCDEFOP", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "B"), Segment("B", "F")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "F"), Segment("B", "F")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "C"), Segment("B", "D")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "D"), Segment("C", "D")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "E"), Segment("C", "E")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "E"), Segment("A", "C")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "O"), Segment("O", "E")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "E"), Segment("O", "E")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "O"), Segment("O", "F")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "F"), Segment("C", "O")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("E", "P"), Segment("F", "P")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("E", "F"), Segment("F", "P")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "P"), Segment("C", "P")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "C"), Segment("C", "P")]))
    f = Fact("eqline", [Segment("A", "O"), Segment("O", "D")])
    diagram.database.add_fact(f)
    new_facts = eqline_to_cevian_middle(diagram, f)
    assert len(diagram.database.cevian) == 1
    assert "ABCDEF" in diagram.database.cevian
    assert len(diagram.database.inverse_cevian) == 2
    assert "AEBD" in diagram.database.inverse_cevian
    assert "AFCD" in diagram.database.inverse_cevian
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment(*"BP"), Segment(*"CP")),
        Ratio(Segment(*"BD"), Segment(*"CD"))
    ])


def test_x_to_cevian():
    """Test x_to_cevian"""
    diagram = Diagram()
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 1)
    diagram.point_dict["B"] = Point(-1, -1)
    diagram.point_dict["C"] = Point(1, -1)
    diagram.point_dict["D"] = Point(0, -1)
    diagram.point_dict["E"] = Point(0.5, 0)
    diagram.point_dict["F"] = Point(-0.5, 0)
    diagram.point_dict["O"] = Point(0, 0)
    for p1, p2 in combinations("ABCDEFOP", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "B"), Segment("B", "F")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "F"), Segment("B", "F")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "C"), Segment("B", "D")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "D"), Segment("C", "D")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "E"), Segment("C", "E")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "E"), Segment("A", "C")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "O"), Segment("O", "E")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "E"), Segment("O", "E")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "O"), Segment("O", "F")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "F"), Segment("C", "O")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "D"), Segment("A", "O")]))
    f = Fact("eqline", [Segment("A", "O"), Segment("O", "D")])
    diagram.database.add_fact(f)
    new_facts = eqline_to_cevian_middle(diagram, f)
    assert len(new_facts) == 0
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("E", "P"), Segment("F", "P")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("E", "F"), Segment("F", "P")]))
    f = Fact("eqline", [Segment("B", "P"), Segment("C", "P")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqline_to_x(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment(*"BP"), Segment(*"CP")),
        Ratio(Segment(*"BD"), Segment(*"CD"))
    ])


def test_eqline_to_cevian_side():
    """Test eqline_to_cevian_side"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 1)
    diagram.point_dict["B"] = Point(-1, -1)
    diagram.point_dict["C"] = Point(1, -1)
    diagram.point_dict["D"] = Point(0, -1)
    diagram.point_dict["E"] = Point(0.5, 0)
    diagram.point_dict["F"] = Point(-0.5, 0)
    diagram.point_dict["O"] = Point(0, 0)
    for p1, p2 in combinations("ABCDEFOP", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "B"), Segment("B", "F")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "F"), Segment("B", "F")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "E"), Segment("C", "E")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "E"), Segment("A", "C")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "O"), Segment("O", "E")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "E"), Segment("O", "E")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "O"), Segment("O", "F")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("C", "F"), Segment("C", "O")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "O"), Segment("O", "D")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "D"), Segment("O", "D")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("E", "P"), Segment("F", "P")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("E", "F"), Segment("F", "P")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "P"), Segment("C", "P")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("B", "C"), Segment("C", "P")]))
    f = Fact("eqline", [Segment("B", "D"), Segment("C", "D")])
    diagram.database.add_fact(f)
    new_facts = eqline_to_cevian_side(diagram, f)
    assert len(diagram.database.cevian) == 2
    assert "ABCDEF" in diagram.database.cevian
    assert "BCOFED" in diagram.database.cevian
    assert len(diagram.database.inverse_cevian) == 4
    assert "AEBD" in diagram.database.inverse_cevian
    assert "AFCD" in diagram.database.inverse_cevian
    assert "CDOE" in diagram.database.inverse_cevian
    assert "BDOF" in diagram.database.inverse_cevian
    assert len(new_facts) == 2
    assert new_facts[0] == new_facts[1]
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment(*"BP"), Segment(*"CP")),
        Ratio(Segment(*"BD"), Segment(*"CD"))
    ])
