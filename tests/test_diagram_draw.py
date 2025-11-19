r"""Test the Diagram module. The numbers are hand-calculated so don't change."""

import random

from matplotlib import pyplot as plt

from tonggeometry.action import Action
from tonggeometry.constructor import (
    BaseAcuteTriangle, BaseInscribedQuad, BaseInscribedTri, BisectorLine,
    Centroid, CircumscribedCircle, ExCircle, InCircle, IntersectLineCircleOff,
    IntersectLineCircleOn, IntersectLineLine, MidArc, MidPoint, Orthocenter,
    Parallel, Parallelogram, Perpendicular, PerpendicularLine, Reflect)
from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import (Angle, Circle, Ratio,
                                                      Segment, Triangle)


def test_d_1():
    """Test IMO drawing P1."""
    Diagram().apply_actions([
        Action(BaseAcuteTriangle, "", "CBA"),
        Action(PerpendicularLine, "CBA", "E"),
        Action(PerpendicularLine, "BCA", "F"),
        Action(IntersectLineLine, "BECF", "H"),
        Action(CircumscribedCircle, "ABC", "I"),
        Action(MidPoint, "BC", "M"),
        Action(IntersectLineCircleOn, "MAI", "D"),
        Action(CircumscribedCircle, "EFD", "J"),
        Action(Reflect, "IJD", "aG"),
    ]).draw("tests/figures/IMO_P1.pdf")
    plt.close()


def test_d_2():
    """Test IMO drawing P2."""
    Diagram().apply_actions([
        Action(BaseAcuteTriangle, "", "CBA"),
        Action(InCircle, "ABC", "IXYZ"),
        Action(MidPoint, "BC", "D"),
        Action(CircumscribedCircle, "BCI", "G"),
        Action(MidArc, "BCG", "P"),
        Action(CircumscribedCircle, "ABD", "J"),
        Action(IntersectLineCircleOn, "PBJ", "E"),
        Action(CircumscribedCircle, "ADC", "L"),
        Action(IntersectLineCircleOn, "LCL", "F"),
    ]).draw("tests/figures/IMO_P2.pdf")
    plt.close()


def test_d_3():
    """Test IMO drawing P3."""
    Diagram().apply_actions([
        Action(BaseAcuteTriangle, "", "CBA"),
        Action(CircumscribedCircle, "ABC", "O"),
        Action(MidArc, "BCO", "S"),
        Action(MidArc, "CBO", "N"),
        Action(Orthocenter, "ABC", "HXYW"),
        Action(IntersectLineCircleOn, "HSO", "P"),
        Action(IntersectLineLine, "ACSP", "Z"),
        Action(Parallel, "NAZH", "U"),
        Action(IntersectLineLine, "NUAB", "K"),
    ]).draw("tests/figures/IMO_P3.pdf")
    plt.close()


def test_d_4():
    """Test IMO drawing P4."""
    Diagram().apply_actions([
        Action(BaseInscribedQuad, "", "DABCO"),
        Action(Centroid, "ABD", "ZEQP"),
        Action(MidPoint, "CD", "F"),
        Action(Parallelogram, "EOF", "M")
    ]).draw("tests/figures/IMO_P4.pdf")
    plt.close()


def test_d_5():
    """Test IMO drawing P5."""
    Diagram().apply_actions([
        Action(BaseInscribedTri, "", "CABO"),
        Action(Centroid, "ABC", "GFEZ"),
        Action(PerpendicularLine, "BAC", "D"),
        Action(IntersectLineCircleOff, "GDOA", "KS")
    ]).draw("tests/figures/IMO_P5.pdf")
    plt.close()


def test_d_6():
    """Test IMO drawing P6."""
    Diagram().apply_actions([
        Action(BaseAcuteTriangle, "", "CAB"),
        Action(PerpendicularLine, "BAC", "D"),
        Action(PerpendicularLine, "ACB", "E"),
        Action(IntersectLineLine, "ADCE", "H"),
        Action(BisectorLine, "AHE", "P"),
        Action(IntersectLineLine, "PHBC", "Q"),
        Action(MidPoint, "AC", "M"),
        Action(BisectorLine, "ABC", "F"),
        Action(IntersectLineLine, "BFHM", "R")
    ]).draw("tests/figures/IMO_P6.pdf")
    plt.close()


def test_d_7():
    """Test IMO drawing P7."""
    Diagram().apply_actions([
        Action(BaseAcuteTriangle, "", "CAB"),
        Action(PerpendicularLine, "BAC", "E"),
        Action(PerpendicularLine, "ACB", "F"),
        Action(CircumscribedCircle, "ABC", "G"),
        Action(CircumscribedCircle, "BEF", "H"),
        Action(Reflect, "GHB", "aD"),
        Action(PerpendicularLine, "AGC", "U"),
        Action(Perpendicular, "AGU", "Z"),
        Action(IntersectLineLine, "ZAEF", "X"),
        Action(IntersectLineLine, "ZCEF", "Y"),
    ]).draw("tests/figures/IMO_P7.pdf")
    plt.close()


def test_d_8():
    """Test IMO drawing P8."""
    Diagram().apply_actions([
        Action(BaseInscribedTri, "", "CABO"),
        Action(MidPoint, "AB", "D"),
        Action(MidPoint, "CA", "E"),
        Action(MidArc, "CBO", "M"),
        Action(MidArc, "BCO", "N"),
        Action(IntersectLineLine, "OEAN", "P"),
    ]).draw("tests/figures/IMO_P8.pdf")
    plt.close()


def test_d_9():
    """Test IMO drawing P9."""
    Diagram().apply_actions([
        Action(BaseInscribedTri, "", "CABO"),
        Action(BisectorLine, "ACB", "D"),
        Action(BisectorLine, "ABC", "E"),
        Action(IntersectLineLine, "BEDC", "I"),
        Action(ExCircle, "BAC", "JUVW"),
        Action(IntersectLineCircleOff, "DEOA", "QP"),
        Action(CircumscribedCircle, "PIQ", "F"),
    ]).draw("tests/figures/IMO_P9.pdf")
    plt.close()


def test_sampling():
    """Test handcrafted sampling on diagram."""
    d = Diagram()
    all_valid_actions = d.new_valid_actions()
    while True:
        if d.is_terminal:
            break
        print(d.depth)
        print(len(all_valid_actions))
        act = all_valid_actions.pop(random.randrange(len(all_valid_actions)))
        print(act)
        node_key = d.key
        d = d.apply_action(act)
        if node_key == "Null()":
            all_valid_actions = d.new_valid_actions()
        else:
            all_valid_actions += d.new_valid_actions()
    d.draw("./tests/figures/sampled.pdf")
    plt.close()


def test_pruning():
    """Test prune action."""
    d = Diagram().apply_actions([
        Action(BaseAcuteTriangle, "", "CBA"),
        Action(PerpendicularLine, "CBA", "E"),
        Action(PerpendicularLine, "BCA", "F"),
        Action(IntersectLineLine, "BECF", "H"),
        Action(CircumscribedCircle, "ABC", "I"),
        Action(MidPoint, "BC", "M"),
        Action(IntersectLineCircleOn, "MAI", "D"),
        Action(CircumscribedCircle, "EFD", "J"),
        Action(Reflect, "IJD", "aG"),
        Action(IntersectLineCircleOff, "MHIA", "LN"),
        Action(PerpendicularLine, "AHM", "O"),
        Action(IntersectLineLine, "OHBC", "K"),
    ])
    d.draw("./tests/figures/prune_orig.pdf")
    acts_orderless = d.prune("DGH")
    Diagram().apply_actions(acts_orderless).draw(
        "./tests/figures/prune_orderless.pdf")
    plt.close()


def test_check_num():
    """Test check_num"""
    d = Diagram()
    f = Fact("eqline", [Segment(*"AB"), Segment(*"CD")])
    d.point_dict = {
        "A": Point(0, 0),
        "B": Point(1, 0),
        "C": Point(3, 0),
        "D": Point(4, 0)
    }
    assert d.check_num(f) is True
    d = Diagram()
    f = Fact("eqcircle", [Circle("A", ["B"]), Circle("A", ["B"])])
    d.point_dict = {
        "A": Point(0, 0),
        "B": Point(1, 0),
        "C": Point(0, 1),
    }
    assert d.check_num(f) is True
    f = Fact("eqcircle", [Circle("A", ["B"]), Circle(None, [*"CDE"])])
    d.point_dict = {
        "A": Point(0, 0),
        "B": Point(1, 0),
        "C": Point(0, 1),
        "D": Point(-1, 0),
        "E": Point(0, -1),
    }
    assert d.check_num(f) is True
    f = Fact("eqcircle", [Circle(None, [*"ABC"]), Circle(None, [*"DEF"])])
    d.point_dict = {
        "A": Point(1, 0),
        "B": Point(0, 1),
        "C": Point(-1, 0),
        "D": Point(-1, 0),
        "E": Point(0, -1),
        "F": Point(0, 1),
    }
    assert d.check_num(f) is True
    f = Fact("perp", [Angle(*"ABC")])
    d.point_dict = {"A": Point(1, 0), "B": Point(0, 0), "C": Point(0, -0.001)}
    assert d.check_num(f) is True
    f = Fact("perp", [Angle(*"ABC")])
    d.point_dict = {"A": Point(1, 0), "B": Point(0, 0), "C": Point(0.001, 0)}
    assert d.check_num(f) is False
    f = Fact("cong", [Segment(*"AB"), Segment(*"CD")])
    d.point_dict = {
        "A": Point(1, 0),
        "B": Point(0, 0),
        "C": Point(0.001, 0),
        "D": Point(1.001, 0)
    }
    assert d.check_num(f) is True
    f = Fact("para", [Segment(*"AB"), Segment(*"CD")])
    ff = Fact("eqline", [Segment(*"AB"), Segment(*"CD")])
    d.point_dict = {
        "A": Point(0, 0),
        "B": Point(1, 0),
        "C": Point(3, 1),
        "D": Point(4, 1)
    }
    assert d.check_num(f) is True and d.check_num(ff) is False
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    d.point_dict = {
        "A": Point(-1, 0),
        "B": Point(0, 0),
        "C": Point(-1, 1),
        "D": Point(1, 0),
        "E": Point(0, 0),
        "F": Point(-1, 1)
    }
    assert d.check_num(f) is True
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    d.point_dict = {
        "A": Point(-1, 0),
        "B": Point(0, 0),
        "C": Point(-1, 1),
        "D": Point(0, 0),
        "E": Point(1, 0),
        "F": Point(0, 1)
    }
    assert d.check_num(f) is True
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    d.point_dict = {
        "A": Point(-1, 0),
        "B": Point(0, 0),
        "C": Point(-1, 1),
        "D": Point(1, 0),
        "E": Point(0, 0),
        "F": Point(1, 1)
    }
    assert d.check_num(f) is False
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    d.point_dict = {
        "A": Point(-1, 0),
        "B": Point(0, 0),
        "C": Point(-1, 1),
        "D": Point(0, 0),
        "E": Point(1, 0),
        "F": Point(0, 1)
    }
    assert d.check_num(f) is True
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    d.point_dict = {
        "A": Point(-1, 0),
        "B": Point(0, 0),
        "C": Point(1, 0),
        "D": Point(1, 0),
        "E": Point(0, 0),
        "F": Point(2, 0)
    }
    assert d.check_num(f) is False
    f = Fact("eqangle", [Angle(*"ABC"), Angle(*"DEF")])
    d.point_dict = {
        "A": Point(-1, 0),
        "B": Point(0, 0),
        "C": Point(1, 0),
        "D": Point(1, 0),
        "E": Point(0, 0),
        "F": Point(-1, 0)
    }
    assert d.check_num(f) is True
    f = Fact("midp", ["A", Segment(*"BC")])
    d.point_dict = {"A": Point(0, 0), "B": Point(-1, 0), "C": Point(1, 0)}
    assert d.check_num(f) is True
    f = Fact("simtri", [Triangle(*"ABC"), Triangle(*"DEF")])
    d.point_dict = {
        "A": Point(0, 0),
        "B": Point(1, 0),
        "C": Point(2, 1),
        "D": Point(0, 0),
        "E": Point(2, 0),
        "F": Point(4, 2)
    }
    assert d.check_num(f) is True
    f = Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])
    d.point_dict = {
        "A": Point(0, 0),
        "B": Point(1, 0),
        "C": Point(2, 1),
        "D": Point(0, 0),
        "E": Point(2, 0),
        "F": Point(4, 2)
    }
    assert d.check_num(f) is False
    f = Fact("contri", [Triangle(*"ABC"), Triangle(*"DEF")])
    ff = Fact("simtri", [Triangle(*"ABC"), Triangle(*"DEF")])
    d.point_dict = {
        "A": Point(0, 0),
        "B": Point(1, 0),
        "C": Point(2, 1),
        "D": Point(2, 0),
        "E": Point(3, 0),
        "F": Point(4, 1)
    }
    assert d.check_num(f) is True and d.check_num(ff) is True
    f = Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"CD")),
        Ratio(Segment(*"EF"), Segment(*"GH"))
    ])
    d.point_dict = {
        "A": Point(0, 0),
        "B": Point(1, 0),
        "C": Point(2, 1),
        "D": Point(2, 0),
        "E": Point(-1, -1),
        "F": Point(-1, -2),
        "G": Point(3, 0),
        "H": Point(4, 0)
    }
    assert d.check_num(f) is True

    f = Fact("eqratio", [
        Ratio(Segment(*"AB"), Segment(*"CD")),
        Ratio(Segment(*"EF"), Segment(*"GH"))
    ])
    d.point_dict = {
        "A": Point(0, 0),
        "B": Point(1, 0),
        "C": Point(2, 1),
        "D": Point(2, 0),
    }
    assert d.check_num(f) is False
