r"""Test the point constructor module. The numbers are hand-calculated so don't change."""

from matplotlib import pyplot as plt

from tonggeometry.action import Action
from tonggeometry.constructor import (
    AnyArc, AnyPoint, BaseAcuteTriangle, BaseInscribedTri, CenterCircle,
    ExCircle, ExSimiliCenter, ExtendEqual, InSimiliCenter,
    IntersectCircleCircle, IntersectLineCircleOff, IntersectLineCircleOn,
    IntersectLineLine, MidArc, MidPoint, Reflect)
from tonggeometry.diagram import Diagram


def test_extend_equal():
    """Test the extend equal constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    assert len(ExtendEqual.valid_actions(d, "ABCD")) == 12
    d = d.apply_action(Action(ExtendEqual, "AD", "E"))
    d.draw("tests/figures/extend_equal.pdf")
    assert len(d.all_names) == 47
    assert len(d.point_dict) == 5
    assert len(ExtendEqual.valid_actions(d, "E")) == 8
    plt.close()


def test_midpoint():
    """Test the midpoint constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(MidPoint.valid_actions(d, "ABC")) == 3
    d = d.apply_action(Action(MidPoint, "AB", "D"))
    d.draw("tests/figures/midpoint.pdf")
    assert len(d.all_names) == 48
    assert len(d.point_dict) == 4
    assert len(MidPoint.valid_actions(d, "D")) == 3
    plt.close()


def test_anypoint():
    """Test the anypoint constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(AnyPoint.valid_actions(d, "ABC")) == 3
    d = d.apply_action(Action(AnyPoint, "AB", "D"), verbose=True)
    d.draw("tests/figures/anypoint.pdf")
    assert len(d.all_names) == 48
    assert len(d.point_dict) == 4
    assert len(AnyPoint.valid_actions(d, "D")) == 3
    plt.close()


def test_midarc():
    """Test the midarc constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    assert len(MidArc.valid_actions(d, "ABCD")) == 6
    d = d.apply_action(Action(MidArc, "ABD", "E"))
    d.draw("tests/figures/midarc.pdf")
    assert len(d.all_names) == 47
    assert len(d.point_dict) == 5
    assert len(MidArc.valid_actions(d, "E")) == 6
    plt.close()


def test_anyarc():
    """Test the anyarc constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    assert len(AnyArc.valid_actions(d, "ABCD")) == 6
    d = d.apply_action(Action(AnyArc, "ABD", "E"))
    d = d.apply_action(Action(AnyArc, "BAD", "F"))
    d.draw("tests/figures/anyarc.pdf")
    assert len(d.all_names) == 46
    assert len(d.point_dict) == 6
    assert len(AnyArc.valid_actions(d, "F")) == 8
    plt.close()


def test_intersect_line_line():
    """Test the intersection of lines constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    assert len(IntersectLineLine.valid_actions(d, "ABCD")) == 3
    d = d.apply_action(Action(IntersectLineLine, "CDAB", "E"))
    d.draw("tests/figures/intersect_line_line.pdf")
    assert len(d.all_names) == 47
    assert len(d.point_dict) == 5
    assert len(IntersectLineLine.valid_actions(d, "E")) == 0
    plt.close()


def test_intersect_line_circle_off():
    """Test the intersection of a line and a circle constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    d = d.apply_action(Action(ExtendEqual, "AB", "E"))
    assert len(IntersectLineCircleOff.valid_actions(d, "E")) == 1
    d = d.apply_action(Action(IntersectLineCircleOff, "EDDA", "FG"))
    d.draw("tests/figures/intersect_line_circle_off.pdf")
    assert len(d.all_names) == 45
    assert len(d.point_dict) == 7
    assert len(IntersectLineCircleOff.valid_actions(d, "FG")) == 0
    plt.close()


def test_intersect_line_circle_on():
    """Test the intersection of a line and a circle constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    assert len(IntersectLineCircleOn.valid_actions(d, "ABCD")) == 3
    d = d.apply_action(Action(IntersectLineCircleOn, "DAD", "E"))
    d.draw("tests/figures/intersect_line_circle_on.pdf")
    assert len(d.all_names) == 47
    assert len(d.point_dict) == 5
    assert len(IntersectLineCircleOn.valid_actions(d, "E")) == 0
    plt.close()


def test_intersect_circle_circle():
    """Test the intersection of circles constructor"""
    d = Diagram().apply_actions([
        Action(BaseInscribedTri, "", "ABCD"),
        Action(ExCircle, "ABC", "EFGH")
    ])
    assert len(IntersectCircleCircle.valid_actions(d, "EFGH")) == 1
    d = d.apply_action(Action(IntersectCircleCircle, "DAEF", "IJ"))
    d.draw("tests/figures/intersect_circle_circle.pdf")
    assert len(d.all_names) == 42
    assert len(d.point_dict) == 10
    assert len(IntersectCircleCircle.valid_actions(d, "IJ")) == 0
    plt.close()


def test_reflect():
    """Test the reflection constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    assert len(Reflect.valid_actions(d, "ABCD")) == 12
    d = d.apply_action(Action(Reflect, "ABD", "EF"))
    d.draw("tests/figures/reflect.pdf")
    assert len(d.all_names) == 46
    assert len(d.point_dict) == 6
    assert len(Reflect.valid_actions(d, "EF")) == 30
    plt.close()


def test_in_simili_center():
    """Test the insimilicenter of circles"""
    d = Diagram().apply_actions([
        Action(BaseAcuteTriangle, "", "CBA"),
        Action(MidPoint, "CB", "D"),
        Action(AnyPoint, "BD", "E"),
        Action(AnyPoint, "CD", "F"),
        Action(CenterCircle, "BE"),
        Action(CenterCircle, "CF"),
    ])
    assert len(
        InSimiliCenter.valid_actions(d, "CF", pick_rep=False,
                                     from_circle=True)) == 1
    d = d.apply_action(Action(InSimiliCenter, "BECF", "G"))
    d.draw("tests/figures/in_simili_center.pdf")
    assert len(d.all_names) == 45
    assert len(d.point_dict) == 7
    assert len(InSimiliCenter.valid_actions(d, "G")) == 0
    plt.close()


def test_ex_simili_center():
    """Test the exsimilicenter of circles"""
    d = Diagram().apply_actions([
        Action(BaseAcuteTriangle, "", "CBA"),
        Action(AnyPoint, "CB", "D"),
        Action(CenterCircle, "BD"),
        Action(CenterCircle, "CD"),
    ])
    assert len(
        ExSimiliCenter.valid_actions(d, "CD", pick_rep=False,
                                     from_circle=True)) == 1
    d = d.apply_action(Action(ExSimiliCenter, "BDCD", "E"))
    d.draw("tests/figures/ex_simili_center.pdf")
    assert len(d.all_names) == 47
    assert len(d.point_dict) == 5
    assert len(InSimiliCenter.valid_actions(d, "E")) == 0
    plt.close()
