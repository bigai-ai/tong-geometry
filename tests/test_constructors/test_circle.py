r"""Test the circle constructor module. The numbers are hand-calculated so don't change."""

from matplotlib import pyplot as plt

from tonggeometry.action import Action
from tonggeometry.constructor import (BaseAcuteTriangle, CenterCircle,
                                      CircumscribedCircle, ExCircle, InCircle)
from tonggeometry.diagram import Diagram


def test_circumscribed_circle():
    """Test the circumscribed circle constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(CircumscribedCircle.valid_actions(d, "ABC")) == 1
    d = d.apply_action(Action(CircumscribedCircle, "ABC", "D"))
    d.draw("tests/figures/circumscribed_circle.pdf")
    assert len(d.all_names) == 48
    assert len(d.point_dict) == 4
    assert len(CircumscribedCircle.valid_actions(d, "D")) == 3
    plt.close()


def test_incircle():
    """Test the incircle constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(InCircle.valid_actions(d, "ABC")) == 1
    d = d.apply_action(Action(InCircle, "ABC", "DEFG"))
    d.draw("tests/figures/incircle.pdf")
    assert len(d.all_names) == 45
    assert len(d.point_dict) == 7
    assert len(InCircle.valid_actions(d, "DEFG")) == 31
    plt.close()


def test_excircle():
    """Test the excircle constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(ExCircle.valid_actions(d, "ABC")) == 3
    d = d.apply_action(Action(ExCircle, "ABC", "DEFG"))
    d.draw("tests/figures/excircle.pdf")
    assert len(d.all_names) == 45
    assert len(d.point_dict) == 7
    assert len(ExCircle.valid_actions(d, "DEFG")) == 93
    plt.close()


def test_center_circle():
    """Test the center circle constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(CenterCircle.valid_actions(d, "ABC")) == 6
    d = d.apply_action(Action(CenterCircle, "AB", ""))
    d.draw("tests/figures/center_circle.pdf")
    assert len(d.all_names) == 49
    assert len(d.point_dict) == 3
    plt.close()
