r"""Test the triangle constructor module. The numbers are hand-calculated so don't change."""

from matplotlib import pyplot as plt

from tonggeometry.action import Action
from tonggeometry.constructor import (BaseAcuteTriangle, BaseInscribedTri,
                                      BisectorLine, Centroid, InCenter,
                                      IsogonalConjugate, Orthocenter,
                                      PerpendicularLine)
from tonggeometry.diagram import Diagram


def test_bisector_line():
    """Test the internal angle bisector constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(BisectorLine.valid_actions(d, "ABC")) == 3
    d = d.apply_action(Action(BisectorLine, "ABC", "D"))
    d.draw("tests/figures/bisector_line.pdf")
    assert len(d.all_names) == 48
    assert len(d.point_dict) == 4
    assert len(BisectorLine.valid_actions(d, "D")) == 6
    plt.close()


def test_perpendicular_line():
    """Test the perpendicular line constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(PerpendicularLine.valid_actions(d, "ABC")) == 3
    d = d.apply_action(Action(PerpendicularLine, "ABC", "D"))
    d.draw("tests/figures/perpendicular_line.pdf")
    assert len(d.all_names) == 48
    assert len(d.point_dict) == 4
    print(PerpendicularLine.valid_actions(d, "D"))
    assert len(PerpendicularLine.valid_actions(d, "D")) == 4
    plt.close()


def test_centroid():
    """Test the centroid constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(Centroid.valid_actions(d, "ABC")) == 1
    d = d.apply_action(Action(Centroid, "ABC", "DEFG"))
    d.draw("tests/figures/centroid.pdf")
    assert len(d.all_names) == 45
    assert len(d.point_dict) == 7
    assert len(Centroid.valid_actions(d, "DEFG")) == 28
    plt.close()


def test_orthocenter():
    """Test the orthocenter constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(Orthocenter.valid_actions(d, "ABC")) == 1
    d = d.apply_action(Action(Orthocenter, "ABC", "DEFG"))
    d.draw("tests/figures/orthocenter.pdf")
    assert len(d.all_names) == 45
    assert len(d.point_dict) == 7
    assert len(Orthocenter.valid_actions(d, "DEFG")) == 16
    plt.close()


def test_isogonalconjugate():
    """Test the isogonal conjugate constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    assert len(IsogonalConjugate.valid_actions(d, "ABCD")) == 4
    d = d.apply_action(Action(IsogonalConjugate, "ABCD", "E"))
    d.draw("tests/figures/isogonalconjugate.pdf")
    assert len(d.all_names) == 47
    assert len(d.point_dict) == 5
    assert len(IsogonalConjugate.valid_actions(d, "E")) == 16
    plt.close()


def test_incenter():
    """Test the incenter constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(InCenter.valid_actions(d, "ABC")) == 1
    d = d.apply_action(Action(InCenter, "ABC", "D"))
    d.draw("tests/figures/incenter.pdf")
    assert len(d.all_names) == 48
    assert len(d.point_dict) == 4
    assert len(InCenter.valid_actions(d, "D")) == 3


# def test_circumcenter():
#     """Test the circumcenter constructor"""
#     d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
#     assert len(Circumcenter.valid_actions(d, "ABC")) == 1
#     d = d.apply_action(Action(Circumcenter, "ABC", "D"))
#     d.draw("tests/figures/circumcenter.pdf")
#     assert len(d.all_names) == 48
#     assert len(d.point_dict) == 4
#     assert len(Circumcenter.valid_actions(d, "D")) == 3

# def test_excenter():
#     """Test the excenter constructor"""
#     d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
#     assert len(Excenter.valid_actions(d, "ABC")) == 3
#     d = d.apply_action(Action(Excenter, "ABC", "D"))
#     d.draw("tests/figures/excenter.pdf")
#     assert len(d.all_names) == 48
#     assert len(d.point_dict) == 4
#     assert len(Excenter.valid_actions(d, "D")) == 9

# def test_equal_angle():
#     """Test the equal angle constructor"""
#     d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
#     assert len(EqualAngles.valid_actions(d, "ABC")) == 3
#     d = d.apply_action(Action(EqualAngles, "ABC", "D"))
#     d.draw("tests/figures/equal_angle.pdf")
#     assert len(d.all_names) == 48
#     assert len(d.point_dict) == 4
#     assert len(EqualAngles.valid_actions(d, "D")) == 9
#     plt.close()
