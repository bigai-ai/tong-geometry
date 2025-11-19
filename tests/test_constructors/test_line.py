r"""Test the line constructor module. The numbers are hand-calculated so don't change."""

from matplotlib import pyplot as plt

from tonggeometry.action import Action
from tonggeometry.constructor import BaseInscribedTri, Parallel, Perpendicular
from tonggeometry.diagram import Diagram


def test_perpendicular_intersect():
    """Test the perpendicular intersect constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    assert len(Perpendicular.valid_actions(d, "ABCD")) == 24
    d = d.apply_action(Action(Perpendicular, "BAC", "E"))
    d.draw("tests/figures/perpendicular_intersect.pdf")
    assert len(d.all_names) == 47
    assert len(d.point_dict) == 5
    assert len(Perpendicular.valid_actions(d, "E")) == 21
    plt.close()


def test_parallel_line():
    """Test the parallel intersect constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    assert len(Parallel.valid_actions(d, "ABCD")) == 24
    d = d.apply_action(Action(Parallel, "ABCD", "E"))
    d.draw("tests/figures/parallel_intersect.pdf")
    assert len(d.all_names) == 47
    assert len(d.point_dict) == 5
    assert len(Parallel.valid_actions(d, "E")) == 41
    plt.close()


# def test_connect():
#     """Test the connect constructor"""
#     d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
#     assert len(Connect.valid_actions(d, "ABCD")) == 6
#     d = d.apply_action(Action(Connect, "AD", ""))
#     d.draw("tests/figures/connect.pdf")
#     assert len(d.all_names) == 48
#     assert len(d.point_dict) == 4
#     assert len(Connect.valid_actions(d, "")) == 0
#     assert len(Connect.valid_actions(d, "D")) == 3
#     plt.close()
