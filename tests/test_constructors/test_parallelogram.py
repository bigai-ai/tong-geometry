r"""Test the parallelogram constructor module. The numbers are hand-calculated so don't change."""

from matplotlib import pyplot as plt

from tonggeometry.action import Action
from tonggeometry.constructor import BaseAcuteTriangle, Parallelogram
from tonggeometry.diagram import Diagram


def test_parallelogram():
    """Test the parallelogram constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    assert len(Parallelogram.valid_actions(d, "ABC")) == 3
    d = d.apply_action(Action(Parallelogram, "ABC", "D"))
    d.draw("tests/figures/parallelogram.pdf")
    assert len(d.all_names) == 48
    assert len(d.point_dict) == 4
    assert len(Parallelogram.valid_actions(d, "D")) == 9
    plt.close()
