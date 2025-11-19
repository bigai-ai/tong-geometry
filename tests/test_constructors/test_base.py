r"""Test the base constructor module. The numbers are hand-calculated so don't change."""

from matplotlib import pyplot as plt

from tonggeometry.action import Action
from tonggeometry.constructor import (BaseAcuteTriangle, BaseHarmonicQuad,
                                      BaseInscribedQuad, BaseInscribedTri,
                                      BaseParallelogram)
from tonggeometry.diagram import Diagram


def test_acute_triangle():
    """Test the acute triangle constructor"""
    d = Diagram().apply_actions([Action(BaseAcuteTriangle, "", "ABC")])
    d.draw("tests/figures/base_acute_triangle.pdf")
    assert len(d.all_names) == 49
    assert len(d.point_dict) == 3
    plt.close()


def test_parallelogram():
    """Test the parallelogram constructor"""
    d = Diagram().apply_actions([Action(BaseParallelogram, "", "ABCD")])
    d.draw("tests/figures/base_parallelogram.pdf")
    assert len(d.all_names) == 48
    assert len(d.point_dict) == 4
    plt.close()


def test_inscribed_quad():
    """Test the inscribed quad constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedQuad, "", "ABCDE")])
    d.draw("tests/figures/base_inscribed_quad.pdf")
    assert len(d.all_names) == 47
    assert len(d.point_dict) == 5
    plt.close()


def test_inscribed_tri():
    """Test the inscribed tri constructor"""
    d = Diagram().apply_actions([Action(BaseInscribedTri, "", "ABCD")])
    d.draw("tests/figures/base_inscribed_tri.pdf")
    assert len(d.all_names) == 48
    assert len(d.point_dict) == 4
    plt.close()


def test_harmonic_quad():
    """Test the harmonic quad constructor"""
    d = Diagram().apply_actions([Action(BaseHarmonicQuad, "", "ABCDE")])
    d.draw("tests/figures/base_harmonic_quad.pdf")
    assert len(d.all_names) == 47
    assert len(d.point_dict) == 5
    plt.close()
