r"""Init the contructor module"""

from .base import *
from .circle import *
from .line import *
from .parallelogram import *
from .point import *
from .triangle import *

AllConstructors = [
    AnyPoint,  # C_n^2
    MidPoint,  # C_n^2
    ExtendEqual,  # A_n^2
    CenterCircle,  # A_n^2
    # BisectorLine,  # C_n^3 * 3
    PerpendicularLine,  # C_n^3 * 3
    InCenter,  # C_n^3
    CircumscribedCircle,  # C_n^3
    AnyArc,  # A_n^3 but much less
    MidArc,  # A_n^3 but much less
    Perpendicular,  # A_n^3
    Parallel,  # A_n^4
    IntersectLineLine,  # C_n^4 * 3
    IntersectLineCircleOn,  # A_n^3 + C_n^2 * 2 but much less
    IntersectLineCircleOff,  # C_n^4 * 3 * 4 but much less
    IntersectCircleCircle,  # C_n^4 * 3 * 4 but much less
    IsogonalConjugate,  # C_n^3 * (n - 3)
    # InCircle,  # C_n^3
    # ExCircle,  # C_n^3 * 3
    # Centroid,  # C_n^3
    # Orthocenter,  # C_n^3
    # Parallelogram,  # C_n^3 * 3
    # Reflect,  # C_n^2 * (n - 2)
]

ConstructorIndex = {
    constructor.__name__: idx
    for idx, constructor in enumerate(AllConstructors)
}
