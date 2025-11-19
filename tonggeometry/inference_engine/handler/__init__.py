r"""The handler module. Handler handles predicates. In general, for those that
need looping over, using an ordered set shall be prefered."""

import importlib

from tonggeometry.util import OrderedSet

order = [
    "eqline", "eqcircle", "cong", "midp", "para", "perp", "eqangle", "eqratio",
    "simtri", "contri"
]

ORDER = OrderedSet(zip(order, range(len(order))))

ALL_HANDLERS = {
    name: importlib.import_module(f".{name}", __package__)
    for name in ORDER
}
