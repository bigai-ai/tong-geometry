r"""Utility functions for forward chainer."""

from typing import TYPE_CHECKING, Hashable, Tuple

from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram


def four_joint(a: Hashable, b: Hashable, c: Hashable,
               d: Hashable) -> Tuple[Hashable, OrderedSet]:
    """Check if two distinctive pairs (ab and cd) have one joint."""
    joint = None
    clean_set = OrderedSet()
    for e in [a, b, c, d]:
        if e not in clean_set:
            clean_set[e] = None
        else:
            joint = e
    return joint, clean_set


def sort_two_from_first(one: str, two: str) -> Tuple[str, str]:
    """Sort the second string in a pair based on the first sorted."""
    sort_index = sorted(list(range(len(one))), key=lambda i: one[i])
    return "".join(one[i] for i in sort_index), "".join(two[i]
                                                        for i in sort_index)
