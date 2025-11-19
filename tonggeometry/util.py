r"""Utility functions."""

import sys
from collections import Counter, OrderedDict

OrderedSet = dict if sys.version_info >= (3, 7) else OrderedDict


class BufferNode:
    """Buffer node for diversity"""

    def __init__(self):
        self.next = {}
        self.count = 0
        self.min = 100
        self.len_counts = Counter()

    def iou(self, path):
        """Compute the maximum iou of the path in the tree"""
        node = self
        common = 1
        path_len = len(path) + 1
        iou = common / (path_len + node.min - common)
        for edge in path:
            if edge not in node.next:
                break
            common += 1
            node = node.next[edge]
            iou = max(iou, common / (path_len + node.min - common))
        return iou

    def insert(self, path):
        """Insert the path into the tree"""
        node = self
        path_len = len(path) + 1
        node.min = min(node.min, path_len)
        node.count += 1
        node.len_counts[path_len] += 1
        for edge in path:
            if edge not in node.next:
                node.next[edge] = BufferNode()
            node = node.next[edge]
            node.min = min(node.min, path_len)
            node.count += 1
            node.len_counts[path_len] += 1

    def delete(self, path):
        """Delete the path from the tree"""
        node = self
        path_len = len(path) + 1
        node.count -= 1
        node.len_counts[path_len] -= 1
        if node.len_counts[path_len] == 0:
            node.len_counts.pop(path_len)
            if path_len == node.min:
                node.min = min(node.len_counts.keys()) if len(
                    node.len_counts) > 0 else 100
        for edge in path:
            node.next[edge].count -= 1
            if node.next[edge].count == 0:
                node.next.pop(edge)
                break
            node = node.next[edge]
            node.len_counts[path_len] -= 1
            if node.len_counts[path_len] == 0:
                node.len_counts.pop(path_len)
                if path_len == node.min:
                    node.min = min(node.len_counts.keys())


def unique(iterable):
    """Unique list from input."""
    seen = set()
    out = []
    for item in iterable:
        key = str(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def isclose(a: float, b: float, eps=1e-6) -> bool:
    """Check if two floats are close within tolerance."""
    return abs(a - b) < eps


def bron_kerbosch_v1(R: set, P: set, X: set, graph: dict):
    """Bron-Kerbosch Algorithm for Maximal Cliques Detection, v1
    https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    https://www.geeksforgeeks.org/maximal-clique-problem-recursive-solution
    graph is an undirected graph implemented in a directed way.
    """
    if not P and not X:
        yield R
    while P:
        v = P.pop()
        yield from bron_kerbosch(R.union({v}), P.intersection(graph[v]),
                                 X.intersection(graph[v]), graph)
        X.add(v)


def bron_kerbosch(R: set, P: set, X: set, graph: dict):
    """Bron-Kerbosch Algorithm for Maximal Cliques Detection, v2
    https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    https://github.com/alanmc-zz/python-bors-kerbosch
    graph is an undirected graph implemented in a directed way.
    """
    if not P and not X:
        yield R
    if P:
        (_, pivot) = max((len(graph[v]), v) for v in P.union(X))

        for v in P.difference(graph[pivot]):
            yield from bron_kerbosch(R.union({v}), P.intersection(graph[v]),
                                     X.intersection(graph[v]), graph)
            P.remove(v)
            X.add(v)
