r"""eqcircle handler for Database."""

from typing import TYPE_CHECKING, List, Optional, Union

import tonggeometry.inference_engine.predicate
from tonggeometry.inference_engine.primitives import Circle
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.inference_engine.database import Database
    from tonggeometry.inference_engine.predicate import Fact, Predicate


def known(db: 'Database', c1: Circle, c2: Circle) -> bool:
    """Check if the fact is known."""
    return c1 == c2 or c1 in db.inverse_eqcircle and c2 in db.eqcircle[
        db.inverse_eqcircle[c1]]


def itscc(db: 'Database', c1: Circle, c2: Circle) -> Optional[List]:
    """Return the intersection of two representative circles from the circles
    on the circles."""
    c1 = db.inverse_eqcircle[c1]
    c2 = db.inverse_eqcircle[c2]
    if c1 == c2 or (c1 not in db.intersect_circle_circle
                    or c2 not in db.intersect_circle_circle[c1]):
        return None
    return db.intersect_circle_circle[c1][c2]


def stringify(db: 'Database') -> str:
    """Stringify related contents in datbase."""
    s = "\n> Eqcircle Facts\n"
    for circle in db.circles_circles:
        c = db.circles_circles[circle]
        s += f"  circle( {c.center}, "
        for p in c.points:
            s += f"[{p}] "
        s += ")\n"
    return s


def fact_key(fact: Union['Predicate', 'Fact']) -> str:
    """Generate key."""
    if isinstance(fact.objects[0], Circle) and isinstance(
            fact.objects[1], Circle):  # two Circles
        c1, c2 = fact.objects
    else:  # two lists / tuples of points
        c1 = Circle(fact.objects[0][0], fact.objects[0][1:])
        c2 = Circle(fact.objects[1][0], fact.objects[1][1:])
    sorted_objects = ", ".join(map(lambda x: str(x), sorted([c1, c2])))  # pylint: disable=unnecessary-lambda
    return f"{fact.type} ({sorted_objects})"


def init(db: 'Database'):
    """Initialize handler for db."""
    db.eqcircle = OrderedSet(
    )  # OrderedSet[Circle, OrderedSet[Circle]] (rep -> eqclass)
    db.inverse_eqcircle = {}  # Dict[Circle, Circle] (elem -> rep)
    db.circles_circles = OrderedSet(
    )  # OrderedSet[Circle, Circle], the mapped Circle is the merged one
    db.points_circles = {}  # Dict[Point, OrderedSet[Circle]]
    db.centers_circles = {}  # Dict[Point, OrderedSet[Circle]]
    db.intersect_circle_circle = {}  # {c1: {c2: [p1, p2]}, c2: {c1: [p1, p2]}}
    db.axes = {}  # {c: (ax, p)}, all the points on c's axes
    db.points_axes = {}  # {p: {ax: center}} from points to the circle
    db.radaxes = {
    }  # {c1: {c2: {p: reasons}}, c2: {c1: {p: reasons}}} radical axis
    db.points_radaxes = {
    }  # {p: {c1: {c2: None}, c2: {c1: None}}} inverse link from points
    db.simili = {}  # {c1: {c2: {True: None, False: None}}}


def predicate_to_fact(predicate: 'Predicate') -> 'Fact':
    """Convert predicate to fact."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    if isinstance(predicate.objects[0], Circle) and isinstance(
            predicate.objects[1], Circle):  # two Circles
        return Fact("eqcircle", predicate.objects)
    # two lists / tuples of points
    c1 = Circle(predicate.objects[0][0], predicate.objects[0][1:])
    c2 = Circle(predicate.objects[1][0], predicate.objects[1][1:])
    return Fact("eqcircle", [c1, c2])


def filter(db: 'Database', fact: 'Fact') -> Optional['Fact']:
    """Filter trivial facts. Note that known and filter are different. Known
    facts are not necessarily processed by fc but in db."""
    if fact.objects[0] == fact.objects[1]:
        return None
    return fact


def add_fact(db: 'Database', fact: 'Fact') -> List['Fact']:
    """Add Fact(circle, [c1, c2])."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    facts_from_add = []
    c1, c2 = fact.objects

    if known(db, c1, c2):
        return facts_from_add

    if c1 not in db.inverse_eqcircle and c2 not in db.inverse_eqcircle:
        if c1 > c2:
            c1, c2 = c2, c1
        db.eqcircle[c1] = OrderedSet.fromkeys([c1, c2])
        db.inverse_eqcircle[c1] = c1
        db.inverse_eqcircle[c2] = c1
        db.intersect_line_circle[c1] = {}
        db.intersect_circle_circle[c1] = {}
        c1_key = c1
        c2_key = c2
        c = Circle(c1_key.center, [])
        db.circles_circles[c1_key] = c
        if c1_key.center:
            if c1_key.center in db.centers_circles:
                db.centers_circles[c1_key.center][c1_key] = None
            else:
                db.centers_circles[c1_key.center] = OrderedSet.fromkeys(
                    [c1_key])
        new_points = OrderedSet.fromkeys(
            list(c1_key.points) + list(c2_key.points))
        for p in new_points:
            db.circles_circles[c1_key].add_point(p)
            if p in db.points_circles:
                for cc in db.points_circles[p]:
                    if cc not in db.intersect_circle_circle[c1_key]:
                        db.intersect_circle_circle[c1_key][cc] = []
                        db.intersect_circle_circle[cc][c1_key] = []
                    db.intersect_circle_circle[c1_key][cc].append(p)
                    db.intersect_circle_circle[cc][c1_key].append(p)
                db.points_circles[p][c1_key] = None
            else:
                db.points_circles[p] = OrderedSet.fromkeys([c1_key])
            for s in db.points_lines[p]:
                if c1_key not in db.intersect_line_circle[s]:
                    db.intersect_line_circle[s][c1_key] = []
                    db.intersect_line_circle[c1_key][s] = []
                db.intersect_line_circle[s][c1_key].append(p)
                db.intersect_line_circle[c1_key][s].append(p)
    elif not (c1 in db.inverse_eqcircle and c2 in db.inverse_eqcircle):  # pylint: disable=too-many-nested-blocks
        if c1 not in db.inverse_eqcircle:
            c1, c2 = c2, c1  # c1 existing, c2 new
        key = db.inverse_eqcircle[c1]
        for c in db.eqcircle[key]:
            if c == c1:
                continue
            f = Fact("eqcircle", [c, c2], "add_fact")
            f.add_parent(Fact("eqcircle", [c, c1]))
            f.add_parent(Fact("eqcircle", [c1, c2]))
            facts_from_add.append(f)
        db.eqcircle[key][c2] = None
        db.inverse_eqcircle[c2] = key
        c1_key = key
        c2_key = c2
        c = db.circles_circles[c1_key]
        if not c.center and c2_key.center:
            c.center = c2_key.center
            if c.center in db.centers_circles:
                db.centers_circles[c.center][c1_key] = None
            else:
                db.centers_circles[c.center] = OrderedSet.fromkeys([c1_key])
        new_points = OrderedSet.fromkeys([
            p for p in c2_key.points
            if p not in db.circles_circles[c1_key].points
        ])
        for p in new_points:
            db.circles_circles[c1_key].add_point(p)
            if p in db.points_circles:
                for cc in db.points_circles[p]:
                    if cc not in db.intersect_circle_circle[c1_key]:
                        db.intersect_circle_circle[c1_key][cc] = []
                        db.intersect_circle_circle[cc][c1_key] = []
                    db.intersect_circle_circle[c1_key][cc].append(p)
                    db.intersect_circle_circle[cc][c1_key].append(p)
                db.points_circles[p][c1_key] = None
            else:
                db.points_circles[p] = OrderedSet.fromkeys([c1_key])
            for s in db.points_lines[p]:
                if c1_key not in db.intersect_line_circle[s]:
                    db.intersect_line_circle[s][c1_key] = []
                    db.intersect_line_circle[c1_key][s] = []
                db.intersect_line_circle[s][c1_key].append(p)
                db.intersect_line_circle[c1_key][s].append(p)
    else:  # c1 c2 can't be in the same eqclass otherwise known
        c1_key = db.inverse_eqcircle[c1]
        c2_key = db.inverse_eqcircle[c2]
        if c1_key > c2_key:
            c1_key, c2_key = c2_key, c1_key
            c1, c2 = c2, c1
        to_merge = OrderedSet()
        to_merge.update(db.eqcircle.pop(c2_key))
        for c in db.eqcircle[c1_key]:
            for cc in to_merge:
                if c == c1 and cc == c2:
                    continue
                f = Fact("eqcircle", [c, cc], "add_fact")
                if c1 != c:
                    f.add_parent(Fact("eqcircle", [c, c1]))
                f.add_parent(Fact("eqcircle", [c1, c2]))
                if c2 != cc:
                    f.add_parent(Fact("eqcircle", [c2, cc]))
                facts_from_add.append(f)
        db.eqcircle[c1_key].update(to_merge)
        for k in to_merge:
            db.inverse_eqcircle[k] = c1_key

        cc = db.circles_circles.pop(c2_key)
        if cc.center:
            db.centers_circles[cc.center].pop(c2_key)
            db.circles_circles[c1_key].center = cc.center
            if cc.center in db.centers_circles:
                db.centers_circles[cc.center][c1_key] = None
            else:
                db.centers_circles[cc.center] = OrderedSet.fromkeys([c1_key])
        for p in cc.points:
            db.circles_circles[c1_key].add_point(p)
            db.points_circles[p].pop(c2_key)
            for cc in db.points_circles[p]:
                if cc == c1_key:
                    continue
                if cc not in db.intersect_circle_circle[c1_key]:
                    db.intersect_circle_circle[c1_key][cc] = []
                    db.intersect_circle_circle[cc][c1_key] = []
                if p not in db.intersect_circle_circle[c1_key][cc]:
                    db.intersect_circle_circle[c1_key][cc].append(p)
                    db.intersect_circle_circle[cc][c1_key].append(p)
            db.points_circles[p][c1_key] = None
            for s in db.points_lines[p]:
                if s not in db.intersect_line_circle[c1_key]:
                    db.intersect_line_circle[c1_key][s] = []
                    db.intersect_line_circle[s][c1_key] = []
                if p not in db.intersect_line_circle[c1_key][s]:
                    db.intersect_line_circle[c1_key][s].append(p)
                    db.intersect_line_circle[s][c1_key].append(p)
        for c in db.intersect_circle_circle.pop(c2_key):
            db.intersect_circle_circle[c].pop(c2_key)
        for s in db.intersect_line_circle.pop(c2_key):
            db.intersect_line_circle[s].pop(c2_key)
        if c2_key in db.axes:
            axes_dict = db.axes.pop(c2_key)
            if c1_key not in db.axes:
                db.axes[c1_key] = {}
            db.axes[c1_key].update(axes_dict)
        if c2_key in db.radaxes:
            radaxes_dict = db.radaxes.pop(c2_key)
            if c1_key not in db.radaxes:
                db.radaxes[c1_key] = {}
            for c, p_dict in radaxes_dict.items():
                db.radaxes[c].pop(c2_key)
                if c != c1_key:
                    if c not in db.radaxes[c1_key]:
                        db.radaxes[c1_key][c] = OrderedSet()
                        db.radaxes[c][c1_key] = OrderedSet()
                for p, val in p_dict.items():
                    db.points_radaxes[p][c].pop(c2_key)
                    if c2_key in db.points_radaxes[p]:
                        db.points_radaxes[p].pop(c2_key)
                    if c != c1_key:
                        db.points_radaxes[p][c][c1_key] = None
                        if c1_key not in db.points_radaxes[p]:
                            db.points_radaxes[p][c1_key] = OrderedSet()
                        db.points_radaxes[p][c1_key][c] = None
                        if p not in db.radaxes[c1_key][c]:
                            db.radaxes[c1_key][c][p] = val
                            db.radaxes[c][c1_key][p] = (val[0], val[2], val[1])
        if c2_key in db.simili:
            simili_dict = db.simili.pop(c2_key)
            if c1_key not in db.simili:
                db.simili[c1_key] = OrderedSet()
            db.simili[c1_key].update(simili_dict)
            for c_key in simili_dict:
                if c1_key not in db.simili[c_key]:
                    db.simili[c_key][c1_key] = {True: None, False: None}
                db.simili[c_key][c1_key].update(db.simili[c_key].pop(c2_key))
    return facts_from_add
