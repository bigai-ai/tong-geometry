r"""contri handler for Database."""

from itertools import combinations
from typing import TYPE_CHECKING, List, Optional, Union

import tonggeometry.inference_engine.predicate
from tonggeometry.inference_engine.primitives import Segment, Triangle
from tonggeometry.inference_engine.util import sort_two_from_first
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.inference_engine.database import Database
    from tonggeometry.inference_engine.predicate import Fact, Predicate


def known(db: 'Database', t1: Triangle, t2: Triangle) -> bool:
    """Check if the fact is known."""
    return t1 != t2 and t1 in db.inverse_contri and t2 in db.contri[
        db.inverse_contri[t1]]


def stringify(db: 'Database') -> str:
    """Stringify related contents in datbase."""
    s = "\n> Contri Facts\n"
    for t in db.contri:
        s += "  contri( "
        for tt in db.contri[t]:
            s += f"[{tt.name}] "
        s += ")\n"
    return s


def fact_key(fact: Union['Predicate', 'Fact']) -> str:
    """Generate key."""
    if len(fact.objects) == 2:  # two Triangles
        t1, t2 = fact.objects[0].name, fact.objects[1].name
    else:  # six points
        t1, t2 = "".join(fact.objects[:3]), "".join(fact.objects[3:])
    all_choices = [
        ", ".join(sort_two_from_first(t1, t2)),
        ", ".join(sort_two_from_first(t2, t1))
    ]
    sorted_objects = min(all_choices)
    return f"{fact.type} ({sorted_objects})"


def init(db: 'Database'):
    """Initialize handler in db."""
    db.contri = OrderedSet(
    )  # OrderedSet[Triangle, OrderedSet[Triangle]] (rep -> eqclass)
    db.inverse_contri = {}  # Dict[Triangle, Triangle] (elem -> rep)
    db.segments_contris = {}  # Dict[Segment, OrderedSet[Triangle]]
    db.points_contris = {}  # Dict[Point, OrderedSet[Segement]]


def predicate_to_fact(predicate: 'Predicate') -> 'Fact':
    """Convert predicate to fact."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    if len(predicate.objects) == 2:  # two Triangles
        return Fact("contri", predicate.objects)
    A, B, C, D, E, F = predicate.objects  # six points
    return Fact("contri", [Triangle(A, B, C), Triangle(D, E, F)])


def filter(db: 'Database', fact: 'Fact') -> Optional['Fact']:
    """Filter trivial facts. Note that known and filter are different. Known
    facts are not necessarily processed by fc but in db."""
    t1, t2 = fact.objects
    if len(set(t1.name)) != 3 or len(set(t2.name)) != 3 or t1 == t2:
        return None
    return fact


def add_fact(db: 'Database', fact: 'Fact') -> List['Fact']:
    """Add Fact(contri, [t1, t2]). Be aware that the order of triangle
        vertices matters."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    facts_from_add = []

    t1, t2 = fact.objects

    if known(db, t1, t2):
        return facts_from_add

    if t1 not in db.inverse_contri and t2 not in db.inverse_contri:  # pylint: disable=too-many-nested-blocks
        if t1 > t2:
            t1, t2 = t2, t1
        db.contri[t1] = OrderedSet(zip([t1, t2], [t1, t2]))
        db.inverse_contri[t1] = t1
        db.inverse_contri[t2] = t1
        for t in [t1, t2]:
            for name in combinations(t.name, 2):
                s = Segment(*name)
                if s in db.segments_contris:
                    db.segments_contris[s][t] = None
                else:
                    db.segments_contris[s] = OrderedSet.fromkeys([t])
                    for p in [s.p1, s.p2]:
                        if p in db.points_contris:
                            db.points_contris[p][s] = None
                        else:
                            db.points_contris[p] = OrderedSet.fromkeys([s])
    elif not (t1 in db.inverse_contri and t2 in db.inverse_contri):
        if t1 not in db.inverse_contri:
            t1, t2 = t2, t1  # t1 existing, t2 new
        key = db.inverse_contri[t1]
        t0 = db.contri[key][t1]
        if t1.name != t0.name:
            indices = [t1.name.index(p) for p in t0.name]
            t2 = Triangle(*[t2.name[idx] for idx in indices])
            t1 = t0
        for t in db.contri[key].values():
            if t == t1:
                continue
            f = Fact("contri", [t, t2], "add_fact")
            f.add_parent(Fact("contri", [t, t1]))
            f.add_parent(Fact("contri", [t1, t2]))
            facts_from_add.append(f)
        db.contri[key][t2] = t2
        db.inverse_contri[t2] = key
        for name in combinations(t2.name, 2):
            s = Segment(*name)
            if s in db.segments_contris:
                db.segments_contris[s][t2] = None
            else:
                db.segments_contris[s] = OrderedSet.fromkeys([t2])
                for p in [s.p1, s.p2]:
                    if p in db.points_contris:
                        db.points_contris[p][s] = None
                    else:
                        db.points_contris[p] = OrderedSet.fromkeys([s])
    else:  # t1 t2 can't be in the same eqclass otherwise known
        t1_key = db.inverse_contri[t1]
        t2_key = db.inverse_contri[t2]
        if t1_key > t2_key:
            t1_key, t2_key = t2_key, t1_key
            t1, t2 = t2, t1
        t0 = db.contri[t1_key][t1]
        if t1.name != t0.name:
            indices = [t1.name.index(p) for p in t0.name]
            t2 = Triangle(*[t2.name[idx] for idx in indices])
            t1 = t0
        to_merge = OrderedSet()
        eqclass = db.contri.pop(t2_key)
        t0 = eqclass[t2]
        if t2.name != t0.name:
            indices = [t0.name.index(p) for p in t2.name]
            for t in eqclass:
                new_t = Triangle(*[t.name[idx] for idx in indices])
                for name in combinations(t.name, 2):
                    s = Segment(*name)
                    db.segments_contris[s].pop(t)
                    db.segments_contris[s][new_t] = None
                to_merge[new_t] = new_t
        else:
            to_merge.update(eqclass)
        for t in db.contri[t1_key].values():
            for tt in to_merge.values():
                if t == t1 and tt == t2:
                    continue
                f = Fact("contri", [t, tt], "add_fact")
                if t != t1:
                    f.add_parent(Fact("contri", [t, t1]))
                f.add_parent(Fact("contri", [t1, t2]))
                if tt != t2:
                    f.add_parent(Fact("contri", [t2, tt]))
                facts_from_add.append(f)
        db.contri[t1_key].update(to_merge)
        for k in to_merge:
            db.inverse_contri[k] = t1_key
    return facts_from_add
