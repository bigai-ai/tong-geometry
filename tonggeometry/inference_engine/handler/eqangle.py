r"""eqangle handler for Database."""

from typing import TYPE_CHECKING, List, Optional, Union

import tonggeometry.inference_engine.predicate
from tonggeometry.inference_engine.primitives import Angle
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.inference_engine.database import Database
    from tonggeometry.inference_engine.predicate import Fact, Predicate


def known(db: 'Database', a1: Angle, a2: Angle) -> bool:
    """Check if the fact is known."""
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if db.is_eqline(s1, s2) and db.is_eqline(s3, s4):
        return True
    if (a1 in db.inverse_eqangle and a2 in db.eqangle[db.inverse_eqangle[a1]]):
        eqclass_rep_a1 = db.inverse_eqangle[a1]
        eqclass = db.eqangle[eqclass_rep_a1]
        the_a1 = eqclass[a1]
        the_a2 = eqclass[a2]
        a1_dir = the_a1.s1 == s1 and the_a1.s2 == s2
        a2_dir = the_a2.s1 == s3 and the_a2.s2 == s4
        if a1_dir == a2_dir:
            return True
    return False


def stringify(db: 'Database') -> str:
    """Stringify related contents in datbase."""
    s = "\n> Eqangle Facts\n"
    for a in db.eqangle:
        s += "  eqangle( "
        for aa in db.eqangle[a]:
            s += f"[{aa.name}] "
        s += ")\n"
    return s


def fact_key(fact: Union['Predicate', 'Fact']) -> str:
    """Generate key."""
    if len(fact.objects) == 2:  # two Angles
        a1 = fact.objects[0].name
        a2 = fact.objects[1].name
        a1_p = a1[::-1]
        a2_p = a2[::-1]
    else:  # six points
        a1 = "".join(fact.objects[:3])
        a2 = "".join(fact.objects[3:])
        a1_p = a1[::-1]
        a2_p = a2[::-1]
    all_choices = [
        ", ".join(choice)
        for choice in [[a1, a2], [a2, a1], [a1_p, a2_p], [a2_p, a1_p]]
    ]
    sorted_objects = min(all_choices)
    return f"{fact.type} ({sorted_objects})"


def init(db: 'Database'):
    """Initialize handler in db."""
    db.eqangle = OrderedSet(
    )  # OrderedSet[Angle, OrderedSet[Angle]] (rep -> eqclass)
    db.inverse_eqangle = {}  # Dict[Angle, Angle] (elem -> rep)
    db.segments_eqangles = {}  # Dict[Segment, OrderedSet[Angle]]
    db.points_eqangles = {}  # Dict[Point, OrderedSet[Segment]]


def predicate_to_fact(predicate: 'Predicate') -> 'Fact':
    """Convert predicate to fact."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    if len(predicate.objects) == 2:  # two Angles
        return Fact("eqangle", predicate.objects)
    A, B, C, D, E, F = predicate.objects  # six points
    return Fact("eqangle", [Angle(A, B, C), Angle(D, E, F)])


def filter(db: 'Database', fact: 'Fact') -> Optional['Fact']:
    """Filter trivial facts. Note that known and filter are different. Known
    facts are not necessarily processed by fc but in db. It's necessary to have
    <ABC=<A'BC' where ABA' and CBC' are eqline."""
    a1, a2 = fact.objects
    if len(set(a1.name)) != 3 or len(set(a2.name)) != 3:
        return None
    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2
    if (s1 == s3
            and s2 == s4) or db.is_eqline(s1, s2) and db.is_eqline(s3, s4):
        return None
    return fact


def add_fact(db: 'Database', fact: 'Fact') -> List['Fact']:
    """Add Fact(eqangle, [a1(s1, s2), a2(s3, s4)]). Fact should have updated
    lines and filtered. It's necessary to have <ABC=<A'BC' where ABA' and CBC'
    are eqline."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    facts_from_add = []

    a1, a2 = fact.objects

    if known(db, a1, a2):
        return facts_from_add

    if a1 == a2:  # a1 == a2 but not known, let forward chainer handle
        return facts_from_add

    s1, s2 = a1.s1, a1.s2
    s3, s4 = a2.s1, a2.s2

    if db.is_eqline(s1, s2) and not db.is_eqline(s3, s4) or not db.is_eqline(
            s1, s2) and db.is_eqline(s3, s4):
        return facts_from_add

    if a1 not in db.inverse_eqangle and a2 not in db.inverse_eqangle:  # pylint: disable=too-many-nested-blocks
        if a1 > a2:
            a1, a2 = a2, a1
        db.eqangle[a1] = OrderedSet(zip([a1, a2], [a1, a2]))
        db.inverse_eqangle[a1] = a1
        db.inverse_eqangle[a2] = a1
        for a in [a1, a2]:
            for s in [a.s1, a.s2]:
                if s in db.segments_eqangles:
                    db.segments_eqangles[s][a] = None
                else:
                    db.segments_eqangles[s] = OrderedSet.fromkeys([a])
                    for p in [s.p1, s.p2]:
                        if p in db.points_eqangles:
                            db.points_eqangles[p][s] = None
                        else:
                            db.points_eqangles[p] = OrderedSet.fromkeys([s])
    elif not (a1 in db.inverse_eqangle and a2 in db.inverse_eqangle):
        if a1 not in db.inverse_eqangle:
            a1, a2 = a2, a1  # a1 existing, a2 new
        key = db.inverse_eqangle[a1]
        a0 = db.eqangle[key][a1]
        if a1.name != a0.name:
            a2 = Angle(a2.p3, a2.p2, a2.p1)
            a1 = a0
        for a in db.eqangle[key].values():
            if a == a1:
                continue
            f = Fact("eqangle", [a, a2], "add_fact")
            f.add_parent(Fact("eqangle", [a, a1]))
            f.add_parent(Fact("eqangle", [a1, a2]))
            facts_from_add.append(f)
        db.eqangle[key][a2] = a2
        db.inverse_eqangle[a2] = key
        for s in [a2.s1, a2.s2]:
            if s in db.segments_eqangles:
                db.segments_eqangles[s][a2] = None
            else:
                db.segments_eqangles[s] = OrderedSet.fromkeys([a2])
                for p in [s.p1, s.p2]:
                    if p in db.points_eqangles:
                        db.points_eqangles[p][s] = None
                    else:
                        db.points_eqangles[p] = OrderedSet.fromkeys([s])
    else:  # a1 a2 both in eqangle
        # a1 a2 in the same eqclass, let forward chainer handle it
        if db.inverse_eqangle[a1] == db.inverse_eqangle[a2]:
            return facts_from_add
        # a1 a2 not in the same eqclass
        a1_key = db.inverse_eqangle[a1]
        a2_key = db.inverse_eqangle[a2]
        if a1_key > a2_key:
            a1_key, a2_key = a2_key, a1_key
            a1, a2 = a2, a1
        a0 = db.eqangle[a1_key][a1]
        if a1.name != a0.name:
            a2 = Angle(a2.p3, a2.p2, a2.p1)
            a1 = a0
        to_merge = OrderedSet()
        eqclass = db.eqangle.pop(a2_key)
        a0 = eqclass[a2]
        if a2.name != a0.name:
            for a in eqclass:
                new_a = Angle(a.p3, a.p2, a.p1)
                for s in [a.s1, a.s2]:
                    db.segments_eqangles[s].pop(a)
                    db.segments_eqangles[s][new_a] = None
                to_merge[new_a] = new_a
        else:
            to_merge.update(eqclass)
        for a in db.eqangle[a1_key].values():
            for aa in to_merge.values():
                if a == a1 and aa == a2:
                    continue
                f = Fact("eqangle", [a, aa], "add_fact")
                if a != a1:
                    f.add_parent(Fact("eqangle", [a, a1]))
                f.add_parent(Fact("eqangle", [a1, a2]))
                if aa != a2:
                    f.add_parent(Fact("eqangle", [a2, aa]))
                facts_from_add.append(f)
        db.eqangle[a1_key].update(to_merge)
        for k in to_merge:
            db.inverse_eqangle[k] = a1_key
    return facts_from_add
