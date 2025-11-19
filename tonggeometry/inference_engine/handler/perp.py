r"""perp handler for Database."""

from typing import TYPE_CHECKING, List, Optional, Union

import tonggeometry.inference_engine.predicate
from tonggeometry.inference_engine.primitives import Angle, Segment
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.inference_engine.database import Database
    from tonggeometry.inference_engine.predicate import Fact, Predicate


def known(db: 'Database', a: Angle) -> bool:
    """Check if the fact is known."""
    return a in db.perp


def stringify(db: 'Database') -> str:
    """Stringify related contents in datbase."""
    s = "\n> Perp Facts\n"
    s += "".join([f"  perp( [{a}] )\n" for a in db.perp])
    return s


def fact_key(fact: Union['Predicate', 'Fact']) -> str:
    """Generate key."""
    if len(fact.objects) == 1:  # one Angle
        s = str(fact.objects[0])
    else:  # three points
        s = str(Angle(*fact.objects))
    sorted_objects = s
    return f"{fact.type} ({sorted_objects})"


def init(db: 'Database'):
    """Initialize handler for db."""
    db.perp = OrderedSet()  # OrderedSet[Angle]
    db.segments_perps = {}  # Dict[Segment, OrderedSet[Angle]]
    db.h_segments_perps = {}  # Dict[Segment, OrderedSet[Angle]]
    db.points_perps = {}  # Dict[Point, OrderedSet[Segment]]


def predicate_to_fact(predicate: 'Predicate') -> 'Fact':
    """Convert predicate to fact."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    if len(predicate.objects) == 1:  # one Segment
        return Fact("perp", predicate.objects)
    A, B, C = predicate.objects  # three points
    a = Angle(A, B, C)
    return Fact("perp", [a])


def filter(db: 'Database', fact: 'Fact') -> Optional['Fact']:
    """Filter trivial facts. Note that known and filter are different. Known
    facts are not necessarily processed by fc but in db."""
    a = fact.objects[0]
    if len(set(a.name)) != 3:
        return None
    return fact


def add_fact(db: 'Database', fact: 'Fact') -> List['Fact']:
    """Add Fact(perp, [s1, s2]). Fact should have updated lines and
        filtered. No merge."""
    facts_from_add = []

    a = fact.objects[0]

    if known(db, a):
        return facts_from_add

    db.perp[a] = None
    for name in [str(a.s1), str(a.s2)]:
        s = Segment(*name)
        if s in db.segments_perps:
            db.segments_perps[s][a] = None
        else:
            db.segments_perps[s] = OrderedSet.fromkeys([a])
            for p in [s.p1, s.p2]:
                if p in db.points_perps:
                    db.points_perps[p][s] = None
                else:
                    db.points_perps[p] = OrderedSet.fromkeys([s])
    s = Segment(a.name[0], a.name[2])
    if s in db.h_segments_perps:
        db.h_segments_perps[s][a] = None
    else:
        db.h_segments_perps[s] = OrderedSet.fromkeys([a])
    return facts_from_add
