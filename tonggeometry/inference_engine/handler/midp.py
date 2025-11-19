r"""midp handler for Database."""

from typing import TYPE_CHECKING, List, Optional, Union

import tonggeometry.inference_engine.predicate
from tonggeometry.inference_engine.primitives import Segment
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.inference_engine.database import Database
    from tonggeometry.inference_engine.predicate import Fact, Predicate


def known(db: 'Database', m: str, s: Segment) -> bool:
    """Check if the fact is known."""
    return m in db.midp and s in db.midp[m]


def stringify(db: 'Database') -> str:
    """Stringify related contents in datbase."""
    s = "\n> Midp Facts\n"
    for p in db.midp:
        s += f"  midp( {p}, "
        for ss in db.midp[p]:
            s += f"[{ss}] "
        s += ")\n"
    return s


def fact_key(fact: Union['Predicate', 'Fact']) -> str:
    """Generate key."""
    if len(fact.objects) == 2:  # point and Segment
        M, S = fact.objects
        s = str(S)
    else:  # three points
        M, A, B = fact.objects
        s = "".join(sorted([A, B]))
    sorted_objects = f"{M}, {s}"
    return f"{fact.type} ({sorted_objects})"


def init(db: 'Database'):
    """Initialize handler for db."""
    db.midp = OrderedSet(
    )  # OrderedSet[Point, OrderedSet[Segment]] (rep -> eqclass)
    db.inverse_midp = {}  # Dict[Segment, Point] (elem -> rep)
    db.points_midps = {}  # Dict[Point, OrderedSet[Segment]]
    db.centris = {}  # Dict[Triangle, G]
    db.inverse_centris = {}  # Dict[G, OrderedSet[Triangle]]
    db.h_segments_centris = {}  # Dict[Segment, OrderedSet[Triangle]]


def predicate_to_fact(predicate: 'Predicate') -> 'Fact':
    """Convert predicate to fact."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    if len(predicate.objects) == 2:  # point and Segment
        return Fact("midp", predicate.objects)
    M, A, B = predicate.objects  # three points
    return Fact("midp", [M, Segment(A, B)])


def filter(db: 'Database', fact: 'Fact') -> Optional['Fact']:
    """Filter trivial facts. Note that known and filter are different. Known
    facts are not necessarily processed by fc but in db."""
    M, s = fact.objects
    if s.p1 == s.p2 or M in [s.p1, s.p2]:
        return None
    return fact


def add_fact(db: 'Database', fact: 'Fact') -> List['Fact']:
    """Add Fact(midp, [M, Segment(A, B)]). No merge."""
    facts_from_add = []
    M, s = fact.objects

    if known(db, M, s):
        return facts_from_add

    if M not in db.midp:
        db.midp[M] = OrderedSet.fromkeys([s])
    else:
        db.midp[M][s] = None
    db.inverse_midp[s] = M
    for p in [s.p1, s.p2]:
        if p in db.points_midps:
            db.points_midps[p][s] = None
        else:
            db.points_midps[p] = OrderedSet.fromkeys([s])
    return facts_from_add
