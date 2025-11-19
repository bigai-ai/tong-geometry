r"""cong handler for Database."""

from typing import TYPE_CHECKING, List, Optional, Union

import tonggeometry.inference_engine.predicate
from tonggeometry.inference_engine.primitives import Segment
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.inference_engine.database import Database
    from tonggeometry.inference_engine.predicate import Fact, Predicate


def known(db: 'Database', s1: Segment, s2: Segment) -> bool:
    """Check if the fact is known."""
    return s1 == s2 or s1 in db.inverse_cong and s2 in db.cong[
        db.inverse_cong[s1]]


def stringify(db: 'Database') -> str:
    """Stringify related contents in datbase."""
    s = "\n> Cong Facts\n"
    for c in db.cong:
        s += "  cong( "
        for cc in db.cong[c]:
            s += f"[{cc}] "
        s += ")\n"
    return s


def fact_key(fact: Union['Predicate', 'Fact']) -> str:
    """Generate key."""
    if len(fact.objects) == 2:  # two Segments
        s1 = str(fact.objects[0])
        s2 = str(fact.objects[1])
    else:  # four points
        s1 = "".join(sorted(fact.objects[:2]))
        s2 = "".join(sorted(fact.objects[2:]))
    sorted_objects = ", ".join(sorted([s1, s2]))
    return f"{fact.type} ({sorted_objects})"


def init(db: 'Database'):
    """Initialize handler in db."""
    db.cong = OrderedSet(
    )  # OrderedSet[Segment, OrderedSet[Segment]] (rep -> eqclass)
    db.inverse_cong = {}  # Dict[Segment, Segment] (elem -> rep)
    db.points_congs = {}  # Dict[Point, OrderedSet[Segment]]
    db.l = {}  # l shape (O, P, A, B) where PAB eqline, OA=OB, A<B
    db.l_stick = {}  # from OP to AB
    db.l_radius = {}  # from OA to PB
    db.l_ratio = {}  # from PAB to O


def predicate_to_fact(predicate: 'Predicate') -> 'Fact':
    """Convert predicate to fact."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    if len(predicate.objects) == 2:  # two Segments
        return Fact("cong", predicate.objects)
    A, B, C, D = predicate.objects  # four Points
    return Fact("cong", [Segment(A, B), Segment(C, D)])


def filter(db: 'Database', fact: 'Fact') -> Optional['Fact']:
    """Filter trivial facts. Note that known and filter are different. Known
    facts are not necessarily processed by fc but in db."""
    s1, s2 = fact.objects
    if s1.p1 == s1.p2 or s2.p1 == s2.p2 or s1 == s2:
        return None
    return fact


def add_fact(db: 'Database', fact: 'Fact') -> List['Fact']:
    """Add Fact(cong, [s1, s2])."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    facts_from_add = []
    s1, s2 = fact.objects

    if known(db, s1, s2):
        return facts_from_add

    if s1 not in db.inverse_cong and s2 not in db.inverse_cong:
        if s1 > s2:
            s1, s2 = s2, s1
        db.cong[s1] = OrderedSet.fromkeys([s1, s2])
        db.inverse_cong[s1] = s1
        db.inverse_cong[s2] = s1
        for s in [s1, s2]:
            for p in [s.p1, s.p2]:
                if p in db.points_congs:
                    db.points_congs[p][s] = None
                else:
                    db.points_congs[p] = OrderedSet.fromkeys([s])
    elif not (s1 in db.inverse_cong and s2 in db.inverse_cong):
        if s1 not in db.inverse_cong:
            s1, s2 = s2, s1  # s1 existing, s2 new
        key = db.inverse_cong[s1]
        for s in db.cong[key]:
            if s == s1:
                continue
            f = Fact("cong", [s, s2], "add_fact")
            f.add_parent(Fact("cong", [s, s1]))
            f.add_parent(Fact("cong", [s1, s2]))
            facts_from_add.append(f)
        db.cong[key][s2] = None
        db.inverse_cong[s2] = key
        for p in [s2.p1, s2.p2]:
            if p in db.points_congs:
                db.points_congs[p][s2] = None
            else:
                db.points_congs[p] = OrderedSet.fromkeys([s2])
    else:  # s1 s2 can't be in the same eqclass otherwise known
        s1_key = db.inverse_cong[s1]
        s2_key = db.inverse_cong[s2]
        if s1_key > s2_key:
            s1_key, s2_key = s2_key, s1_key
            s1, s2 = s2, s1
        to_merge = OrderedSet()
        to_merge.update(db.cong.pop(s2_key))
        for s in db.cong[s1_key]:
            for ss in to_merge:
                if s == s1 and ss == s2:
                    continue
                f = Fact("cong", [s, ss], "add_fact")
                if s != s1:
                    f.add_parent(Fact("cong", [s, s1]))
                f.add_parent(Fact("cong", [s1, s2]))
                if ss != s2:
                    f.add_parent(Fact("cong", [s2, ss]))
                facts_from_add.append(f)
        db.cong[s1_key].update(to_merge)
        for k in to_merge:
            db.inverse_cong[k] = s1_key
    return facts_from_add
