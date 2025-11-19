r"""para handler for Database."""

from typing import TYPE_CHECKING, List, Optional, Union

import tonggeometry.inference_engine.predicate
from tonggeometry.inference_engine.primitives import Segment
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.inference_engine.database import Database
    from tonggeometry.inference_engine.predicate import Fact, Predicate


def known(db: 'Database', s1: Segment, s2: Segment) -> bool:
    """Check if the fact is known."""
    return not db.is_eqline(
        s1,
        s2) and s1 in db.inverse_para and s2 in db.para[db.inverse_para[s1]]


def stringify(db: 'Database') -> str:
    """Stringify related contents in datbase."""
    s = "\n> Para Facts\n"
    for seg in db.para:
        s += "  para( "
        for sseg in db.para[seg]:
            s += f"[{sseg}] "
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
    """Initialize handler for db."""
    db.para = OrderedSet(
    )  # OrderedSet[Segment, OrderedSet[Segment]] (rep -> eqclass)
    db.inverse_para = {}  # Dict[Segment, Segment] (elem -> rep)
    db.points_paras = {}  # Dict[Point, OrderedSet[Segment]]


def predicate_to_fact(predicate: 'Predicate') -> 'Fact':
    """Convert predicate to fact."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    if len(predicate.objects) == 2:  # two Segments
        return Fact("para", predicate.objects)
    A, B, C, D = predicate.objects  # four points
    s_AB = Segment(A, B)
    s_CD = Segment(C, D)
    return Fact("para", [s_AB, s_CD])


def filter(db: 'Database', fact: 'Fact') -> Optional['Fact']:
    """Filter trivial facts. Note that known and filter are different. Known
    facts are not necessarily processed by fc but in db."""
    s1, s2 = fact.objects
    if s1.p1 == s1.p2 or s2.p1 == s2.p2 or db.is_eqline(s1, s2):
        return None
    return fact


def add_fact(db: 'Database', fact: 'Fact') -> List['Fact']:
    """Add Fact(para, [s1, s2]). Fact is filtered."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    facts_from_add = []

    s1, s2 = fact.objects

    if db.is_eqline(s1, s2) or known(db, s1, s2):
        return facts_from_add

    p1, p2 = s1.p1, s1.p2
    p3, p4 = s2.p1, s2.p2
    if p3 in (p1, p2) or p4 in (p1, p2):
        return facts_from_add

    if s1 not in db.inverse_para and s2 not in db.inverse_para:
        if s1 > s2:
            s1, s2 = s2, s1
        db.para[s1] = OrderedSet.fromkeys([s1, s2])
        db.inverse_para[s1] = s1
        db.inverse_para[s2] = s1
        for s in [s1, s2]:
            for p in [s.p1, s.p2]:
                if p in db.points_paras:
                    db.points_paras[p][s] = None
                else:
                    db.points_paras[p] = OrderedSet.fromkeys([s])
    elif not (s1 in db.inverse_para and s2 in db.inverse_para):
        if s1 not in db.inverse_para:
            s1, s2 = s2, s1  # s1 existing, s2 new
        key = db.inverse_para[s1]
        for s in db.para[key]:
            if s == s1:
                continue
            f = Fact("para", [s, s2], "add_fact")
            f.add_parent(Fact("para", [s, s1]))
            f.add_parent(Fact("para", [s1, s2]))
            facts_from_add.append(f)
        db.para[key][s2] = None
        db.inverse_para[s2] = key
        for p in [s2.p1, s2.p2]:
            if p in db.points_paras:
                db.points_paras[p][s2] = None
            else:
                db.points_paras[p] = OrderedSet.fromkeys([s2])
    else:  # s1 s2 can't be in the same eqclass otherwise known
        s1_key = db.inverse_para[s1]
        s2_key = db.inverse_para[s2]
        if s1_key > s2_key:
            s1_key, s2_key = s2_key, s1_key
            s1, s2 = s2, s1
        to_merge = OrderedSet()
        to_merge.update(db.para.pop(s2_key))
        for s in db.para[s1_key]:
            for ss in to_merge:
                if s == s1 and ss == s2:
                    continue
                f = Fact("para", [s, ss], "add_fact")
                if s != s1:
                    f.add_parent(Fact("para", [s, s1]))
                f.add_parent(Fact("para", [s1, s2]))
                if ss != s2:
                    f.add_parent(Fact("para", [s2, ss]))
                facts_from_add.append(f)
        db.para[s1_key].update(to_merge)
        for k in to_merge:
            db.inverse_para[k] = s1_key
    return facts_from_add
