r"""eqratio handler for Database."""

from typing import TYPE_CHECKING, List, Optional, Union

import tonggeometry.inference_engine.predicate
from tonggeometry.inference_engine.primitives import Ratio, Segment
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.inference_engine.database import Database
    from tonggeometry.inference_engine.predicate import Fact, Predicate


def known(db: 'Database', r1: Ratio, r2: Ratio) -> bool:
    """Check if the fact is known."""
    s1, s2 = r1.s1, r1.s2
    s3, s4 = r2.s1, r2.s2
    if db.is_cong(s1, s2) and db.is_cong(s3, s4):
        return True
    if (r1 in db.inverse_eqratio and r2 in db.eqratio[db.inverse_eqratio[r1]]):
        eqclass_rep_r1 = db.inverse_eqratio[r1]
        eqclass = db.eqratio[eqclass_rep_r1]
        the_r1 = eqclass[r1]
        the_r2 = eqclass[r2]
        r1_dir = the_r1.s1 == s1 and the_r1.s2 == s2
        r2_dir = the_r2.s1 == s3 and the_r2.s2 == s4
        if r1_dir == r2_dir:
            return True
    return False


def stringify(db: 'Database') -> str:
    """Stringify related contents in datbase."""
    s = "\n> Eqratio Facts\n"
    for r in db.eqratio:
        s += "  eqratio( "
        for rr in db.eqratio[r]:
            s += f"[{rr.name}] "
        s += ")\n"
    return s


def fact_key(fact: Union['Predicate', 'Fact']) -> str:
    """Generate key."""
    if len(fact.objects) == 2:  # two Ratios
        s1 = str(fact.objects[0].s1)
        s2 = str(fact.objects[0].s2)
        s3 = str(fact.objects[1].s1)
        s4 = str(fact.objects[1].s2)
    elif len(fact.objects) == 4:  # four Segments
        s1, s2, s3, s4 = fact.objects
        s1 = str(s1)
        s2 = str(s2)
        s3 = str(s3)
        s4 = str(s4)
    else:  # eight points
        s1 = "".join(sorted(fact.objects[:2]))
        s2 = "".join(sorted(fact.objects[2:4]))
        s3 = "".join(sorted(fact.objects[4:6]))
        s4 = "".join(sorted(fact.objects[6:]))
    all_choices = [
        ", ".join(choice) for choice in [[s1, s2, s3, s4], [s2, s1, s4, s3],
                                         [s3, s4, s1, s2], [s4, s3, s2, s1]]
    ]
    sorted_objects = min(all_choices)
    return f"{fact.type} ({sorted_objects})"


def init(db: 'Database'):
    """Initialize handler in db."""
    db.eqratio = OrderedSet(
    )  # OrderedSet[Ratio, OrderedSet[Ratio]] (rep -> eqclass)
    db.inverse_eqratio = {}  # Dict[Ratio, Ratio] (elem -> rep)
    db.segments_eqratios = {}  # Dict[Segment, OrderedSet[Ratio]]
    db.points_eqratios = {}  # Dict[Point, OrderedSet[Segment]]


def predicate_to_fact(predicate: 'Predicate') -> 'Fact':
    """Convert predicate to fact."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    if len(predicate.objects) == 2:  # two Ratios
        return Fact("eqratio", predicate.objects)
    if len(predicate.objects) == 4:  # four Segments
        return Fact(
            "eqratio",
            [Ratio(*predicate.objects[:2]),
             Ratio(*predicate.objects[2:])])
    A, B, C, D, E, F, G, H = predicate.objects  # eight points
    return Fact("eqratio", [
        Ratio(Segment(A, B), Segment(C, D)),
        Ratio(Segment(E, F), Segment(G, H))
    ])


def filter(db: 'Database', fact: 'Fact') -> Optional['Fact']:
    """Filter trivial facts. Note that known and filter are different. Known
    facts are not necessarily processed by fc but in db."""
    s1 = fact.objects[0].s1
    s2 = fact.objects[0].s2
    s3 = fact.objects[1].s1
    s4 = fact.objects[1].s2
    if s1.p1 == s1.p2 or s2.p1 == s2.p2 or s3.p1 == s3.p2 or s4.p1 == s4.p2:
        return None
    if s1 == s3 and s2 == s4 or db.is_cong(s1, s2) and db.is_cong(s3, s4):
        return None
    return fact


def add_fact(db: 'Database', fact: 'Fact') -> List['Fact']:
    """Add Fact(eqratio, [r1(s1, s2), r2(s3, s4)]). Facts are filtered."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    facts_from_add = []

    r1, r2 = fact.objects

    if known(db, r1, r2):
        return facts_from_add

    s1, s2 = r1.s1, r1.s2
    s3, s4 = r2.s1, r2.s2

    if db.is_cong(s1, s2) and not db.is_cong(s3, s4) or not db.is_cong(
            s1, s2) and db.is_cong(s3, s4):
        return facts_from_add

    if db.is_cong(s1, s4) and db.is_cong(s2, s3):
        return facts_from_add

    if r1 not in db.inverse_eqratio and r2 not in db.inverse_eqratio:  # pylint: disable=too-many-nested-blocks
        if r1 > r2:
            r1, r2 = r2, r1
        db.eqratio[r1] = OrderedSet(zip([r1, r2], [r1, r2]))
        db.inverse_eqratio[r1] = r1
        db.inverse_eqratio[r2] = r1
        for r in [r1, r2]:
            for s in [r.s1, r.s2]:
                if s in db.segments_eqratios:
                    db.segments_eqratios[s][r] = None
                else:
                    db.segments_eqratios[s] = OrderedSet.fromkeys([r])
                    for p in [s.p1, s.p2]:
                        if p in db.points_eqratios:
                            db.points_eqratios[p][s] = None
                        else:
                            db.points_eqratios[p] = OrderedSet.fromkeys([s])
    elif not (r1 in db.inverse_eqratio and r2 in db.inverse_eqratio):
        if r1 not in db.inverse_eqratio:
            r1, r2 = r2, r1  # r1 existing, r2 new
        key = db.inverse_eqratio[r1]
        r0 = db.eqratio[key][r1]
        if r1.name != r0.name:
            r2 = Ratio(r2.s2, r2.s1)
            r1 = r0
        for r in db.eqratio[key].values():
            if r == r1:
                continue
            f = Fact("eqratio", [r, r2], "add_fact")
            f.add_parent(Fact("eqratio", [r, r1]))
            f.add_parent(Fact("eqratio", [r1, r2]))
            facts_from_add.append(f)
        db.eqratio[key][r2] = r2
        db.inverse_eqratio[r2] = key
        for s in [r2.s1, r2.s2]:
            if s in db.segments_eqratios:
                db.segments_eqratios[s][r2] = None
            else:
                db.segments_eqratios[s] = OrderedSet.fromkeys([r2])
                for p in [s.p1, s.p2]:
                    if p in db.points_eqratios:
                        db.points_eqratios[p][s] = None
                    else:
                        db.points_eqratios[p] = OrderedSet.fromkeys([s])
    else:  # r1 r2 both in eqratio
        # r1 r2 in the same eqclass, let forward chainer handle it
        if db.inverse_eqratio[r1] == db.inverse_eqratio[r2]:
            return facts_from_add
        # r1 r2 not in the same eqclass
        r1_key = db.inverse_eqratio[r1]
        r2_key = db.inverse_eqratio[r2]
        if r1_key > r2_key:
            r1_key, r2_key = r2_key, r1_key
            r1, r2 = r2, r1
        r0 = db.eqratio[r1_key][r1]
        if r1.name != r0.name:
            r2 = Ratio(r2.s2, r2.s1)
            r1 = r0
        to_merge = OrderedSet()
        eqclass = db.eqratio.pop(r2_key)
        r0 = eqclass[r2]
        if r2.name != r0.name:
            for r in eqclass:
                new_r = Ratio(r.s2, r.s1)
                db.segments_eqratios[r.s1].pop(r)
                db.segments_eqratios[r.s1][new_r] = None
                db.segments_eqratios[r.s2].pop(r)
                db.segments_eqratios[r.s2][new_r] = None
                to_merge[new_r] = new_r
        else:
            to_merge.update(eqclass)
        for r in db.eqratio[r1_key].values():
            for rr in to_merge.values():
                if r == r1 and rr == r2:
                    continue
                f = Fact("eqratio", [r, rr], "add_fact")
                if r != r1:
                    f.add_parent(Fact("eqratio", [r, r1]))
                f.add_parent(Fact("eqratio", [r1, r2]))
                if rr != r2:
                    f.add_parent(Fact("eqratio", [r2, rr]))
                facts_from_add.append(f)
        db.eqratio[r1_key].update(to_merge)
        for k in to_merge:
            db.inverse_eqratio[k] = r1_key
    return facts_from_add
