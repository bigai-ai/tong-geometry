r"""eqline handler for Database. Segment representative of eqline segements."""

from typing import TYPE_CHECKING, List, Optional, Union

import tonggeometry.inference_engine.predicate
from tonggeometry.inference_engine.primitives import Segment
from tonggeometry.util import OrderedSet

if TYPE_CHECKING:
    from tonggeometry.inference_engine.database import Database
    from tonggeometry.inference_engine.predicate import Fact, Predicate


def known(db: 'Database', s1: Segment, s2: Segment) -> bool:
    """Check if the fact is known."""
    return s1 == s2 or s1 in db.inverse_eqline and s2 in db.eqline[
        db.inverse_eqline[s1]]


def itsll(db: 'Database', s1: Segment, s2: Segment) -> Optional[str]:
    """Return the intersection of two lines. If same or no intersection,
    return None."""
    s1 = db.inverse_eqline[s1]
    s2 = db.inverse_eqline[s2]
    if s1 == s2 or (s1 not in db.intersect_line_line
                    or s2 not in db.intersect_line_line[s1]):
        return None
    return db.intersect_line_line[s1][s2]


def stringify(db: 'Database') -> str:
    """Stringify related contents in datbase."""
    s = "\n> Segment Facts\n"
    for l in db.lines_points:
        s += "  line( "
        for p in db.lines_points[l]:
            s += f"[{p}] "
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
    db.eqline = OrderedSet(
    )  # OrderedSet[Segment, OrderedSet[Segment] (rep -> eqclass)
    db.inverse_eqline = {}  # Dict[Segment, Segment] (elem -> rep)
    db.lines_points = OrderedSet()  # OrderedSet[Segment, OrderedSet[Point]]
    db.points_lines = {}  # Dict[Point, OrderedSet[Segment]]
    db.intersect_line_line = {}  # {l1: {l2: p}, l2: {l1: p}}
    db.intersect_line_circle = {}  # {l: {c: [p1, p2]}, c: {l: [p1, p2]}
    # Pappus
    db.x = {}  # x shapes of (A,B,b,a,x) keys are Aa Bb (A-x-b, B-x-a)
    db.inverse_x = {}  # save inverse x: {index: p} from (A,B,b,a) to x
    db.pappus = {}  # save potential conditions to prove using Pappus (line)
    # Desargues
    db.desargues = {}  # from fact key to new fact based on Desargues config
    # Harmonic
    db.harmonic = {}  # all normalized harmonic quadruples
    db.inverse_harmonic = {}  # from a pair to another harmonic pair {p: {}}
    db.harmonic_map = {}  # from ACB to (D, harmonic), note AC pair goes first
    # Cevians
    db.cevian = {}  # from ABCDEF to O
    db.inverse_cevian = {}  # from BCEF to ABCDEF, itsll, and base


def predicate_to_fact(predicate: 'Predicate') -> 'Fact':
    """Convert predicate to fact."""
    Fact = tonggeometry.inference_engine.predicate.Fact
    if len(predicate.objects) == 2:  # two Segments
        return Fact("eqline", predicate.objects)
    A, B, C, D = predicate.objects  # four Points
    s_AB = Segment(A, B)
    s_CD = Segment(C, D)
    return Fact("eqline", [s_AB, s_CD])


def filter(db: 'Database', fact: 'Fact') -> Optional['Fact']:
    """Filter trivial facts. Note that known and filter are different. Known
    facts are not necessarily processed by fc but in db."""
    s1, s2 = fact.objects
    if s1.p1 == s1.p2 or s2.p1 == s2.p2:
        return None
    if s1 in db.inverse_eqline and s1 == s2:
        return None
    return fact


def add_fact(db: 'Database', fact: 'Fact') -> List['Fact']:
    """Add Fact(eqline, [s1(p1, p2), s2(p3, p4)]).

    Note that all lines have been built before facts are added.
    """
    Fact = tonggeometry.inference_engine.predicate.Fact

    facts_from_add = []
    s1, s2 = fact.objects

    # remember to pass the trivial case for line creation
    if s1 in db.inverse_eqline and known(db, s1, s2):
        return facts_from_add

    # trivial case used for triggering forward chaining
    if s1 == s2:
        s = s1
        db.inverse_eqline[s] = s
        db.eqline[s] = OrderedSet.fromkeys([s])
        db.lines_points[s] = OrderedSet.fromkeys([s.p1, s.p2])
        db.intersect_line_line[s] = {}
        db.intersect_line_circle[s] = {}
        for p in db.lines_points[s]:
            if p in db.points_lines:
                for ss in db.points_lines[p]:
                    db.intersect_line_line[s][ss] = p
                    db.intersect_line_line[ss][s] = p
                db.points_lines[p][s] = None
            else:
                db.points_lines[p] = OrderedSet.fromkeys([s])
            if p in db.points_circles:
                for c in db.points_circles[p]:
                    db.intersect_line_circle[c][s] = [p]
                    db.intersect_line_circle[s][c] = [p]
        return facts_from_add

    s1_key = db.inverse_eqline[s1]
    s2_key = db.inverse_eqline[s2]
    if s1_key > s2_key:
        s1_key, s2_key = s2_key, s1_key
        s1, s2 = s2, s1
    to_merge = OrderedSet()
    to_merge.update(db.eqline.pop(s2_key))
    for s in db.eqline[s1_key]:
        for ss in to_merge:
            if s == s1 and ss == s2:
                continue
            f = Fact("eqline", [s, ss], "add_fact")
            if s != s1:
                f.add_parent(Fact("eqline", [s, s1]))
            f.add_parent(Fact("eqline", [s1, s2]))
            if ss != s2:
                f.add_parent(Fact("eqline", [s2, ss]))
            facts_from_add.append(f)
    db.eqline[s1_key].update(to_merge)
    for k in to_merge:
        db.inverse_eqline[k] = s1_key
    for p in db.lines_points[s2_key]:
        db.points_lines[p].pop(s2_key)
        for s in db.points_lines[p]:
            if s == s1_key:
                continue
            db.intersect_line_line[s][s1_key] = p
            db.intersect_line_line[s1_key][s] = p
        db.points_lines[p][s1_key] = None
        if p in db.points_circles:
            for c in db.points_circles[p]:
                if s1_key not in db.intersect_line_circle[c]:
                    db.intersect_line_circle[c][s1_key] = []
                    db.intersect_line_circle[s1_key][c] = []
                if p not in db.intersect_line_circle[c][s1_key]:
                    db.intersect_line_circle[c][s1_key].append(p)
                    db.intersect_line_circle[s1_key][c].append(p)
    for s in db.intersect_line_line.pop(s2_key):
        db.intersect_line_line[s].pop(s2_key)
    for c in db.intersect_line_circle.pop(s2_key):
        db.intersect_line_circle[c].pop(s2_key)
    db.lines_points[s1_key].update(db.lines_points.pop(s2_key))
    return facts_from_add
