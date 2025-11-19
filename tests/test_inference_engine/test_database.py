r"""Test the database module."""
# pylint: disable=no-member

from itertools import combinations

from tonggeometry.inference_engine.database import Database
from tonggeometry.inference_engine.predicate import Predicate
from tonggeometry.inference_engine.primitives import (Angle, Circle, Ratio,
                                                      Segment, Triangle)


def test_eqline():
    """Test the eqline handler with no para, perp, eqangle."""
    facts = set()
    predicates = [
        Predicate("eqline", ["A", "B", "B", "C"]),
        Predicate("eqline", ["C", "B", "B", "A"]),
        Predicate("eqline", ["A", "B", "D", "E"]),
        Predicate("eqline", ["E", "F", "G", "H"]),
        Predicate("eqline", ["A", "B", "E", "F"])
    ]
    db = Database()
    db.add_fact(Predicate("eqline", [Segment(*"AB"), Segment(*"AB")]))
    db.add_fact(Predicate("eqline", [Segment(*"BC"), Segment(*"BC")]))
    db.add_fact(Predicate("eqline", [Segment(*"DE"), Segment(*"DE")]))
    db.add_fact(Predicate("eqline", [Segment(*"EF"), Segment(*"EF")]))
    db.add_fact(Predicate("eqline", [Segment(*"GH"), Segment(*"GH")]))

    s0 = {
        Segment(*"AB"): {
            Segment(*"BC"): "B"
        },
        Segment(*"BC"): {
            Segment(*"AB"): "B"
        },
        Segment(*"DE"): {
            Segment(*"EF"): "E"
        },
        Segment(*"EF"): {
            Segment(*"DE"): "E"
        },
        Segment(*"GH"): {}
    }
    s1 = {
        Segment(*"AB"): {},
        Segment(*"DE"): {
            Segment(*"EF"): "E"
        },
        Segment(*"EF"): {
            Segment(*"DE"): "E"
        },
        Segment(*"GH"): {}
    }
    s3 = {
        Segment(*"AB"): {
            Segment(*"EF"): "E"
        },
        Segment(*"EF"): {
            Segment(*"AB"): "E"
        },
        Segment(*"GH"): {}
    }
    s4 = {
        Segment(*"AB"): {
            Segment(*"EF"): "E"
        },
        Segment(*"EF"): {
            Segment(*"AB"): "E"
        }
    }
    s5 = {Segment(*"AB"): {}}
    assert db.intersect_line_line == s0
    for predicate, num, s in zip(predicates, [0, 0, 1, 0, 5],
                                 [s1, s1, s3, s4, s5]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_from_add = db.add_fact(fact)
        assert len(facts_from_add) == num
        assert db.intersect_line_line == s
    assert len(db.eqline) == 1
    assert len(db.inverse_eqline) == 5
    for l, name in zip(db.eqline[db.inverse_eqline[Segment(*"AB")]],
                       ["AB", "BC", "DE", "EF", "GH"]):
        assert str(l) == name
    assert all(str(l) == "AB" for l in db.inverse_eqline.values())
    assert len(db.points_lines) == 8
    assert all(
        len(value) == 1 and list(value) == [Segment(*"AB")]
        for value in db.points_lines.values())
    assert len(db.lines_points) == 1
    assert len(db.lines_points[Segment(*"AB")]) == 8
    print(db)


def test_eqcircle():
    """Test the eqcircle handler."""
    facts = set()
    predicates = [
        Predicate("eqcircle", [["O", "A"], [None, "C", "D", "E"]]),
        Predicate("eqcircle", [[None, "C", "D", "E"], ["O", "A"]]),
        Predicate("eqcircle", [["O", "A"], [None, "D", "E", "F"]]),
        Predicate("eqcircle", [[None, "U", "V", "W"], [None, "U", "V", "Z"]]),
        Predicate("eqcircle", [[None, "U", "V", "W"], [None, "D", "E", "F"]]),
    ]
    db = Database()
    points = "OACDEFUVWZ"
    for p1, p2 in combinations(points, 2):
        db.add_fact(Predicate("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    for predicate, num in zip(predicates, [0, 0, 1, 0, 5]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_from_add = db.add_fact(fact)
        assert len(facts_from_add) == num
    assert len(db.eqcircle) == 1
    assert len(db.inverse_eqcircle) == 5
    for c, name in zip(
            db.eqcircle[Circle("O", ["A"])],
        ["O, A", "None, CDE", "None, DEF", "None, UVW", "None, UVZ"]):
        assert str(c) == name
    assert all(str(c) == "O, A" for c in db.inverse_eqcircle.values())
    assert len(db.points_circles) == 9
    assert len(db.centers_circles) == 1
    assert len(db.circles_circles) == 1
    assert all(
        len(value) == 1 and list(value) == [Circle("O", ["A"])]
        for value in db.points_circles.values())
    assert list(db.centers_circles["O"]) == [Circle("O", ["A"])]
    assert db.circles_circles[Circle("O",
                                     ["A"])] == Circle("O", [*"ACDEFUVWZ"])
    print(db)


def test_eqcircle_center_merge():
    """Test the eqcircle handler with center merge."""
    facts = set()
    predicates = [
        Predicate("eqcircle", [[None, "A", "B", "C"], [None, "C", "D", "E"]]),
        Predicate("eqcircle", [["O", "A"], [None, "A", "B", "C"]]),
    ]
    db = Database()
    points = "OABCDE"
    for p1, p2 in combinations(points, 2):
        db.add_fact(Predicate("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    for predicate, num in zip(predicates, [0, 1]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_from_add = db.add_fact(fact)
        assert len(facts_from_add) == num
    assert len(db.points_circles) == 5
    assert all(
        len(value) == 1 and list(value) == [Circle(None, [*"ABC"])]
        for value in db.points_circles.values())
    assert len(db.centers_circles) == 1
    c = list(db.centers_circles["O"])[0]
    assert db.circles_circles[c] == Circle("O", [*"ABCDE"])
    print(db)


def test_eqcircle_center_clean():
    """Test the eqcircle handler with center cleaning."""
    facts = set()
    predicates = [
        Predicate("eqcircle", [["O", "A"], ["O", "B"]]),
        Predicate("eqcircle", [["O", "C"], ["O", "D"]]),
        Predicate("eqcircle", [["O", "A"], ["O", "C"]]),
    ]
    db = Database()
    points = "OABCD"
    for p1, p2 in combinations(points, 2):
        db.add_fact(Predicate("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    for predicate, num in zip(predicates, [0, 0, 3]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_from_add = db.add_fact(fact)
        assert len(facts_from_add) == num
    assert len(db.points_circles) == 4
    assert all(
        len(value) == 1 and list(value) == [Circle("O", ["A"])]
        for value in db.points_circles.values())
    assert len(db.centers_circles) == 1
    c = list(db.centers_circles["O"])[0]
    assert db.circles_circles[c] == Circle("O", [*"ABCD"])
    print(db)


def test_intersections():
    """Test eqline and eqcircle's intersection handling."""
    facts = set()
    predicates = [
        Predicate("eqline", ["A", "B", "A", "B"]),
        Predicate("eqline", ["A", "C", "A", "C"]),
        Predicate("eqline", ["B", "C", "B", "C"]),
        Predicate("eqline", ["A", "B", "A", "C"]),
        Predicate("eqline", ["A", "B", "B", "C"]),
        Predicate("eqline", ["O", "A", "O", "A"]),
        Predicate("eqline", ["O", "B", "O", "B"]),
        Predicate("eqline", ["O", "C", "O", "C"]),
        Predicate("eqcircle", [["O", "A"], ["O", "B"]]),
        Predicate("eqline", ["D", "A", "D", "A"]),
        Predicate("eqline", ["D", "B", "D", "B"]),
        Predicate("eqline", ["D", "C", "D", "C"]),
        Predicate("eqline", ["D", "O", "D", "O"]),
        Predicate("eqline", ["A", "B", "A", "D"]),
        Predicate("eqline", ["A", "B", "B", "D"]),
        Predicate("eqline", ["A", "B", "C", "D"]),
        Predicate("eqline", ["M", "A", "M", "A"]),
        Predicate("eqline", ["M", "B", "M", "B"]),
        Predicate("eqline", ["M", "C", "M", "C"]),
        Predicate("eqline", ["M", "O", "M", "O"]),
        Predicate("eqline", ["M", "D", "M", "D"]),
        Predicate("eqcircle", [["M", "A"], ["M", "D"]]),
    ]
    db = Database()
    for predicate, num in zip(
            predicates,
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 0, 0, 0, 0, 0, 0]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_from_add = db.add_fact(fact)
        assert len(facts_from_add) == num
    print(db.intersect_line_circle[Circle("M", ["A"])][Segment("A", "B")])
    assert len(db.intersect_line_circle[Circle("O", ["A"])][Segment("A",
                                                                    "B")]) == 2
    assert len(db.intersect_line_circle[Circle("M", ["A"])][Segment("A",
                                                                    "B")]) == 2
    assert len(db.intersect_circle_circle) == 2
    assert db.intersect_circle_circle[Circle("O", ["A"])] == {
        Circle("M", ["A"]): ["A"]
    }
    assert db.intersect_circle_circle[Circle("M", ["A"])] == {
        Circle("O", ["A"]): ["A"]
    }
    print(db)


def test_cong():
    """Test the cong handler."""
    facts = set()
    predicates = [
        Predicate("cong", ["B", "A", "D", "C"]),
        Predicate("cong", ["D", "C", "A", "B"]),
        Predicate("cong", ["A", "B", "F", "E"]),
        Predicate("cong", ["Q", "P", "S", "M"]),
        Predicate("cong", ["Q", "P", "A", "B"]),
    ]
    db = Database()
    for predicate, num in zip(predicates, [0, 0, 1, 0, 5]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_from_add = db.add_fact(fact)
        assert len(facts_from_add) == num
    assert len(db.cong) == 1
    assert len(db.inverse_cong) == 5
    for s, name in zip(db.cong[Segment(*"AB")],
                       ["AB", "CD", "EF", "MS", "PQ"]):
        assert str(s) == name
    assert all(str(s) == "AB" for s in db.inverse_cong.values())
    assert len(db.points_congs) == 10
    assert all(len(value) == 1 for value in db.points_congs.values())
    for p in "ABCDEFPQSM":
        assert len(db.points_congs[p]) == 1
    print(db)


def test_midp():
    """Test the midp handler."""
    facts = set()
    predicates = [
        Predicate("midp", ["M", "A", "B"]),
        Predicate("midp", ["M", "C", "D"]),
        Predicate("midp", ["E", "A", "C"])
    ]
    db = Database()
    for predicate, num in zip(predicates, [0, 0, 0]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_from_add = db.add_fact(fact)
        assert len(facts_from_add) == num
    assert len(db.midp) == 2
    assert len(db.inverse_midp) == 3
    assert list(db.midp["M"]) == [Segment(*"AB"), Segment(*"CD")]
    assert list(db.midp["E"]) == [Segment(*"AC")]
    assert all(db.inverse_midp[s] == "M"
               for s in [Segment(*"AB"), Segment(*"CD")])
    assert db.inverse_midp[Segment(*"AC")] == "E"
    assert len(db.points_midps) == 4
    assert list(db.points_midps["A"]) == [Segment(*"AB"), Segment(*"AC")]
    assert list(db.points_midps["C"]) == [Segment(*"CD"), Segment(*"AC")]
    assert len(db.points_midps["A"]) == 2
    assert len(db.points_midps["B"]) == 1
    assert len(db.points_midps["C"]) == 2
    assert len(db.points_midps["D"]) == 1
    print(db)


def test_para():
    """Test the para handler."""
    facts = set()
    predicates = [
        Predicate("para", ["A", "B", "E", "D"]),
        Predicate("para", ["E", "D", "B", "A"]),
        Predicate("para", ["B", "A", "C", "H"]),
        Predicate("para", ["U", "V", "C", "Y"]),
        Predicate("para", ["C", "Y", "B", "A"]),
    ]
    db = Database()
    for s_name in ["AB", "ED", "CH", "UV", "CY"]:
        s = Segment(*s_name)
        db.add_fact(Predicate("eqline", [s, s]))
    for predicate, num in zip(predicates, [0, 0, 1, 0, 5]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_to_add = db.add_fact(fact)
        assert len(facts_to_add) == num
    assert len(db.para) == 1
    assert len(db.inverse_para) == 5
    for l, name in zip(db.para[Segment(*"AB")],
                       ["AB", "DE", "CH", "CY", "UV"]):
        assert str(l) == name
    assert all(str(l) == "AB" for l in db.inverse_para.values())
    for p in "ABEDCHUVY":
        if p == "C":
            assert len(db.points_paras[p]) == 2
        else:
            assert len(db.points_paras[p]) == 1
    print(db)


def test_perp():
    """Test the perp handler."""
    facts = set()
    predicates = [
        Predicate("perp", ["A", "B", "E"]),
        Predicate("perp", ["E", "D", "B"]),
        Predicate("perp", ["D", "E", "G"]),
        Predicate("perp", ["C", "H", "U"]),
        Predicate("perp", ["A", "B", "C"])
    ]
    db = Database()
    for s_name in ["AB", "BE", "ED", "BD", "EG", "CH", "HU", "BC"]:
        s = Segment(*s_name)
        db.add_fact(Predicate("eqline", [s, s]))
    for predicate, num in zip(predicates, [0, 0, 0, 0, 0]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_to_add = db.add_fact(fact)
        assert len(facts_to_add) == num
    assert len(db.perp) == 5
    assert len(db.segments_perps) == 8
    assert len(db.h_segments_perps) == 5
    for p in "ABEDGCHU":
        if p == "B":
            assert len(db.points_perps[p]) == 4
        elif p == "E":
            assert len(db.points_perps[p]) == 3
        elif p in ["C", "D", "H"]:
            assert len(db.points_perps[p]) == 2
        else:
            assert len(db.points_perps[p]) == 1
    print(db)


def test_eqangle():
    """Test the eqangle handler."""
    facts = set()
    predicates = [
        Predicate("eqangle", ["B", "A", "C", "D", "F", "E"]),
        Predicate("eqangle", ["E", "F", "D", "C", "A", "B"]),
        Predicate("eqangle", ["C", "A", "B", "X", "U", "V"]),
        Predicate("eqangle", ["I", "J", "L", "K", "O", "P"]),
        Predicate("eqangle", ["B", "A", "C", "I", "J", "L"])
    ]
    db = Database()
    for predicate, num in zip(predicates, [0, 0, 1, 0, 5]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_to_add = db.add_fact(fact)
        assert len(facts_to_add) == num
    assert len(db.eqangle) == 1
    assert len(db.inverse_eqangle) == 5
    for a, name in zip(db.eqangle[db.inverse_eqangle[Angle(*"BAC")]],
                       ["BAC", "DFE", "VUX", "IJL", "KOP"]):
        assert a.name == name
    assert all(a.name == "BAC" for a in db.inverse_eqangle.values())
    for s in db.segments_eqangles:
        assert len(db.segments_eqangles[s]) == 1
    for p in db.points_eqangles:
        if p in ["A", "F", "U", "J", "O"]:
            assert len(db.points_eqangles[p]) == 2
        else:
            assert len(db.points_eqangles[p]) == 1
    print(db)


def test_eqratio():
    """Test the eqratio handler."""
    facts = set()
    predicates = [
        Predicate("eqratio", ["B", "A", "C", "D", "F", "E", "H", "G"]),
        Predicate("eqratio", ["H", "G", "E", "F", "D", "C", "A", "B"]),
        Predicate("eqratio", ["A", "B", "C", "D", "U", "V", "W", "X"]),
        Predicate("eqratio", ["I", "J", "L", "K", "O", "P", "N", "M"]),
        Predicate("eqratio", ["A", "B", "C", "D", "I", "J", "L", "K"])
    ]
    db = Database()
    for predicate, num in zip(predicates, [0, 0, 1, 0, 5]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts.add(fact)
        facts_to_add = db.add_fact(fact)
        assert len(facts_to_add) == num
    assert len(db.eqratio) == 1
    assert len(db.inverse_eqratio) == 5
    for r, name in zip(
            db.eqratio[db.inverse_eqratio[Ratio(Segment(*"AB"),
                                                Segment(*"CD"))]],
        ["AB, CD", "EF, GH", "UV, WX", "IJ, KL", "OP, MN"]):
        assert r.name == name
    assert all(r.name == "AB, CD" for r in db.inverse_eqratio.values())
    assert len(db.segments_eqratios) == 10
    assert all(len(value) == 1 for value in db.segments_eqratios.values())
    for s in db.segments_eqratios:
        for r in db.segments_eqratios[s]:
            another_r = db.eqratio[db.inverse_eqratio[r]][r]
            assert another_r.name == r.name
    for p in "ABCDEFGHUVWXIJKLOPMN":
        assert len(db.points_eqratios[p]) == 1
    print(db)


def test_simtri():
    """Test the simtri handler."""
    facts = set()
    predicates = [
        Predicate("simtri", ["A", "B", "C", "P", "Q", "R"]),
        Predicate("simtri", ["A", "C", "B", "P", "R", "Q"]),
        Predicate("simtri", ["C", "B", "A", "E", "F", "D"]),
        Predicate("simtri", ["A", "D", "E", "W", "C", "E"]),
        Predicate("simtri", ["A", "C", "B", "D", "E", "A"]),
    ]
    db = Database()
    for predicate, num in zip(predicates, [0, 0, 1, 0, 5]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts_to_add = db.add_fact(fact)
        facts.add(fact)
        assert len(facts_to_add) == num
    assert len(db.simtri) == 1
    assert len(db.inverse_simtri) == 5
    for t, name in zip(db.simtri[db.inverse_simtri[Triangle(*"ABC")]],
                       ["ABC", "PQR", "DFE", "DAE", "CWE"]):
        assert t.name == name
    assert all(t.name == "ABC" for t in db.inverse_simtri.values())
    assert len(db.segments_simtris) == 3 * 5 - 1
    assert len(db.segments_simtris[Segment(*"DE")]) == 2
    for s in db.segments_simtris:
        for t in db.segments_simtris[s]:
            another_t = db.simtri[db.inverse_simtri[t]][t]
            assert another_t.name == t.name
    for p in "ABCPQRDEWF":
        if p == "E":
            assert len(db.points_simtris[p]) == 5
        elif p in ["A", "C"]:
            assert len(db.points_simtris[p]) == 4
        elif p == "D":
            assert len(db.points_simtris[p]) == 3
        else:
            assert len(db.points_simtris[p]) == 2
    print(db)


def test_contri():
    """Test the contri handler."""
    facts = set()
    predicates = [
        Predicate("contri", ["A", "B", "C", "P", "Q", "R"]),
        Predicate("contri", ["A", "C", "B", "P", "R", "Q"]),
        Predicate("contri", ["C", "B", "A", "E", "F", "D"]),
        Predicate("contri", ["A", "D", "E", "W", "C", "E"]),
        Predicate("contri", ["A", "C", "B", "D", "E", "A"]),
    ]
    db = Database()
    for predicate, num in zip(predicates, [0, 0, 1, 0, 5]):
        fact = db.predicate_to_fact(predicate)
        if fact in facts:
            continue
        facts_to_add = db.add_fact(fact)
        facts.add(fact)
        assert len(facts_to_add) == num
    assert len(db.contri) == 1
    assert len(db.inverse_contri) == 5
    for t, name in zip(db.contri[db.inverse_contri[Triangle(*"ABC")]],
                       ["ABC", "PQR", "DFE", "DAE", "CWE"]):
        assert t.name == name
    assert all(t.name == "ABC" for t in db.inverse_contri.values())
    assert len(db.segments_contris) == 3 * 5 - 1
    assert len(db.segments_contris[Segment(*"DE")]) == 2
    for s in db.segments_contris:
        for t in db.segments_contris[s]:
            another_t = db.contri[db.inverse_contri[t]][t]
            assert another_t.name == t.name
    for p in "ABCPQRDEWF":
        if p == "E":
            assert len(db.points_contris[p]) == 5
        elif p in ["A", "C"]:
            assert len(db.points_contris[p]) == 4
        elif p == "D":
            assert len(db.points_contris[p]) == 3
        else:
            assert len(db.points_contris[p]) == 2
    print(db)
