r"""Test the eqratio module in forward_chainer."""

from itertools import combinations

from tonggeometry.constructor.primitives import Point
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Fact
from tonggeometry.inference_engine.primitives import Circle, Ratio, Segment
from tonggeometry.inference_engine.rule.eqratio import (
    cong_and_eqratio_to_cong, cong_and_eqratio_to_eqratio,
    eqline_and_eqline_and_eqratio_to_eqratio, eqline_and_eqratio_to_simili,
    eqratio_and_cong_to_cong, eqratio_and_cong_to_eqratio,
    eqratio_and_eqline_and_eqline_to_eqratio, eqratio_and_eqline_to_simili,
    eqratio_and_eqratio_to_eqratio, eqratio_to_cong, eqratio_to_eqratio)


def test_eqratio_to_eqratio():
    """Test eqratio_to_eqratio"""
    diagram = Diagram()
    f = Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("C", "D")),
        Ratio(Segment("E", "F"), Segment("G", "H"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_to_eqratio(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("E", "F")),
        Ratio(Segment("C", "D"), Segment("G", "H"))
    ])


def test_eqratio_to_cong():
    """Test eqratio_to_cong"""
    diagram = Diagram()
    f = Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("C", "D")),
        Ratio(Segment("C", "D"), Segment("B", "A"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("cong", [Segment("A", "B"), Segment("C", "D")])


def test_eqratio_and_cong_to_cong():
    """Test eqratio_and_cong_to_cong"""
    diagram = Diagram()
    diagram.database.add_fact(
        Fact("cong", [Segment("A", "B"), Segment("C", "D")]))
    f = Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("C", "D")),
        Ratio(Segment("E", "F"), Segment("G", "H"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_cong_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("cong", [Segment("E", "F"), Segment("G", "H")])


def test_cong_and_eqratio_to_cong():
    """Test cong_and_eqratio_to_cong"""
    diagram = Diagram()
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment("A", "B"), Segment("C", "D")),
            Ratio(Segment("E", "F"), Segment("G", "H"))
        ]))
    f = Fact("cong", [Segment("A", "B"), Segment("C", "D")])
    diagram.database.add_fact(f)
    new_facts = cong_and_eqratio_to_cong(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("cong", [Segment("E", "F"), Segment("G", "H")])


def test_eqratio_and_eqline_and_eqline_to_eqratio():
    """Test eqratio_and_eqline_and_eqline_to_eqratio"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 0)
    diagram.point_dict["C"] = Point(2, 0)
    diagram.point_dict["D"] = Point(0, 1)
    diagram.point_dict["E"] = Point(1, 1)
    diagram.point_dict["F"] = Point(2, 1)
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "B"), Segment("B", "C")]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("D", "E"), Segment("E", "F")]))
    f = Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("B", "C")),
        Ratio(Segment("D", "E"), Segment("E", "F"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_eqline_and_eqline_to_eqratio(diagram, f)
    assert len(new_facts) == 2
    assert new_facts == [
        Fact("eqratio", [
            Ratio(Segment("C", "B"), Segment("A", "C")),
            Ratio(Segment("F", "E"), Segment("D", "F"))
        ]),
        Fact("eqratio", [
            Ratio(Segment("A", "B"), Segment("A", "C")),
            Ratio(Segment("D", "E"), Segment("D", "F"))
        ])
    ]


def test_eqline_and_eqline_and_eqratio_to_eqratio():
    """Test eqline_and_eqline_and_eqratio_to_eqratio"""
    diagram = Diagram()
    diagram.point_dict["A"] = Point(0, 0)
    diagram.point_dict["B"] = Point(1, 0)
    diagram.point_dict["C"] = Point(2, 0)
    diagram.point_dict["D"] = Point(0, 1)
    diagram.point_dict["E"] = Point(1, 1)
    diagram.point_dict["F"] = Point(2, 1)
    for p1, p2 in combinations("ABCDEF", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment("A", "B"), Segment("B", "C")),
            Ratio(Segment("D", "E"), Segment("E", "F"))
        ]))
    diagram.database.add_fact(
        Fact("eqline",
             [Segment("A", "B"), Segment("B", "C")]))
    f = Fact("eqline", [Segment("D", "E"), Segment("E", "F")])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqline_and_eqratio_to_eqratio(diagram, f)
    assert len(new_facts) == 2
    assert new_facts == [
        Fact("eqratio", [
            Ratio(Segment("C", "B"), Segment("A", "C")),
            Ratio(Segment("F", "E"), Segment("D", "F"))
        ]),
        Fact("eqratio", [
            Ratio(Segment("A", "B"), Segment("A", "C")),
            Ratio(Segment("D", "E"), Segment("D", "F"))
        ])
    ]


def test_eqratio_and_cong_to_eqratio():
    """Test eqratio_and_cong_to_eqratio"""
    diagram = Diagram()
    diagram.database.add_fact(
        Fact("cong", [Segment("A", "B"), Segment("C", "D")]))
    diagram.database.add_fact(
        Fact("cong", [Segment("B", "C"), Segment("D", "F")]))
    diagram.database.add_fact(
        Fact("cong", [Segment("D", "E"), Segment("X", "Y")]))
    diagram.database.add_fact(
        Fact("cong", [Segment("E", "F"), Segment("U", "V")]))
    f = Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("B", "C")),
        Ratio(Segment("D", "E"), Segment("E", "F"))
    ])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_cong_to_eqratio(diagram, f)
    assert len(new_facts) == 8
    assert Fact("eqratio", [
        Ratio(Segment("C", "D"), Segment("B", "C")),
        Ratio(Segment("A", "B"), Segment("B", "C"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment("C", "D"), Segment("B", "C")),
        Ratio(Segment("D", "E"), Segment("E", "F"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("D", "F")),
        Ratio(Segment("A", "B"), Segment("B", "C"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment("A", "B"), Segment("D", "F")),
        Ratio(Segment("D", "E"), Segment("E", "F"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment("X", "Y"), Segment("E", "F")),
        Ratio(Segment("D", "E"), Segment("E", "F"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment("X", "Y"), Segment("E", "F")),
        Ratio(Segment("A", "B"), Segment("B", "C"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment("D", "E"), Segment("U", "V")),
        Ratio(Segment("D", "E"), Segment("E", "F"))
    ]) in new_facts
    assert Fact("eqratio", [
        Ratio(Segment("D", "E"), Segment("U", "V")),
        Ratio(Segment("A", "B"), Segment("B", "C"))
    ]) in new_facts


def test_cong_and_eqratio_to_eqratio():
    """Test cong_and_eqratio_to_eqratio"""
    diagram = Diagram()
    diagram.database.add_fact(
        Fact("eqratio", [
            Ratio(Segment("A", "B"), Segment("C", "D")),
            Ratio(Segment("E", "F"), Segment("A", "B"))
        ]))
    f = Fact("cong", [Segment("A", "B"), Segment("G", "H")])
    diagram.database.add_fact(f)
    new_facts = cong_and_eqratio_to_eqratio(diagram, f)
    assert len(new_facts) == 2
    assert new_facts == [
        Fact("eqratio", [
            Ratio(Segment("A", "B"), Segment("C", "D")),
            Ratio(Segment("G", "H"), Segment("C", "D"))
        ]),
        Fact("eqratio", [
            Ratio(Segment("E", "F"), Segment("A", "B")),
            Ratio(Segment("E", "F"), Segment("G", "H"))
        ])
    ]


def test_eqratio_and_eqratio_to_eqratio():
    """Test eqratio_and_eqratio_to_eqratio"""
    diagram = Diagram()
    a = Segment("A", "B")
    b = Segment("C", "D")
    c = Segment("E", "F")
    d = Segment("G", "H")
    e = Segment("I", "J")
    diagram.database.add_fact(Fact("eqratio", [Ratio(a, b), Ratio(c, d)]))
    f = Fact("eqratio", [Ratio(b, c), Ratio(e, c)])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_eqratio_to_eqratio(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqratio", [Ratio(a, c), Ratio(e, d)])


def test_eqratio_and_eqline_to_simili():
    """Test eqratio_and_eqline_to_simili"""
    diagram = Diagram()
    a = Segment("O", "P")
    b = Segment("o", "P")
    c = Segment("O", "X")
    d = Segment("o", "Y")
    for p1, p2 in combinations("OoPXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.point_dict["O"] = Point(-1, 0)
    diagram.point_dict["o"] = Point(1, 0)
    diagram.point_dict["P"] = Point(0, 0)
    diagram.database.add_fact(Fact("eqline", [a, b]))
    f = Fact("eqratio", [Ratio(a, b), Ratio(c, d)])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_eqline_to_simili(diagram, f)
    assert len(new_facts) == 0
    assert diagram.database.simili == {
        Circle("O", ["X"]): {
            Circle("o", ["Y"]): {
                True: ('P', 'O', 'X', 'o', 'Y'),
                False: None
            }
        },
        Circle("o", ["Y"]): {
            Circle("O", ["X"]): {
                True: ('P', 'o', 'Y', 'O', 'X'),
                False: None
            }
        }
    }

    diagram = Diagram()
    a = Segment("O", "P")
    b = Segment("o", "P")
    c = Segment("O", "X")
    d = Segment("o", "Y")
    for p1, p2 in combinations("OoPXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.point_dict["O"] = Point(-1, 0)
    diagram.point_dict["o"] = Point(1, 0)
    diagram.point_dict["P"] = Point(2, 0)
    diagram.database.add_fact(Fact("eqline", [a, b]))
    f = Fact("eqratio", [Ratio(c, d), Ratio(a, b)])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_eqline_to_simili(diagram, f)
    assert len(new_facts) == 0
    assert diagram.database.simili == {
        Circle("O", ["X"]): {
            Circle("o", ["Y"]): {
                False: ('P', 'O', 'X', 'o', 'Y'),
                True: None
            }
        },
        Circle("o", ["Y"]): {
            Circle("O", ["X"]): {
                False: ('P', 'o', 'Y', 'O', 'X'),
                True: None
            }
        }
    }


def test_eqline_and_eqratio_to_simili():
    """Test eqline_and_eqratio_to_simili"""
    diagram = Diagram()
    a = Segment("O", "P")
    b = Segment("o", "P")
    c = Segment("O", "X")
    d = Segment("o", "Y")
    for p1, p2 in combinations("OoPXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.point_dict["O"] = Point(-1, 0)
    diagram.point_dict["o"] = Point(1, 0)
    diagram.point_dict["P"] = Point(0, 0)
    diagram.database.add_fact(Fact("eqratio", [Ratio(a, b), Ratio(c, d)]))
    f = Fact("eqline", [a, b])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqratio_to_simili(diagram, f)
    assert len(new_facts) == 0
    assert diagram.database.simili == {
        Circle("O", ["X"]): {
            Circle("o", ["Y"]): {
                True: ('P', 'O', 'X', 'o', 'Y'),
                False: None
            }
        },
        Circle("o", ["Y"]): {
            Circle("O", ["X"]): {
                True: ('P', 'o', 'Y', 'O', 'X'),
                False: None
            }
        }
    }

    diagram = Diagram()
    a = Segment("O", "P")
    b = Segment("o", "P")
    c = Segment("O", "X")
    d = Segment("o", "Y")
    for p1, p2 in combinations("OoPXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.point_dict["O"] = Point(-1, 0)
    diagram.point_dict["o"] = Point(1, 0)
    diagram.point_dict["P"] = Point(2, 0)
    diagram.database.add_fact(Fact("eqratio", [Ratio(c, d), Ratio(a, b)]))
    f = Fact("eqline", [b, a])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqratio_to_simili(diagram, f)
    assert len(new_facts) == 0
    assert diagram.database.simili == {
        Circle("O", ["X"]): {
            Circle("o", ["Y"]): {
                False: ('P', 'O', 'X', 'o', 'Y'),
                True: None
            }
        },
        Circle("o", ["Y"]): {
            Circle("O", ["X"]): {
                False: ('P', 'o', 'Y', 'O', 'X'),
                True: None
            }
        }
    }


def test_monge():
    """Test monge"""
    diagram = Diagram()
    a = Segment("O", "P")
    b = Segment("o", "P")
    c = Segment("O", "X")
    d = Segment("o", "Y")
    for p1, p2 in combinations("OoPXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.point_dict["O"] = Point(-1, 0)
    diagram.point_dict["o"] = Point(1, 0)
    diagram.point_dict["P"] = Point(2, 0)
    diagram.database.simili[Circle("O", "X")] = {
        Circle("W", "Z"): {
            True: None,
            False: ("p", "O", "X", "W", "Z")
        }
    }
    diagram.database.simili[Circle("W", "Z")] = {
        Circle("O", "X"): {
            True: None,
            False: ("p", "W", "Z", "O", "X")
        }
    }
    diagram.database.simili[Circle("o", "Y")] = {
        Circle("W", "Z"): {
            True: None,
            False: ("q", "o", "Y", "W", "Z")
        }
    }
    diagram.database.simili[Circle("W", "Z")] = {
        Circle("o", "Y"): {
            True: None,
            False: ("q", "W", "Z", "o", "Y")
        }
    }
    diagram.database.add_fact(Fact("eqline", [a, b]))
    f = Fact("eqratio", [Ratio(a, b), Ratio(c, d)])
    diagram.database.add_fact(f)
    new_facts = eqratio_and_eqline_to_simili(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqline", [Segment(*"Pp"), Segment(*"Pq")])

    diagram = Diagram()
    a = Segment("O", "P")
    b = Segment("o", "P")
    c = Segment("O", "X")
    d = Segment("o", "Y")
    for p1, p2 in combinations("OoPXY", 2):
        diagram.database.add_fact(
            Fact("eqline", [Segment(p1, p2), Segment(p1, p2)]))
    diagram.point_dict["O"] = Point(-1, 0)
    diagram.point_dict["o"] = Point(1, 0)
    diagram.point_dict["P"] = Point(2, 0)
    diagram.database.simili[Circle("O", "X")] = {
        Circle("W", "Z"): {
            False: None,
            True: ("p", "O", "X", "W", "Z")
        }
    }
    diagram.database.simili[Circle("W", "Z")] = {
        Circle("O", "X"): {
            False: None,
            True: ("p", "W", "Z", "O", "X")
        }
    }
    diagram.database.simili[Circle("o", "Y")] = {
        Circle("W", "Z"): {
            False: None,
            True: ("q", "o", "Y", "W", "Z")
        }
    }
    diagram.database.simili[Circle("W", "Z")] = {
        Circle("o", "Y"): {
            False: None,
            True: ("q", "W", "Z", "o", "Y")
        }
    }
    diagram.database.add_fact(Fact("eqratio", [Ratio(c, d), Ratio(a, b)]))
    f = Fact("eqline", [b, a])
    diagram.database.add_fact(f)
    new_facts = eqline_and_eqratio_to_simili(diagram, f)
    assert len(new_facts) == 1
    assert new_facts[0] == Fact("eqline", [Segment(*"Pp"), Segment(*"Pq")])
