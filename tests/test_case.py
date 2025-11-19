r"""Test all cases in the case folder."""
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import,eval-used

import glob

import pytest

from tonggeometry.action import Action
from tonggeometry.constructor import *
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import (Fact, fact_transform,
                                                     get_fact_dep)
from tonggeometry.inference_engine.primitives import *


@pytest.mark.parametrize("case_path", sorted(glob.glob("tests/cases/*.txt")))
def test_case(case_path: str):
    """Test one case in the case folder."""
    print(f"Testing {case_path}")
    with open(case_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    d = Diagram(max_depth=100)
    delimiter_count = 0
    context = []
    aux = []
    deps = ""
    all_fact_names = set()
    all_to_names = set()
    all_actions = []
    all_map = {}
    for l in lines:
        print(l)
    for l in lines:
        if l.startswith("#"):
            continue
        if l == '\n':
            delimiter_count += 1
        elif not l.startswith("Fact"):
            action_str = l.strip()
            action = eval(action_str)
            all_map.update(d.order(action)[0])
            order_action = d.order_action(action, all_map, True)
            all_actions += d.new_valid_actions()
            print(order_action)
            assert order_action in all_actions
            d = d.apply_action(order_action)
            if delimiter_count == 0:
                context.append(order_action)
            else:
                aux.append(order_action)
            all_to_names.update(d.to_names_seq[-1])
        else:
            fact_str = l.strip()
            fact = eval(fact_str)
            order_fact = d.order_fact(fact, all_map)
            order_fact = fact_transform(d, order_fact)
            deps += get_fact_dep(order_fact)
            all_fact_names.update(d.used_facts[order_fact][0].dependency)
            assert order_fact in d.used_facts
    pruned_context = d.prune(deps)
    print(context)
    print(pruned_context)
    assert context == pruned_context
    pruned_all = d.prune("".join(all_fact_names))
    print(context + aux)
    print(pruned_all)
    assert context + aux == pruned_all
    if len(aux) > 0:
        assert d.reward > 0
