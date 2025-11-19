r"""Test random action order"""
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import,eval-used

import random
from functools import cmp_to_key, partial

import pytest

from tonggeometry.action import Action, compare
from tonggeometry.constructor import BaseAcuteTriangle
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.primitives import *


@pytest.mark.parametrize("loop_id", range(32))
def test_random_order(loop_id):
    """Test random action order"""
    d = Diagram()
    d = d.apply_action(Action(BaseAcuteTriangle, "", "ABC"))
    all_valid_actions = []
    while True:
        if d.is_terminal:
            break
        new_valid = d.new_valid_actions()
        comparator = partial(compare, new_points=d.to_names_seq[-1])
        assert sorted(new_valid, key=cmp_to_key(comparator)) == new_valid
        all_valid_actions += new_valid
        idx = random.randrange(len(all_valid_actions))
        act = all_valid_actions[idx]
        all_valid_actions = all_valid_actions[idx + 1:]
        try:
            d = d.apply_action(act)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(e)
            print(d.actions)
            print(act)
            break
