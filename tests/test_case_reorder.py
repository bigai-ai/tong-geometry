r"""Reorder all test cases into the normalized form."""
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import,eval-used

import glob
import os
import time
from collections import defaultdict, namedtuple
from functools import cmp_to_key, partial

import numpy as np
import pytest

from tonggeometry.action import Action, action_to_string, compare
from tonggeometry.constructor import *
from tonggeometry.constructor.primitives import Circle as RealCircle
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import (Fact, fact_transform,
                                                     get_fact_dep)
from tonggeometry.inference_engine.primitives import *

Dependency = namedtuple("Dependency", ["points", "circles"])


def compare_actions(tpl1, tpl2, constructor, new_points):
    """Compare actions of the same constructor."""
    if constructor.lt(tpl1[0].from_names, tpl2[0].from_names, new_points):
        return -1
    return 1


def circle_satisfy(c, d):
    """Check if the circle c is satisfied in d."""
    for name in d.circle_dict.keys():
        cc = Circle(name[0], [name[1]])
        if d.database.is_eqcircle(c, cc):
            return True
    return False


def point_sort(dd, input_actions, circle_dict, all_map, d):
    """Sort the actions in the action_dict for dd based on the normalization."""
    output_order_actions = []
    if len(input_actions) == 0:
        return output_order_actions
    action_dict = defaultdict(lambda: [])
    for action in input_actions:
        action_dict[action.constructor_name].append(action)
    sorted_action_dict_keys = sorted(action_dict.keys(),
                                     key=lambda x: ConstructorIndex[x])
    for sorted_key in sorted_action_dict_keys:
        action_list = action_dict[sorted_key]
        mapped_from_names = []
        constructor = action_list[0].constructor
        for action in action_list:
            inv_to = False
            old_from_names = action.from_names
            if sorted_key in ["AnyArc", "MidArc"]:
                circle_center = all_map[old_from_names[2]]
                circle_p = all_map[old_from_names[0]]
                if not circle_satisfy(Circle(circle_center, [circle_p]), dd):
                    circle_dict[circle_center][circle_p][sorted_key].append(
                        action)
                    continue
                action.from_names = "".join(all_map[char]
                                            for char in old_from_names)
                action.convert(dd)
            elif sorted_key == "IntersectLineCircleOn":
                circle_center = all_map[old_from_names[2]]
                circle_p = all_map[old_from_names[1]]
                if not circle_satisfy(Circle(circle_center, [circle_p]), dd):
                    circle_dict[circle_center][circle_p][sorted_key].append(
                        action)
                    continue
                action.from_names = "".join(all_map[char]
                                            for char in old_from_names)
                action.convert(dd)
            elif sorted_key == "IntersectLineCircleOff":
                circle_center = all_map[old_from_names[2]]
                circle_p = all_map[old_from_names[3]]
                if not circle_satisfy(Circle(circle_center, [circle_p]), dd):
                    circle_dict[circle_center][circle_p][sorted_key].append(
                        action)
                    continue
                action.from_names = "".join(all_map[char]
                                            for char in old_from_names)
                action.convert(dd)
                if (d.point_dict[old_from_names[0]] -
                        d.point_dict[old_from_names[1]]
                    ).dot(dd.point_dict[action.from_names[0]] -
                          dd.point_dict[action.from_names[1]]) < 0:
                    inv_to = True
            elif sorted_key == "IntersectCircleCircle":
                good = True
                circle_center = all_map[old_from_names[0]]
                circle_p = all_map[old_from_names[1]]
                if not circle_satisfy(Circle(circle_center, [circle_p]), dd):
                    circle_dict[circle_center][circle_p][sorted_key].append(
                        action)
                    good = False
                circle_center = all_map[old_from_names[2]]
                circle_p = all_map[old_from_names[3]]
                if not circle_satisfy(Circle(circle_center, [circle_p]), dd):
                    circle_dict[circle_center][circle_p][sorted_key].append(
                        action)
                    good = False
                if not good:
                    continue
                if all_map[old_from_names[0]] > all_map[old_from_names[2]]:
                    inv_to = True
                action.from_names = "".join(all_map[char]
                                            for char in old_from_names)
                action.convert(dd)
            else:
                action.from_names = "".join(all_map[char]
                                            for char in old_from_names)
                action.convert(dd)
            mapped_from_names.append((action, inv_to))
        comparator = partial(compare_actions,
                             constructor=constructor,
                             new_points=dd.to_names_seq[-1])
        mapped_from_names.sort(key=cmp_to_key(comparator))
        output_order_actions.extend(mapped_from_names)
    return output_order_actions


def circle_sort(dd, circle, circle_dict, all_map, d, aux_dict=None):
    """Sort the actions in the circle_dict for dd based on the normalization."""
    output_order_actions = []
    if circle is None or len(circle) == 0:  # pylint: disable=too-many-nested-blocks
        actions_dict = defaultdict(lambda: defaultdict(lambda: []))
        for center in list(circle_dict.keys()):
            pp_dict = circle_dict[center]
            for pp in list(pp_dict.keys()):
                the_circle = Circle(center, [pp])
                if not circle_satisfy(the_circle, dd):
                    continue
                for name, val in pp_dict.pop(pp).items():
                    for action in val:
                        inv_to = False
                        old_from_names = action.from_names
                        if name == "IntersectCircleCircle" and (  # pylint: disable=too-many-boolean-expressions
                                not (circle_satisfy(
                                    Circle(all_map[old_from_names[0]],
                                           [all_map[old_from_names[1]]]), dd)
                                     and circle_satisfy(
                                         Circle(all_map[old_from_names[2]],
                                                [all_map[old_from_names[3]]]),
                                         dd)) or
                            (all_map[old_from_names[0]] in circle_dict
                             and all_map[old_from_names[1]]
                             in circle_dict[old_from_names[0]]) or
                            (all_map[old_from_names[2]] in circle_dict
                             and all_map[old_from_names[3]]
                             in circle_dict[old_from_names[2]])):
                            continue
                        action.from_names = "".join(all_map[char]
                                                    for char in old_from_names)
                        action.convert(dd)
                        if name == "IntersectLineCircleOff" and (
                                d.point_dict[old_from_names[0]] -
                                d.point_dict[old_from_names[1]]
                        ).dot(dd.point_dict[action.from_names[0]] -
                              dd.point_dict[action.from_names[1]]) < 0:
                            inv_to = True
                        if name == "IntersectCircleCircle" and all_map[
                                old_from_names[0]] > all_map[
                                    old_from_names[2]]:
                            inv_to = True
                        level = aux_dict[max(action.from_names)][1]
                        actions_dict[level][name].append((action, inv_to))
            if len(pp_dict) == 0:
                circle_dict.pop(center)
        for level in sorted(actions_dict.keys()):
            for name in sorted(actions_dict[level].keys(),
                               key=lambda x: ConstructorIndex[x]):
                action_list = actions_dict[level][name]
                constructor = action_list[0][0].constructor
                new_points = aux_dict[max(action_list[0][0].from_names)][0]
                comparator = partial(compare_actions,
                                     constructor=constructor,
                                     new_points=new_points)
                output_order_actions.extend(
                    sorted(action_list, key=cmp_to_key(comparator)))
        return output_order_actions
    center, p = circle
    if center not in circle_dict:
        return output_order_actions
    the_circle = Circle(center, [p])
    action_dict = defaultdict(lambda: [])
    for pp in list(circle_dict[center].keys()):
        if not dd.database.is_eqcircle(the_circle, Circle(center, [pp])):
            continue
        for key, val in circle_dict[center].pop(pp).items():
            action_dict[key].extend(val)
    if len(circle_dict[center]) == 0:
        circle_dict.pop(center)
    sorted_action_dict_keys = sorted(action_dict.keys(),
                                     key=lambda x: ConstructorIndex[x])
    for sorted_key in sorted_action_dict_keys:
        action_list = action_dict[sorted_key]
        mapped_from_names = []
        constructor = action_list[0].constructor
        for action in action_list:
            inv_to = False
            old_from_names = action.from_names
            if sorted_key == "IntersectCircleCircle" and not (circle_satisfy(
                    Circle(all_map[old_from_names[0]],
                           [all_map[old_from_names[1]]]),
                    dd) and circle_satisfy(
                        Circle(all_map[old_from_names[2]],
                               [all_map[old_from_names[3]]]), dd)):
                continue
            action.from_names = "".join(all_map[char]
                                        for char in old_from_names)
            action.convert(dd)
            if sorted_key == "IntersectLineCircleOff" and (
                    d.point_dict[old_from_names[0]] -
                    d.point_dict[old_from_names[1]]
            ).dot(dd.point_dict[action.from_names[0]] -
                  dd.point_dict[action.from_names[1]]) < 0:
                inv_to = True
            if sorted_key == "IntersectCircleCircle" and all_map[
                    old_from_names[0]] > all_map[old_from_names[2]]:
                inv_to = True
            mapped_from_names.append((action, inv_to))
        comparator = partial(compare_actions,
                             constructor=constructor,
                             new_points=dd.to_names_seq[-1])
        mapped_from_names.sort(key=cmp_to_key(comparator))
        output_order_actions.extend(mapped_from_names)
    return output_order_actions


def get_dependency(action):  # pylint: disable=redefined-outer-name
    """Get dependency of actions."""
    points = "".join(sorted(set(action.from_names)))
    circles = ()
    if action.constructor_name == "AnyArc":
        c = Circle(action.from_names[-1], [action.from_names[0]])
        circles += (c, )
    elif action.constructor_name == "MidArc":
        c = Circle(action.from_names[-1], [action.from_names[0]])
        circles += (c, )
    elif action.constructor_name == "IntersectLineCircleOn":
        c = Circle(action.from_names[-1], [action.from_names[-2]])
        circles += (c, )
    elif action.constructor_name == "IntersectLineCircleOff":
        c = Circle(action.from_names[-2], [action.from_names[-1]])
        circles += (c, )
    elif action.constructor_name == "IntersectCircleCircle":
        c = Circle(action.from_names[0], [action.from_names[1]])
        circles += (c, )
        c = Circle(action.from_names[2], [action.from_names[3]])
        circles += (c, )
    return Dependency(points=points, circles=circles)


def dependency_satisfy(dep, diagram):
    """Check if dep satisfies requirements in diagram."""
    for point in dep.points:
        if point not in diagram.database.points_lines:
            return False
    for c in dep.circles:
        real_c = RealCircle(diagram.point_dict[c.center],
                            diagram.point_dict[c.min_point])
        found = False
        for name, dict_c in diagram.circle_dict.items():
            cc = Circle(name[0], [name[1]])
            if real_c == dict_c and diagram.database.is_eqcircle(c, cc):
                found = True
                break
        if not found:
            return False
    return True


def tuple_compare(tuple1, tuple2, new_points):
    """Compare tuples based on the first element."""
    a1, _ = tuple1
    a2, _ = tuple2
    return compare(a1, a2, new_points)


@pytest.mark.parametrize("case_path", sorted(glob.glob("tests/cases/*.txt")))
def test_case_reorder(case_path):
    """Test case when reordered."""
    with open(case_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    all_actions = {}
    all_facts = []
    counter = 0
    delimit = 0
    old_d = Diagram(max_depth=100)
    for l in lines:
        l = l.strip()
        if l.startswith("Action"):
            action = eval(l)
            old_d = old_d.apply_action(action)
            all_actions[counter] = action
            counter += 1
        elif l.startswith("Fact"):
            fact = eval(l)
            all_facts.append(fact)
        elif l == "":
            delimit += 1
    if len(all_actions) == 0:
        return
    # # init
    # all_map = {}
    # d = Diagram(max_depth=100)
    # base_action = all_actions.pop(0)
    # exec_actions = [base_action]
    # all_valid_actions = [Action(BaseAcuteTriangle, "", "ABC")]
    # # loop
    # steps = 0
    # while exec_actions:
    #     exec_action = exec_actions.pop(0)
    #     all_map.update(d.order(exec_action)[0])
    #     print(exec_action)
    #     order_action = d.order_action(exec_action, all_map, True)
    #     print(order_action)
    #     print(all_map)
    #     # # when only reading context actions, check if simplest form
    #     # if exec_action.from_names:
    #     #     assert max(order_action.from_names) == max(
    #     #         all_map[old_char] for old_char in exec_action.from_names)
    #     assert order_action in all_valid_actions
    #     order_action_idx = all_valid_actions.index(order_action)
    #     all_valid_actions = all_valid_actions[order_action_idx + 1:]
    #     d = d.apply_action(order_action)
    #     steps += 1
    #     all_valid_actions += d.new_valid_actions()
    #     keys_to_pop = []
    #     focus = []
    #     for key, old_action in all_actions.items():
    #         if not set(old_action.from_names) <= set(all_map.keys()):
    #             continue
    #         tmp_all_map = dict(all_map.items())
    #         if old_action.to_names:
    #             for to_char in old_action.to_names:
    #                 tmp_all_map[to_char] = to_char
    #         new_action = d.order_action(old_action, tmp_all_map, True)
    #         new_action_deps = get_dependency(new_action)
    #         if dependency_satisfy(new_action_deps, d):
    #             keys_to_pop.append(key)
    #             focus.append((new_action, old_action))
    #     for key_to_pop in keys_to_pop:
    #         all_actions.pop(key_to_pop)
    #     comparator = partial(tuple_compare, new_points=d.to_names_seq[-1])
    #     sorted_focus = sorted(focus, key=cmp_to_key(comparator))
    #     exec_actions += [x[1] for x in sorted_focus]
    # assert d.depth == old_d.depth

    dd = Diagram(max_depth=100)
    all_map = {}
    inv_map = {}
    exec_row_to_action = {}
    exec_matrix = np.zeros((len(old_d.actions) - 1, len(dd.all_names) + 1),
                           dtype=bool)
    for idx, d_action in enumerate(old_d.actions[1:]):
        d_action_with_to = Action(d_action.constructor, d_action.from_names,
                                  old_d.to_names_seq[idx + 1])
        exec_row_to_action[idx] = d_action_with_to
        row = idx
        for char in d_action_with_to.from_names:
            if char >= "a":
                col = ord(char) - ord("a") + 26
            else:
                col = ord(char) - ord("A")
            exec_matrix[row][col] = True
    # base action never changes in both d and dd
    d_action = old_d.actions[0]
    base_action = Action(d_action.constructor, d_action.from_names,
                         old_d.to_names_seq[0])
    # actions in buffer should have sorted actions with mapped from_name
    exec_buffer = [(base_action, False)]
    exec_circle_actions = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [])))
    points_to_new_points = {}
    while exec_buffer:
        exec_action, inv_to = exec_buffer.pop(0)
        rep_mapping, inv_mapping = dd.order(exec_action)
        all_map.update(rep_mapping)
        inv_map.update(inv_mapping)
        old_to_names = exec_action.to_names
        new_to_names = "".join(all_map[char] for char in old_to_names)
        exec_action.to_names = new_to_names
        if inv_to:
            all_map[old_to_names[0]] = new_to_names[1]
            all_map[old_to_names[1]] = new_to_names[0]
            inv_map[new_to_names[0]] = old_to_names[1]
            inv_map[new_to_names[1]] = old_to_names[0]
        dd = dd.apply_action(exec_action)
        for new_to_name in new_to_names:
            points_to_new_points[new_to_name] = (new_to_names, dd.depth)
        if exec_action.constructor_name != "CenterCircle":
            cols = []
            for new_to_name in new_to_names:
                to_name = inv_map[new_to_name]
                if to_name >= "a":
                    col = ord(to_name) - ord("a") + 26
                else:
                    col = ord(to_name) - ord("A")
                cols.append(col)
            exec_buffer.extend(
                circle_sort(dd, None, exec_circle_actions, all_map, old_d,
                            points_to_new_points))
            exec_matrix[:, cols] = False
            exec_actions_rows = np.where(
                np.count_nonzero(exec_matrix, axis=-1) == 0)[0]
            exec_matrix[exec_actions_rows, -1] = True
            exec_actions = [
                exec_row_to_action[row_idx] for row_idx in exec_actions_rows
            ]
            exec_buffer.extend(
                point_sort(dd, exec_actions, exec_circle_actions, all_map,
                           old_d))
        else:
            exec_buffer.extend(
                circle_sort(dd, exec_action.from_names, exec_circle_actions,
                            all_map, old_d))
    assert dd.depth == old_d.depth

    for idx, fact in enumerate(all_facts):
        order_fact = dd.order_fact(fact, all_map)
        order_fact = fact_transform(dd, order_fact)
        assert order_fact in dd.used_facts
        context_deps = get_fact_dep(order_fact)
        proof_deps = dd.used_facts[order_fact][0].dependency
        context = dd.prune(context_deps)
        proof = dd.prune(proof_deps)
        # dd = Diagram(mode=0)
        # all_timestamp = []
        # for action in proof:
        #     start_time = time.time()
        #     dd = dd.apply_action(action)
        #     inf_time = time.time() - start_time
        #     all_timestamp.append(inf_time)
        base_name = f"ordered_{idx}_" + os.path.basename(case_path)
        proof_actions_indices = list(range(len(proof)))
        context_actions_indices = get_idx_list(context, proof)
        assert set(context_actions_indices) <= set(proof_actions_indices)
        diff = set(proof_actions_indices) - set(context_actions_indices)
        diff_list = sorted(list(diff))
        dd_write = Diagram()
        # all_mapp = {}
        for action_idx in context_actions_indices + diff_list:
            action = proof[action_idx]
            # all_mapp.update(dd.order(action)[0])
            # order_action = dd.order_action(action, all_mapp, True)
            dd_write = dd_write.apply_action(action)
        with open("./raw_stats/" + base_name, "w", encoding="utf-8") as f:
            for action in context:
                f.write(action_to_string(action, action.to_names) + "\n")
            f.write("\n")
            for action in proof:
                f.write(action_to_string(action, action.to_names) + "\n")
                # f.write(str(inf_time))
                # f.write("\n")
            f.write("\n")
            f.write(str(order_fact))
            f.write("\n")
            f.write("\n")
            for action in dd_write.actions:
                f.write(action_to_string(action, action.to_names) + "\n")
            assert order_fact in dd_write.used_facts
            if dd_write.reward > 0:
                f.write("\n")
                for fact in dd_write.good_facts:
                    score, diff = dd_write.score_fact(fact)[:2]
                    f.write(f"{str(fact)} {score} {diff}\n")


def get_idx_list(subset_list, full_list):
    """Given subset_list is a subset of full_list, get subset_list's ids."""
    pt = 0
    idx_list = []
    for idx, item in enumerate(full_list):
        if item == subset_list[pt]:
            pt += 1
            idx_list.append(idx)
        if pt >= len(subset_list):
            break
    return idx_list
