r"""CPU multiprocessing priority random search based on action dependency."""
# pylint: disable=too-many-nested-blocks

import argparse
import bisect
import functools
import glob
import logging
import math
import multiprocessing
import os
import pickle
import random
import signal
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cmp_to_key, partial

import numpy as np

from tonggeometry.action import Action
from tonggeometry.constructor import (BaseAcuteTriangle, CircumscribedCircle,
                                      ConstructorIndex, InCenter,
                                      PerpendicularLine)
from tonggeometry.constructor.primitives import Line
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import Circle, get_fact_dep
from tonggeometry.util import BufferNode, bron_kerbosch, isclose


@dataclass
class TreePriorNode:
    """Tree prior's node"""

    def __init__(self):
        self.next = {}
        self.dist = Counter()


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


def preserve_traceback(func):
    """A decorator to preserve multiprocessing traceback."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            # Capture the full traceback and re-raise the exception with it
            exc_info = sys.exc_info()
            raise Exception("".join(  # pylint: disable=broad-exception-raised
                traceback.format_exception(*exc_info))) from exc

    return wrapper


def init_worker():
    """Worker initializer to ignore kill signal."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_sym_composite(actions, diagram, pairs):
    """Get symmetry composites from actions."""
    composite_actions = []
    waiting_dict = {key: {} for key in ConstructorIndex}
    for action in actions:
        action_mirror = diagram.order_action(action, pairs, False)
        non_mirror = action_mirror == action  # not involve mirror points
        if non_mirror:
            key = (action.gen_depth, action.constructor_name, True)
            composite_actions.append((key, action, None, non_mirror, None))
            continue
        on_axis = None
        if action_mirror.constructor_name in ["MidArc", "AnyArc"]:
            action_mirror.from_names = action_mirror.from_names[
                1] + action_mirror.from_names[0] + action_mirror.from_names[2]
        elif action_mirror.constructor_name == "IntersectLineCircleOff":
            on_axis = action_mirror.from_names[:2] == action.from_names[:2]
        elif action_mirror.constructor_name == "IntersectCircleCircle":
            on_axis = action_mirror.from_names[0::2] == action.from_names[0::2]
        action_mirror.convert(diagram)
        if action_mirror == action:  # self-mirror
            key = (action.gen_depth, action.constructor_name, True)
            composite_actions.append((key, action, None, non_mirror, on_axis))
        elif action.from_names in waiting_dict[action.constructor_name]:
            early_action = waiting_dict[action.constructor_name].pop(
                action.from_names)
            key = (early_action.gen_depth, early_action.constructor_name,
                   False)
            # GENERATED POINTS MUST BE ORDERED DURING SEARCH, NO rev_names
            # if early_action.constructor_name == "IntersectCircleCircle":
            #     action.rev_names = True
            composite_actions.append(
                (key, early_action, action, non_mirror, on_axis))
        else:
            waiting_dict[action_mirror.constructor_name][
                action_mirror.from_names] = action
    return composite_actions


def take_composite_action(diagram,
                          composite_action,
                          pairs,
                          problem_path,
                          cpu_id,
                          count,
                          check,
                          logger,
                          ck_func="ordered"):
    """The diagram takes the (self-)symmetric composite action."""

    def check_diagram_unordered(d, map_dict):
        """Check diagram function"""
        num = 0
        level_score = 0
        d_score = 0
        to_write_d = False
        for good_problem in d.good_problems:
            dd = Diagram()
            all_map = {}
            inv_map = {}
            level_score_tmp = 0
            d_score_tmp = 0
            for ctr, action_idx in enumerate(good_problem[0] +
                                             good_problem[1]):
                d_action = d.actions[action_idx]
                action = Action(d_action.constructor, d_action.from_names,
                                d.to_names_seq[action_idx])
                rep_mapping, inv_mapping = dd.order(action)
                all_map.update(rep_mapping)
                inv_map.update(inv_mapping)
                order_action = dd.order_action(action, all_map, True)
                if (action.constructor_name == "IntersectLineCircleOff" and
                    (d.point_dict[d_action.from_names[0]] -
                     d.point_dict[d_action.from_names[1]]
                     ).dot(dd.point_dict[order_action.from_names[0]] -
                           dd.point_dict[order_action.from_names[1]]) < 0):
                    order_action.from_names = order_action.from_names[
                        1] + order_action.from_names[
                            0] + order_action.from_names[2:]
                if (action.constructor_name == "IntersectCircleCircle"
                        and all_map[d_action.from_names[0]]
                        > all_map[d_action.from_names[2]]):
                    order_action.from_names = order_action.from_names[
                        2:] + order_action.from_names[:2]
                dd = dd.apply_action(order_action)
                if dd.terminal_flag:
                    logger.error("TERMINAL")
                    print("TERMINAL")
                    logger.error(str(d.actions))
                    print(d.actions)
                    logger.error(str(map_dict))
                    print(map_dict)
                    logger.error(str(dd.actions))
                    print(dd.actions)
                    logger.error(str(order_action))
                    print(order_action)
                    break
                p_score_string = [0] * 4
                if ctr >= len(good_problem[0]) and dd.reward > 0:
                    to_write_d = True
                    file_name = os.path.join(
                        problem_path, f"{cpu_id}-{count}-{d.depth}-{num}.txt")
                    with open(file_name, "w", encoding="utf-8") as file:
                        for i in range(len(good_problem[0])):
                            action = dd.actions[i]
                            file.write(f"{str(action)}\n")
                        file.write("\n")
                        for i in range(len(good_problem[0]), len(dd.actions)):
                            action = dd.actions[i]
                            file.write(f"{str(action)}\n")
                        file.write("\n")
                        for fact, fact_meta in zip(dd.good_facts,
                                                   dd.good_facts_meta):
                            context_max, aux_min = fact_meta
                            if context_max >= len(
                                    good_problem[0]) or aux_min < len(
                                        good_problem[0]):
                                continue
                            if fact.type == "eqline":
                                p_score_string[0] = 1
                            elif fact.type == "eqcircle":
                                p_score_string[1] = 1
                            elif fact.type == "simtri":
                                p_score_string[2] = 1
                            elif fact.type == "cong":
                                p_score_string[3] = 1
                            score, diff, fact_depth = dd.score_fact(fact)[:3]
                            inv_fact = dd.order_fact(
                                dd.order_fact(fact, inv_map), map_dict)
                            same = 0
                            if any(p not in all_map
                                   for p in get_fact_dep(inv_fact)):
                                sym = 0
                            else:
                                inv_fact = dd.order_fact(inv_fact, all_map)
                                if inv_fact == fact:
                                    sym = 1
                                    same = 1
                                elif inv_fact in dd.used_facts:
                                    sym = 1
                                else:
                                    sym = 0
                            file.write(
                                f"{str(fact)} {score} {diff} {fact_depth} " +
                                f"{context_max} {aux_min} {sym} {same}\n")
                    num += 1
                p_score = int("".join(map(str, p_score_string)), 2)
                level_score_tmp |= p_score
                d_score_tmp += p_score
            level_score = max(level_score, level_score_tmp)
            d_score = max(d_score, d_score_tmp)
        logger.debug("%i %i", level_score, d_score)
        if to_write_d:
            file_name = os.path.join(problem_path,
                                     f"{cpu_id}-{count}-{d.depth}.conf")
            with open(file_name, "w", encoding="utf-8") as file:
                file.write("\n".join(str(d.actions)[1:-1].split(", ")))
        return d_score

    def check_diagram_ordered(d, map_dict):
        """Check diagram function"""
        num = 0
        level_score = 0
        d_score = 0
        to_write_d = False
        # maximum clique algorithm to maximally reuse computation and filter gp
        # complexity 3^(n/3)
        all_good_problems = d.good_problems.keys
        graph = {i: set() for i in range(len(all_good_problems))}
        for idx, good_problem in enumerate(all_good_problems):
            g_idx_0, g_idx_1 = good_problem
            sg_idx_0, sg_idx_1 = set(g_idx_0), set(g_idx_1)
            for idxx, good_problemm in enumerate(all_good_problems[idx + 1:]):
                idxx += idx + 1
                g_idxx_0, g_idxx_1 = good_problemm
                # no intersections between aux and context
                if len(sg_idx_0.intersection(g_idxx_1)) == 0 and len(
                        sg_idx_1.intersection(g_idxx_0)) == 0:
                    graph[idx].add(idxx)
                    graph[idxx].add(idx)
        all_cliques = []
        used = set()
        for i in range(len(all_good_problems)):
            if i in used:
                continue
            i_all_cliques = bron_kerbosch({i}, set(graph[i]), set(), graph)
            max_clique = ()
            for i_clique in i_all_cliques:
                if len(i_clique) > len(max_clique):
                    max_clique = i_clique
            all_cliques.append(max_clique)
            used.update(max_clique)
            for j in max_clique:
                for k in graph[j]:
                    graph[k].remove(j)
                graph.pop(j)
        filtered_good_problems = []
        for clique in all_cliques:
            context = set()
            for good_problem_idx in clique:
                good_problem = all_good_problems[good_problem_idx]
                context.update(good_problem[0])
            action_indices = sorted(list(context))
            dd = Diagram()
            for action_idx in action_indices:
                d_action = d.actions[action_idx]
                dd_action = Action(d_action.constructor, d_action.from_names,
                                   d.to_names_seq[action_idx])
                dd = dd.apply_action(dd_action)
                if dd.terminal_flag:
                    logger.error("FILTER ERROR")
                    print("FILTER ERROR")
                    logger.error(d.actions)
                    print(d.actions)
                    logger.error(action_indices)
                    print(action_indices)
                    break
            for good_problem_idx in clique:
                good_problem = all_good_problems[good_problem_idx]
                node = d.good_problems.retrieve(good_problem)
                for fact in node.seps[len(good_problem[0])]:
                    if fact not in dd.used_facts:
                        filtered_good_problems.append(good_problem)
                        break
        for good_problem in filtered_good_problems:  # pylint: disable=too-many-nested-blocks
            dd = Diagram()
            all_map = {}
            inv_map = {}
            level_score_tmp = 0
            d_score_tmp = 0
            exec_row_to_action = {}
            aux_row_to_action = {}
            exec_matrix = np.zeros(
                (len(good_problem[0]) - 1, len(dd.all_names) + 1), dtype=bool)
            aux_matrix = np.zeros(
                (len(good_problem[1]), len(dd.all_names) + 1), dtype=bool)
            # build auxiliary data structures for faster action selection
            for part_idx, good_problem_part in enumerate(good_problem):
                if part_idx == 0:
                    row_to_action = exec_row_to_action
                    matrix = exec_matrix
                    good_problem_part = good_problem_part[1:]
                else:
                    row_to_action = aux_row_to_action
                    matrix = aux_matrix
                for row, action_idx in enumerate(good_problem_part):
                    d_action = d.actions[action_idx]
                    d_action_with_to = Action(d_action.constructor,
                                              d_action.from_names,
                                              d.to_names_seq[action_idx])
                    row_to_action[row] = d_action_with_to
                    for char in d_action_with_to.from_names:
                        if char >= "a":
                            col = ord(char) - ord("a") + 26
                        else:
                            col = ord(char) - ord("A")
                        matrix[row][col] = True
            # base action never changes in both d and dd
            d_action = d.actions[0]
            base_action = Action(d_action.constructor, d_action.from_names,
                                 d.to_names_seq[0])
            # actions in buffer should have sorted actions with mapped from_name
            exec_buffer = [(base_action, False)]
            aux_buffer = []
            exec_circle_actions = defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: [])))
            aux_circle_actions = defaultdict(
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
                if dd.terminal_flag:
                    logger.error("TERMINAL")
                    print("TERMINAL")
                    logger.error(str(d.actions))
                    print(d.actions)
                    logger.error(str(map_dict))
                    print(map_dict)
                    logger.error(str(dd.actions))
                    print(dd.actions)
                    logger.error(str(exec_action))
                    print(exec_action)
                    break
                for new_to_name in new_to_names:
                    points_to_new_points[new_to_name] = (new_to_names,
                                                         dd.depth)
                p_score_string = [0] * 4
                if len(dd.actions) > len(good_problem[0]) and dd.reward > 0:
                    to_write_dd = False
                    to_write = ""
                    for i in range(len(good_problem[0])):
                        to_write += f"{str(dd.actions[i])}\n"
                    to_write += "\n"
                    for i in range(len(good_problem[0]), len(dd.actions)):
                        to_write += f"{str(dd.actions[i])}\n"
                    to_write += "\n"
                    for fact, fact_meta in zip(dd.good_facts,
                                               dd.good_facts_meta):
                        context_max, aux_min = fact_meta
                        if context_max != len(
                                good_problem[0]) - 1 or aux_min < len(
                                    good_problem[0]):
                            continue
                        to_write_dd = True
                        to_write_d = True
                        if fact.type == "eqline":
                            p_score_string[0] = 1
                        elif fact.type == "eqcircle":
                            p_score_string[1] = 1
                        elif fact.type == "simtri":
                            p_score_string[2] = 1
                        elif fact.type == "cong":
                            p_score_string[3] = 1
                        score, diff, fact_depth = dd.score_fact(fact)[:3]
                        inv_fact = dd.order_fact(dd.order_fact(fact, inv_map),
                                                 map_dict)
                        same = 0
                        if any(p not in all_map
                               for p in get_fact_dep(inv_fact)):
                            sym = 0
                        else:
                            inv_fact = dd.order_fact(inv_fact, all_map)
                            if inv_fact == fact:
                                sym = 1
                                same = 1
                            elif inv_fact in dd.used_facts:
                                sym = 1
                            else:
                                sym = 0
                        to_write += (
                            f"{str(fact)} {score} {diff} {fact_depth} " +
                            f"{context_max} {aux_min} {sym} {same}\n")
                    if to_write_dd:
                        file_name = os.path.join(
                            problem_path,
                            f"{cpu_id}-{count}-{d.depth}-{num}.txt")
                        with open(file_name, "w", encoding="utf-8") as f:
                            f.write(to_write)
                        num += 1
                p_score = int("".join(map(str, p_score_string)), 2)
                level_score_tmp |= p_score
                d_score_tmp += p_score

                if len(dd.actions) == len(good_problem[0]):
                    exec_buffer = aux_buffer

                if exec_action.constructor_name != "CenterCircle":
                    cols = []
                    for new_to_name in new_to_names:
                        to_name = inv_map[new_to_name]
                        if to_name >= "a":
                            col = ord(to_name) - ord("a") + 26
                        else:
                            col = ord(to_name) - ord("A")
                        cols.append(col)
                    if len(dd.actions) < len(good_problem[0]):
                        exec_buffer.extend(
                            circle_sort(dd, None, exec_circle_actions, all_map,
                                        d, points_to_new_points))
                        exec_matrix[:, cols] = False
                        exec_actions_rows = np.where(
                            np.count_nonzero(exec_matrix, axis=-1) == 0)[0]
                        exec_matrix[exec_actions_rows, -1] = True
                        exec_actions = [
                            exec_row_to_action[row_idx]
                            for row_idx in exec_actions_rows
                        ]
                        exec_buffer.extend(
                            point_sort(dd, exec_actions, exec_circle_actions,
                                       all_map, d))
                    aux_buffer.extend(
                        circle_sort(dd, None, aux_circle_actions, all_map, d,
                                    points_to_new_points))
                    aux_matrix[:, cols] = False
                    aux_actions_rows = np.where(
                        np.count_nonzero(aux_matrix, axis=-1) == 0)[0]
                    aux_matrix[aux_actions_rows, -1] = True
                    aux_actions = [
                        aux_row_to_action[row_idx]
                        for row_idx in aux_actions_rows
                    ]
                    aux_buffer.extend(
                        point_sort(dd, aux_actions, aux_circle_actions,
                                   all_map, d))
                else:
                    if len(dd.actions) < len(good_problem[0]):
                        exec_buffer.extend(
                            circle_sort(dd, exec_action.from_names,
                                        exec_circle_actions, all_map, d))
                    aux_buffer.extend(
                        circle_sort(dd, exec_action.from_names,
                                    aux_circle_actions, all_map, d))
            if not dd.terminal_flag and len(
                    dd.actions) != len(good_problem[0]) + len(good_problem[1]):
                logger.error("SORT ERROR")
                print("SORT ERROR")
                logger.error(str(d.actions))
                print(d.actions)
                logger.error(str(good_problem))
                print(good_problem)

            level_score = max(level_score, level_score_tmp)
            d_score = max(d_score, d_score_tmp)
        logger.debug("%i %i", level_score, d_score)
        if to_write_d:
            file_name = os.path.join(problem_path,
                                     f"{cpu_id}-{count}-{d.depth}.conf")
            with open(file_name, "w", encoding="utf-8") as file:
                file.write("\n".join(str(d.actions)[1:-1].split(", ")))
        return d_score

    # def check_diagram_wild(d, map_dict):
    #     """Check diagram function"""
    #     d_score = 0
    #     level_score = 0
    #     if len(d.good_problems) == 0:
    #         return d_score
    #     dd = Diagram(mode=0)
    #     all_map = {}
    #     inv_map = {}
    #     exec_row_to_action = {}
    #     exec_matrix = np.zeros((len(d.actions) - 1, len(dd.all_names) + 1),
    #                            dtype=bool)
    #     for idx, d_action in enumerate(d.actions[1:]):
    #         d_action_with_to = Action(d_action.constructor,
    #                                   d_action.from_names,
    #                                   d.to_names_seq[idx + 1])
    #         exec_row_to_action[idx] = d_action_with_to
    #         row = idx
    #         for char in d_action_with_to.from_names:
    #             if char >= "a":
    #                 col = ord(char) - ord("a") + 26
    #             else:
    #                 col = ord(char) - ord("A")
    #             exec_matrix[row][col] = True
    #     # base action never changes in both d and dd
    #     d_action = d.actions[0]
    #     base_action = Action(d_action.constructor, d_action.from_names,
    #                          d.to_names_seq[0])
    #     # actions in buffer should have sorted actions with mapped from_name
    #     exec_buffer = [(base_action, False)]
    #     exec_circle_actions = defaultdict(
    #         lambda: defaultdict(lambda: defaultdict(lambda: [])))
    #     points_to_new_points = {}
    #     while exec_buffer:
    #         exec_action, inv_to = exec_buffer.pop(0)
    #         rep_mapping, inv_mapping = dd.order(exec_action)
    #         all_map.update(rep_mapping)
    #         inv_map.update(inv_mapping)
    #         old_to_names = exec_action.to_names
    #         new_to_names = "".join(all_map[char] for char in old_to_names)
    #         exec_action.to_names = new_to_names
    #         if inv_to:
    #             all_map[old_to_names[0]] = new_to_names[1]
    #             all_map[old_to_names[1]] = new_to_names[0]
    #             inv_map[new_to_names[0]] = old_to_names[1]
    #             inv_map[new_to_names[1]] = old_to_names[0]
    #         dd = dd.apply_action(exec_action)
    #         if dd.terminal_flag:
    #             logger.error("ERROR DURING REORDER")
    #             print("ERROR DURING REORDER")
    #             logger.error(str(d.actions))
    #             print(d.actions)
    #             logger.error(str(dd.actions))
    #             print(dd.actions)
    #             logger.error(str(exec_action))
    #             print(exec_action)
    #             break
    #         for new_to_name in new_to_names:
    #             points_to_new_points[new_to_name] = (new_to_names, dd.depth)
    #         if exec_action.constructor_name != "CenterCircle":
    #             cols = []
    #             for new_to_name in new_to_names:
    #                 to_name = inv_map[new_to_name]
    #                 if to_name >= "a":
    #                     col = ord(to_name) - ord("a") + 26
    #                 else:
    #                     col = ord(to_name) - ord("A")
    #                 cols.append(col)
    #             exec_buffer.extend(
    #                 circle_sort(dd, None, exec_circle_actions, all_map, d,
    #                             points_to_new_points))
    #             exec_matrix[:, cols] = False
    #             exec_actions_rows = np.where(
    #                 np.count_nonzero(exec_matrix, axis=-1) == 0)[0]
    #             exec_matrix[exec_actions_rows, -1] = True
    #             exec_actions = [
    #                 exec_row_to_action[row_idx]
    #                 for row_idx in exec_actions_rows
    #             ]
    #             exec_buffer.extend(
    #                 point_sort(dd, exec_actions, exec_circle_actions, all_map,
    #                            d))
    #         else:
    #             exec_buffer.extend(
    #                 circle_sort(dd, exec_action.from_names,
    #                             exec_circle_actions, all_map, d))
    #     if dd.terminal_flag:
    #         return d_score
    #     if not dd.terminal_flag and len(dd.actions) != len(d.actions):
    #         logger.error("SORT ERROR")
    #         print("SORT ERROR")
    #         logger.error(str(d.actions))
    #         print(d.actions)
    #         return d_score
    #     num = 0
    #     dd_map_dict = {
    #         all_map[key]: all_map[val]
    #         for key, val in map_dict.items()
    #     }
    #     for good_problem in dd.good_problems:
    #         all_facts = dd.good_problems.retrieve(good_problem).seps[len(
    #             good_problem[0])]
    #         good = False
    #         for fact in all_facts:
    #             if fact.type not in ["eqratio", "eqangle", "midp", "contri"]:
    #                 good = True
    #                 break
    #         if not good:
    #             continue
    #         dd_write = Diagram()
    #         all_map_write = {}
    #         inv_map_write = {}
    #         for action_idx in good_problem[0] + good_problem[1]:
    #             dd_action = dd.actions[action_idx]
    #             action = Action(dd_action.constructor, dd_action.from_names,
    #                             dd.to_names_seq[action_idx])
    #             old_to_names = action.to_names
    #             rep_mapping, inv_mapping = dd_write.order(action)
    #             all_map_write.update(rep_mapping)
    #             inv_map_write.update(inv_mapping)
    #             order_action = dd_write.order_action(action, all_map_write,
    #                                                  False)
    #             new_to_names = order_action.to_names
    #             if (action.constructor_name == "IntersectLineCircleOff" and
    # order_action.from_names[
    #                     0] > order_action.from_names[1]):
    #                 order_action.from_names = order_action.from_names[
    #                     1] + order_action.from_names[
    #                         0] + order_action.from_names[2:]
    #                 all_map_write[old_to_names[0]] = new_to_names[1]
    #                 all_map_write[old_to_names[1]] = new_to_names[0]
    #                 inv_map_write[new_to_names[0]] = old_to_names[1]
    #                 inv_map_write[new_to_names[1]] = old_to_names[0]
    #             if action.constructor_name == "IntersectCircleCircle" and order_action.from_names[
    #                     0] > order_action.from_names[2]:
    #                 order_action.from_names = order_action.from_names[
    #                     2:] + order_action.from_names[:2]
    #                 all_map_write[old_to_names[0]] = new_to_names[1]
    #                 all_map_write[old_to_names[1]] = new_to_names[0]
    #                 inv_map_write[new_to_names[0]] = old_to_names[1]
    #                 inv_map_write[new_to_names[1]] = old_to_names[0]
    #             order_action.s_check = True
    #             dd_write = dd_write.apply_action(order_action, draw_only=True)
    #             if dd_write.terminal_flag:
    #                 logger.error("DRAW ERROR")
    #                 print("DRAW ERROR")
    #                 logger.error(str(d.actions))
    #                 print(d.actions)
    #                 logger.error(str(dd.actions))
    #                 print(dd.actions)
    #                 logger.error(str(dd_write.actions))
    #                 print(dd_write.actions)
    #                 logger.error(str(order_action))
    #                 print(order_action)
    #                 break
    #         if dd_write.terminal_flag:
    #             continue
    #         file_name = os.path.join(problem_path,
    #                                  f"{cpu_id}-{count}-{d.depth}-{num}.txt")
    #         with open(file_name, "w", encoding="utf-8") as file:
    #             for i in range(len(good_problem[0])):
    #                 action = dd_write.actions[i]
    #                 file.write(f"{str(action)}\n")
    #             file.write("\n")
    #             for i in range(len(good_problem[0]), len(dd_write.actions)):
    #                 action = dd_write.actions[i]
    #                 file.write(f"{str(action)}\n")
    #             file.write("\n")
    #             for fact in all_facts:
    #                 dd.trace_fact(fact)
    #                 score, diff, fact_depth = dd.score_fact(fact)[:3]
    #                 ordered_fact = dd_write.order_fact(fact, all_map_write)
    #                 context_max = len(good_problem[0]) - 1
    #                 aux_min = len(good_problem[0])
    #                 inv_fact = dd.order_fact(fact, dd_map_dict)
    #                 same = 0
    #                 if any(p not in all_map_write
    #                        for p in get_fact_dep(inv_fact)):
    #                     sym = 0
    #                 else:
    #                     if inv_fact == fact:
    #                         sym = 1
    #                         same = 1
    #                     elif inv_fact in dd.used_facts:
    #                         sym = 1
    #                     else:
    #                         sym = 0
    #                 file.write(
    #                     f"{str(ordered_fact)} {score} {diff} {fact_depth} " +
    #                     f"{context_max} {aux_min} {sym} {same}\n")
    #         num += 1
    #     for good_problem in dd.good_problems:
    #         level_score_tmp = 0
    #         d_score_tmp = 0
    #         node = dd.good_problems.retrieve(good_problem)
    #         for sep_facts in node.seps.values():
    #             p_score_string = [0] * 4
    #             for sep_fact in sep_facts:
    #                 if sep_fact.type == "eqline":
    #                     p_score_string[0] = 1
    #                 elif sep_fact.type == "eqcircle":
    #                     p_score_string[1] = 1
    #                 elif sep_fact.type == "simtri":
    #                     p_score_string[2] = 1
    #                 elif sep_fact.type == "cong":
    #                     p_score_string[3] = 1
    #             p_score = int("".join(map(str, p_score_string)), 2)
    #             level_score_tmp |= p_score
    #             d_score_tmp += p_score
    #         level_score = max(level_score, level_score_tmp)
    #         d_score = max(d_score, d_score_tmp)
    #     logger.debug("%i %i", level_score, d_score)
    #     if len(dd.good_problems) > 0:
    #         file_name = os.path.join(problem_path,
    #                                  f"{cpu_id}-{count}-{d.depth}.conf")
    #         with open(file_name, "w", encoding="utf-8") as file:
    #             file.write("\n".join(str(d.actions)[1:-1].split(", ")))
    #     return d_score

    if ck_func == "unordered":
        check_diagram = check_diagram_unordered
    else:
        check_diagram = check_diagram_ordered

    additional_actions = []
    additional_pairs = {}
    _, first_action, second_action, non_mirror, on_axis = composite_action
    new_pairs = dict(pairs)
    d_score = 0
    for ith, action in enumerate([first_action, second_action]):
        if action is None:
            break
        try:
            old_diagram = diagram
            diagram = diagram.apply_action(action)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("EXCEPTION")
            print("EXCEPTION")
            logger.error(str(e))
            print(e)
            logger.error(str(diagram.actions))
            print(diagram.actions)
            logger.error(str(action))
            print(action)
            break
        if ith == 0:
            if diagram.terminal_flag:
                return (diagram, -1, {}, [])
            additional_actions += diagram.new_valid_actions()
            # self symmetry, special case for two points
            self_sym = second_action is None and not non_mirror
            to_reverse = (
                first_action.constructor_name == "IntersectLineCircleOff"
                and not on_axis) or (first_action.constructor_name
                                     == "IntersectCircleCircle" and on_axis)
            if self_sym and to_reverse:
                for p, pp in zip(diagram.to_names_seq[-1],
                                 diagram.to_names_seq[-1][::-1]):
                    additional_pairs[p] = pp
            else:
                for p in diagram.to_names_seq[-1]:
                    additional_pairs[p] = p
        else:
            # duplicates
            if (diagram.terminal_flag and not diagram.OOB_terminal
                    and not diagram.inf_terminal and diagram.full_dup):
                logger.warning("%s, %s",
                               str(diagram.actions)[1:-1], str(second_action))
                diagram = old_diagram
                # fall back
                if (action.constructor_name == "IntersectLineCircleOff"
                        and not on_axis) or (action.constructor_name
                                             == "IntersectCircleCircle"
                                             and on_axis):
                    for p, pp in zip(diagram.to_names_seq[-1],
                                     diagram.to_names_seq[-1][::-1]):
                        new_pairs[p] = pp
                    if check:
                        d_score = check_diagram(diagram, new_pairs)
                return (diagram, d_score, new_pairs, additional_actions)
            if diagram.terminal_flag:
                return (diagram, -1, {}, [])
            additional_actions += diagram.new_valid_actions()
            # IntersectLineCircleOff: off_axis and direction change
            # IntersectCircleCircle: on_axis or no swap (default)
            if (action.constructor_name == "IntersectLineCircleOff"  # pylint: disable=too-many-boolean-expressions
                    and not on_axis and
                (diagram.point_dict[pairs[first_action.from_names[0]]] -
                 diagram.point_dict[pairs[first_action.from_names[1]]]
                 ).dot(diagram.point_dict[action.from_names[0]] -
                       diagram.point_dict[action.from_names[1]])
                    < 0) or (action.constructor_name == "IntersectCircleCircle"
                             and (on_axis or pairs[first_action.from_names[0]]
                                  < pairs[first_action.from_names[2]])):
                for p, pp in zip(diagram.to_names_seq[-2],
                                 diagram.to_names_seq[-1][::-1]):
                    additional_pairs[p] = pp
                    additional_pairs[pp] = p
            else:
                for p, pp in zip(diagram.to_names_seq[-2],
                                 diagram.to_names_seq[-1]):
                    additional_pairs[p] = pp
                    additional_pairs[pp] = p
        new_pairs.update(additional_pairs)
        if check:
            d_score = check_diagram(diagram, new_pairs)
        if diagram.depth == diagram.max_depth:
            break

    return (diagram, d_score, new_pairs, additional_actions)


def prior_sample(composite_actions, prior):
    """Return an action index where actions types follow the prior dist."""
    good = False
    counter = Counter()
    for composite_action in composite_actions:
        key = composite_action[0]
        if key in prior:
            good = True
        counter[key] += 1
    if not good:
        sample_idx = random.randrange(len(composite_actions))
    else:
        weights = []
        length = len(composite_actions)
        for composite_action in composite_actions:
            key = composite_action[0]
            weights.append(prior[key] * length / counter[key])
        sample_idx = random.choices(list(range(len(composite_actions))),
                                    weights=weights,
                                    k=1)[0]
    return sample_idx


@preserve_traceback
def worker(input):
    """MCTS worker."""
    (node_pairs, all_actions, seed, loop, buffer_size, per_sample_loop,
     problem_path, log_path, cpu_id, prior_root, ck_func, tree_prior) = input
    logging.basicConfig(filename=os.path.join(log_path, f"{cpu_id}.log"),
                        filemode="a",
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S %p',
                        encoding="utf-8",
                        level=logging.DEBUG)
    logger = logging.getLogger(f"{cpu_id}")
    logger.debug("Process %i (pid %i) starts from parent (pid %i)", cpu_id,
                 os.getpid(), os.getppid())
    dd = Diagram().apply_actions([Action(BaseAcuteTriangle, "")])
    p_C = dd.point_dict["C"]
    p_M = (dd.point_dict["A"] + dd.point_dict["B"]) / 2
    l_G = Line(p_M, p_C)
    dd = Diagram().apply_actions(
        [Action(BaseAcuteTriangle, ""),
         Action(InCenter, "ABC")])
    p_I = dd.point_dict["D"]
    l_I = Line(p_I, p_C)
    dd = Diagram().apply_actions(
        [Action(BaseAcuteTriangle, ""),
         Action(CircumscribedCircle, "ABC")])
    p_O = dd.point_dict["D"]
    l_O = Line(p_O, p_C)
    dd = Diagram().apply_actions(
        [Action(BaseAcuteTriangle, ""),
         Action(PerpendicularLine, "ACB")])
    p_H = dd.point_dict["D"]
    l_H = Line(p_H, p_C)
    if os.path.exists(os.path.join(log_path, f"{cpu_id}_state.pkl")):
        logger.debug("Picked up from where it ended")
        with open(os.path.join(log_path, f"{cpu_id}_state.pkl"), "rb") as f:
            state = pickle.load(f)
        count = state["count"]
        buffer = state["buffer"]
        buffer_node = state["buffer_node"]
        random_state = state["random_state"]
        random.setstate(random_state)
        start_path = state["start_path"]
        if len(start_path) == 0:
            sample = node_pairs
            node_all_actions = all_actions
            prior_head = prior_root
            seen_start = {"A", "B", "C"}
            all_ks_start = [0]
            k_counter_start = {0: 2}
            all_cs_start = [p_M]
            c_counter_start = {0: 2}
            num_G_start = 1
            num_I_start = 1
            num_O_start = 1
            num_H_start = 1
            num_V_start = 2
        else:
            d, pairs = node_pairs
            node_all_actions = [] + all_actions
            prior_head = prior_root
            seen_start = {"A", "B", "C"}
            all_ks_start = [0]
            k_counter_start = {0: 2}
            all_cs_start = [p_M]
            c_counter_start = {0: 2}
            num_G_start = 1
            num_I_start = 1
            num_O_start = 1
            num_H_start = 1
            num_V_start = 2
            for action_idx in start_path:
                action = node_all_actions[action_idx]
                if tree_prior:
                    c_key = action[0]
                    prior_head = prior_head.next[c_key]
                d_old_len = len(d.point_dict)
                d, _, pairs, d_new_actions = take_composite_action(
                    d, action, pairs, problem_path, cpu_id, count, False,
                    logger)
                d_new_len = len(d.point_dict)
                for idx in range(d_old_len, d_new_len):
                    p = d.alphabet[idx]
                    if p in seen_start:
                        continue
                    p_sym = pairs[p]
                    if p == p_sym:
                        pp = d.point_dict[p]
                        if pp in l_G:
                            num_G_start += 1
                        if pp in l_I:
                            num_I_start += 1
                        if pp in l_O:
                            num_O_start += 1
                        if pp in l_H:
                            num_H_start += 1
                        if isclose(pp.x, 0, 1e-4):
                            num_V_start += 1
                        seen_start.add(p)
                        continue
                    pp = d.point_dict[p]
                    pp_sym = d.point_dict[p_sym]
                    if isclose(pp.y, pp_sym.y, 1e-4):
                        k = 0
                    elif isclose(pp.x, pp_sym.x, 1e-4):
                        k = float("inf")
                    else:
                        k = (pp.y - pp_sym.y) / (pp.x - pp_sym.x)
                    c = (pp + pp_sym) / 2
                    match = False
                    for idx, kk in enumerate(all_ks_start):
                        if (k == float("inf")
                                and kk == float("inf")) or isclose(
                                    k, kk, 1e-4):
                            k_counter_start[idx] += 2
                            match = True
                            break
                    if not match:
                        all_ks_start.append(k)
                        k_counter_start[len(all_ks_start) - 1] = 2
                    match = False
                    for idx, cc in enumerate(all_cs_start):
                        if c == cc:
                            c_counter_start[idx] += 2
                            match = True
                            break
                    if not match:
                        all_cs_start.append(c)
                        c_counter_start[len(all_cs_start) - 1] = 2
                    if c in l_G:
                        num_G_start += 2
                    if c in l_I:
                        num_I_start += 2
                    if c in l_O:
                        num_O_start += 2
                    if c in l_H:
                        num_H_start += 2
                    if isclose(c.x, 0, 1e-4) and isclose(pp.y, pp_sym.y, 1e-4):
                        num_V_start += 2
                    seen_start.add(p)
                    seen_start.add(p_sym)
                node_all_actions = node_all_actions[action_idx +
                                                    1:] + get_sym_composite(
                                                        d_new_actions, d,
                                                        pairs)
            sample = (d, pairs)
    else:
        logger.debug("Starting new")
        random.seed(seed)
        buffer = []
        buffer_node = BufferNode()
        count = 0
        sample = node_pairs
        node_all_actions = all_actions
        start_path = ()
        prior_head = prior_root
        seen_start = {"A", "B", "C"}
        all_ks_start = [0]
        k_counter_start = {0: 2}
        all_cs_start = [p_M]
        c_counter_start = {0: 2}
        num_G_start = 1
        num_I_start = 1
        num_O_start = 1
        num_H_start = 1
        num_V_start = 2
    while count != loop:
        logger.debug("%i th loop", count)
        d, pairs = sample
        logger.debug(str(d.actions)[1:-1])
        d_all_actions = [] + node_all_actions
        path = start_path
        prior = prior_head
        seen = set(seen_start)
        all_ks = all_ks_start[:]
        k_counter = dict(k_counter_start)
        all_cs = all_cs_start[:]
        c_counter = dict(c_counter_start)
        num_G = num_G_start
        num_I = num_I_start
        num_O = num_O_start
        num_H = num_H_start
        num_V = num_V_start
        while True:
            if not tree_prior:
                prior_dist = prior[d.depth]
            else:
                prior_dist = prior.dist
            action_idx = prior_sample(d_all_actions, prior_dist)
            action = d_all_actions[action_idx]
            c_key = action[0]
            if c_key not in prior_dist:
                break
            logger.debug("%s, %s", str(action[1]), str(action[2]))
            try:
                d_old_len = len(d.point_dict)
                (d, d_score, pairs,
                 additional_actions) = take_composite_action(
                     d, action, pairs, problem_path, cpu_id, count, True,
                     logger, ck_func)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error("EXCEPTION")
                print("EXCEPTION")
                logger.error(str(e))
                print(e)
                logger.error(str(d.actions))
                print(d.actions)
                logger.error(str(action))
                print(action)
                logger.error(pairs)
                print(pairs)
                break
            if d.is_terminal:
                break
            d_new_len = len(d.point_dict)
            for idx in range(d_old_len, d_new_len):
                p = d.alphabet[idx]
                if p in seen:
                    continue
                p_sym = pairs[p]
                if p == p_sym:
                    pp = d.point_dict[p]
                    if pp in l_G:
                        num_G += 1
                    if pp in l_I:
                        num_I += 1
                    if pp in l_O:
                        num_O += 1
                    if pp in l_H:
                        num_H += 1
                    if isclose(pp.x, 0, 1e-4):
                        num_V += 1
                    seen.add(p)
                    continue
                pp = d.point_dict[p]
                pp_sym = d.point_dict[p_sym]
                if isclose(pp.y, pp_sym.y, 1e-4):
                    k = 0
                elif isclose(pp.x, pp_sym.x, 1e-4):
                    k = float("inf")
                else:
                    k = (pp.y - pp_sym.y) / (pp.x - pp_sym.x)
                c = (pp + pp_sym) / 2
                match = False
                for idx, kk in enumerate(all_ks):
                    if (k == float("inf") and kk == float("inf")) or isclose(
                            k, kk, 1e-4):
                        k_counter[idx] += 2
                        match = True
                        break
                if not match:
                    all_ks.append(k)
                    k_counter[len(all_ks) - 1] = 2
                match = False
                for idx, cc in enumerate(all_cs):
                    if c == cc:
                        c_counter[idx] += 2
                        match = True
                        break
                if not match:
                    all_cs.append(c)
                    c_counter[len(all_cs) - 1] = 2
                if c in l_G:
                    num_G += 2
                if c in l_I:
                    num_I += 2
                if c in l_O:
                    num_O += 2
                if c in l_H:
                    num_H += 2
                if isclose(c.x, 0, 1e-4) and isclose(pp.y, pp_sym.y, 1e-4):
                    num_V += 2
                seen.add(p)
                seen.add(p_sym)
            if d.depth >= 17 and (max(k_counter.values()) / d_new_len >= 0.45
                                  or max(c_counter.values()) / d_new_len >= 0.6
                                  or any(e / d_new_len >= 0.6
                                         for e in (num_G, num_I, num_O, num_H,
                                                   num_V))):
                break
            if tree_prior:
                prior = prior.next[c_key]
            if (tree_prior and len(prior.dist)
                    == 0) or (not tree_prior and len(prior[d.depth]) == 0):
                break
            d_all_actions = d_all_actions[action_idx + 1:] + get_sym_composite(
                additional_actions, d, pairs)
            path += (action_idx, )
            if d_score == 0 or d.depth > 17 or (len(start_path) + 1) / (
                    len(path) + 1) > 0.8 or buffer_node.iou(path) > 0.8:
                continue
            if len(buffer) < buffer_size or d_score > buffer[0][-1]:
                val = (path, d_score)
                idx = bisect.bisect_right(buffer, val[-1], key=lambda x: x[-1])
                if idx == 0 or buffer[idx - 1][0] != path:
                    buffer.insert(idx, val)
                    buffer_node.insert(path)
            if len(buffer) > buffer_size:
                top_path = buffer[0][0]
                buffer = buffer[1:]
                buffer_node.delete(top_path)
        file_name = os.path.join(problem_path, f"{cpu_id}-{count}.path")
        with open(file_name, "w", encoding="utf-8") as file:
            file.write("\n".join(str(d.actions)[1:-1].split(", ")))
        if len(buffer) > 0:
            avg_score = sum(elem[-1] for elem in buffer) / len(buffer)
            logger.debug("%.2f average score for %d buffer nodes", avg_score,
                         len(buffer))
        count += 1
        if count % per_sample_loop == 0:
            if len(buffer) > 0 and random.uniform(0, 1) >= 0.5:
                logger.debug("exploit")
                weights = [math.log(elem[-1] + 1, 2) for elem in buffer]
                sample_idx = random.choices(list(range(len(buffer))),
                                            weights=weights,
                                            k=1)[0]
                start_path, d_score = buffer.pop(sample_idx)
                buffer_node.delete(start_path)
                d, pairs = node_pairs
                node_all_actions = [] + all_actions
                prior_head = prior_root
                seen_start = {"A", "B", "C"}
                all_ks_start = [0]
                k_counter_start = {0: 2}
                all_cs_start = [p_M]
                c_counter_start = {0: 2}
                num_G_start = 1
                num_I_start = 1
                num_O_start = 1
                num_H_start = 1
                num_V_start = 2
                for action_idx in start_path:
                    action = node_all_actions[action_idx]
                    if tree_prior:
                        c_key = action[0]
                        prior_head = prior_head.next[c_key]
                    d_old_len = len(d.point_dict)
                    d, _, pairs, d_new_actions = take_composite_action(
                        d, action, pairs, problem_path, cpu_id, count, False,
                        logger)
                    node_all_actions = node_all_actions[
                        action_idx + 1:] + get_sym_composite(
                            d_new_actions, d, pairs)
                    d_new_len = len(d.point_dict)
                    for idx in range(d_old_len, d_new_len):
                        p = d.alphabet[idx]
                        if p in seen_start:
                            continue
                        p_sym = pairs[p]
                        if p == p_sym:
                            pp = d.point_dict[p]
                            if pp in l_G:
                                num_G_start += 1
                            if pp in l_I:
                                num_I_start += 1
                            if pp in l_O:
                                num_O_start += 1
                            if pp in l_H:
                                num_H_start += 1
                            if isclose(pp.x, 0, 1e-4):
                                num_V_start += 1
                            seen_start.add(p)
                            continue
                        pp = d.point_dict[p]
                        pp_sym = d.point_dict[p_sym]
                        if isclose(pp.y, pp_sym.y, 1e-4):
                            k = 0
                        elif isclose(pp.x, pp_sym.x, 1e-4):
                            k = float("inf")
                        else:
                            k = (pp.y - pp_sym.y) / (pp.x - pp_sym.x)
                        c = (pp + pp_sym) / 2
                        match = False
                        for idx, kk in enumerate(all_ks_start):
                            if (k == float("inf")
                                    and kk == float("inf")) or isclose(
                                        k, kk, 1e-4):
                                k_counter_start[idx] += 2
                                match = True
                                break
                        if not match:
                            all_ks_start.append(k)
                            k_counter_start[len(all_ks_start) - 1] = 2
                        match = False
                        for idx, cc in enumerate(all_cs_start):
                            if c == cc:
                                c_counter_start[idx] += 2
                                match = True
                                break
                        if not match:
                            all_cs_start.append(c)
                            c_counter_start[len(all_cs_start) - 1] = 2
                        if c in l_G:
                            num_G_start += 2
                        if c in l_I:
                            num_I_start += 2
                        if c in l_O:
                            num_O_start += 2
                        if c in l_H:
                            num_H_start += 2
                        if isclose(c.x, 0, 1e-4) and isclose(
                                pp.y, pp_sym.y, 1e-4):
                            num_V_start += 2
                        seen_start.add(p)
                        seen_start.add(p_sym)
                sample = (d, pairs)
                logger.debug("Node score: %d, Node weight: %.2f", d_score,
                             weights[sample_idx])
            else:
                logger.debug("explore")
                sample = node_pairs
                node_all_actions = all_actions
                start_path = ()
                prior_head = prior_root
                seen_start = {"A", "B", "C"}
                all_ks_start = [0]
                k_counter_start = {0: 2}
                all_cs_start = [p_M]
                c_counter_start = {0: 2}
                num_G_start = 1
                num_I_start = 1
                num_O_start = 1
                num_H_start = 1
                num_V_start = 2
            random_state = random.getstate()
            state = {
                "count": count,
                "random_state": random_state,
                "buffer": buffer,
                "buffer_node": buffer_node,
                "start_path": start_path
            }
            with open(os.path.join(log_path, f"{cpu_id}_state.pkl"),
                      "wb") as f:
                pickle.dump(state, f)
    logger.debug("END")


def parallel_level(input):
    """For multiprocessing the level."""
    (action_idx, d, pairs, d_composite_actions, base_level, next_level, prior,
     logger) = input
    level_results = {
        "base_level_diagrams": [],
        "base_level_actions": [],
        "next_level_diagrams": [],
        "next_level_actions": []
    }
    composite_action = d_composite_actions[action_idx]
    if composite_action[0] not in prior[d.depth]:
        return level_results
    d_new, _, new_pairs, d_new_actions = take_composite_action(
        d, composite_action, pairs, "", 0, 0, False, logger)
    if d_new.is_terminal:
        return level_results
    d_new_composite_actions = d_composite_actions[action_idx +
                                                  1:] + get_sym_composite(
                                                      d_new_actions, d_new,
                                                      new_pairs)
    has_decendents = False
    for action in d_new_composite_actions:
        action_key = action[0]
        if action_key in prior[d_new.depth]:
            has_decendents = True
            break
    if not has_decendents:
        return level_results
    if len(d_new.actions) == base_level:
        level_results["base_level_diagrams"].append((d_new, new_pairs))
        level_results["base_level_actions"].append(d_new_composite_actions)
    if len(d_new.actions) == next_level:
        level_results["next_level_diagrams"].append((d_new, new_pairs))
        level_results["next_level_actions"].append(d_new_composite_actions)
    return level_results


def main():
    """Entry point for distributed priority search."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ck-func",
                        default="ordered",
                        choices=["ordered", "unordered"],
                        help="what check_diagram function to use")
    parser.add_argument("--total-samples",
                        type=int,
                        default=2048,
                        help="total number of samples to run")
    parser.add_argument("--nodes",
                        type=int,
                        default=1,
                        help="total number of nodes")
    parser.add_argument("--node-id",
                        type=int,
                        default=0,
                        help="id of the node, starting from 0")
    parser.add_argument("--cpus-per-node",
                        type=int,
                        default=32,
                        help="total number of cpus per node")
    parser.add_argument("--save-dir",
                        type=str,
                        default="./data/",
                        help="directory to save data and log")
    parser.add_argument("--buffer-size",
                        type=int,
                        default=100,
                        help="buffer size")
    parser.add_argument("--per-sample-loop",
                        type=int,
                        default=10,
                        help="number of loops per node sample.")
    parser.add_argument("--stats",
                        type=str,
                        default="./stats_sym",
                        help="path to human data for collecting prior")
    parser.add_argument("--expand",
                        action="store_true",
                        help="whether to perform expansion of head; DON'T USE")
    parser.add_argument("--augment",
                        action="store_true",
                        help="whether to augment prior")
    parser.add_argument("--tree",
                        action="store_true",
                        help="whether to use tree-shaped prior")
    parser.add_argument("--seed", type=int, default=542, help="random seed")
    args = parser.parse_args()

    problem_path = os.path.join(args.save_dir, "problems")
    log_path = os.path.join(args.save_dir, "log")
    if not os.path.isdir(args.save_dir):
        os.makedirs(problem_path)
        os.makedirs(log_path)
    prior = {depth: Counter() for depth in range(1, Diagram().max_depth)}
    stats_files = glob.glob(os.path.join(args.stats, "*.txt"))
    tree_prior_head = TreePriorNode()
    for stats_file in stats_files:
        tree_prior_node = tree_prior_head
        with open(stats_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            if l == "\n":
                break
            parts = l.strip().split(" ")
            depth = int(parts[0])
            if depth == 0 or depth >= Diagram().max_depth:
                continue
            gen_depth = int(parts[1])
            actions = "".join(parts[2:]).split(";")
            dep_action = actions[0]
            left_bracket = dep_action.index("(")
            left_comma = dep_action.index(",")
            action_type = dep_action[left_bracket + 1:left_comma]
            self_sym = actions[1].startswith("None")
            key = (gen_depth, action_type, self_sym)
            prior[depth][key] += 1
            tree_prior_node.dist[key] += 1
            if key not in tree_prior_node.next:
                tree_prior_node.next[key] = TreePriorNode()
            tree_prior_node = tree_prior_node.next[key]
    for depth, depth_dict in prior.items():
        total_sum = sum(depth_dict.values())
        for key in depth_dict:
            depth_dict[key] /= total_sum
    frontiers = [tree_prior_head]
    while len(frontiers) > 0:
        node = frontiers.pop(0)
        if len(node.next) == 0:
            continue
        total_sum = sum(node.dist.values())
        for key in node.dist:
            node.dist[key] /= total_sum
        for new_node in node.next.values():
            frontiers.append(new_node)
    if args.tree:
        prior = tree_prior_head
    elif args.augment:
        accum_prior = {}
        accum_prior[1] = prior[1]
        depth_dict = prior[2]
        depth_m1_dict = prior[1]
        accum_depth_dict = Counter()
        for key in depth_dict:
            if key in depth_m1_dict:
                accum_depth_dict[
                    key] = 1 / 2 * depth_dict[key] + 1 / 2 * depth_m1_dict[key]
            else:
                accum_depth_dict[key] = 1 / 2 * depth_dict[key]
        for key in depth_m1_dict:
            if key in depth_dict:
                continue
            accum_depth_dict[key] = 1 / 2 * depth_m1_dict[key]
        accum_prior[2] = accum_depth_dict
        for depth in range(3, Diagram().max_depth):
            depth_dict = prior[depth]
            depth_m1_dict = prior[depth - 1]
            depth_m2_dict = prior[depth - 2]
            accum_depth_dict = Counter()
            for key in depth_dict:
                if key in depth_m1_dict and key in depth_m2_dict:
                    accum_depth_dict[
                        key] = 1 / 3 * depth_dict[key] + 1 / 3 * depth_m1_dict[
                            key] + 1 / 3 * depth_m2_dict[key]
                elif key in depth_m1_dict:
                    accum_depth_dict[key] = 1 / 3 * depth_dict[
                        key] + 1 / 3 * depth_m1_dict[key]
                elif key in depth_m2_dict:
                    accum_depth_dict[key] = 1 / 3 * depth_dict[
                        key] + 1 / 3 * depth_m2_dict[key]
                else:
                    accum_depth_dict[key] = 1 / 3 * depth_dict[key]
            for key in depth_m1_dict:
                if key in depth_dict:
                    continue
                if key in depth_m2_dict:
                    accum_depth_dict[key] = 1 / 3 * depth_m1_dict[
                        key] + 1 / 3 * depth_m2_dict[key]
                else:
                    accum_depth_dict[key] = 1 / 3 * depth_m1_dict[key]
            for key in depth_m2_dict:
                if key in depth_dict or key in depth_m1_dict:
                    continue
                accum_depth_dict[key] = 1 / 3 * depth_m2_dict[key]
            accum_prior[depth] = accum_depth_dict
        prior = accum_prior
    root = Diagram(mode=0)
    d = root.apply_action(Action(BaseAcuteTriangle, "", "ABC"))
    d_actions = d.new_valid_actions()
    pairs = {"A": "B", "B": "A", "C": "C"}
    first_level_diagrams = []
    first_level_actions = []
    second_level_diagrams = []
    second_level_actions = []
    third_level_diagrams = []
    third_level_actions = []
    fourth_level_diagrams = []
    fourth_level_actions = []
    d_composite_actions = get_sym_composite(d_actions, d, pairs)
    logger = logging.getLogger("root")

    if args.expand:
        with multiprocessing.Pool(processes=args.cpus_per_node) as pool:
            results = pool.map(
                parallel_level,
                [(action_idx, d, pairs, d_composite_actions, 2, 3, prior,
                  logger) for action_idx in range(len(d_composite_actions))])

        for result in results:
            first_level_diagrams.extend(result["base_level_diagrams"])
            first_level_actions.extend(result["base_level_actions"])
            second_level_diagrams.extend(result["next_level_diagrams"])
            second_level_actions.extend(result["next_level_actions"])

        first_layer_diagrams = first_level_diagrams + second_level_diagrams
        first_layer_actions = first_level_actions + second_level_actions

        with multiprocessing.Pool(processes=args.cpus_per_node) as pool:
            results = pool.map(
                parallel_level,
                [(action_idx, first_level_diagrams[idx][0],
                  first_level_diagrams[idx][1], first_level_actions[idx], 3, 4,
                  prior, logger) for idx in range(len(first_level_diagrams))
                 for action_idx in range(len(first_level_actions[idx]))])

        for result in results:
            second_level_diagrams.extend(result["base_level_diagrams"])
            second_level_actions.extend(result["base_level_actions"])
            third_level_diagrams.extend(result["next_level_diagrams"])
            third_level_actions.extend(result["next_level_actions"])

        second_layer_diagrams = second_level_diagrams + third_level_diagrams
        second_layer_actions = second_level_actions + third_level_actions

        with multiprocessing.Pool(processes=args.cpus_per_node) as pool:
            results = pool.map(
                parallel_level,
                [(action_idx, second_level_diagrams[idx][0],
                  second_level_diagrams[idx][1], second_level_actions[idx], 4,
                  5, prior, logger)
                 for idx in range(len(second_level_diagrams))
                 for action_idx in range(len(second_level_actions[idx]))])

        for result in results:
            third_level_diagrams.extend(result["base_level_diagrams"])
            third_level_actions.extend(result["base_level_actions"])
            fourth_level_diagrams.extend(result["next_level_diagrams"])
            fourth_level_actions.extend(result["next_level_actions"])

        third_layer_diagrams = third_level_diagrams + fourth_level_diagrams
        third_layer_actions = third_level_actions + fourth_level_actions

        problem_path = os.path.join(args.save_dir, "problems")
        log_path = os.path.join(args.save_dir, "log")
        os.makedirs(problem_path)
        os.makedirs(log_path)
        total_cpus = args.nodes * args.cpus_per_node
        if total_cpus < len(second_layer_diagrams):
            diagrams = first_layer_diagrams
            actions = first_layer_actions
        elif total_cpus < len(third_layer_diagrams):
            diagrams = second_layer_diagrams
            actions = second_layer_actions
        else:
            diagrams = third_layer_diagrams
            actions = third_layer_actions
        cpus_per_diagram, remainder = divmod(total_cpus, len(diagrams))
        splits = [cpus_per_diagram] * len(diagrams)
        for i in range(1, remainder + 1):
            splits[-i] += 1
        total_cpus_used = 0
        job_assignment = {}
        for diagram_id, cpus in enumerate(splits):
            for i in range(cpus):
                cpu_id = total_cpus_used + i
                job_assignment[cpu_id] = (diagram_id, cpus, i)
            total_cpus_used += cpus
        input_list = []
        per_cpu_loops = args.total_samples // total_cpus
        for node_cpu_id in range(args.cpus_per_node):
            cpu_id = args.node_id * args.cpus_per_node + node_cpu_id
            diagram_id, _, _ = job_assignment[cpu_id]
            diagram_pairs = diagrams[diagram_id]
            diagram_actions = actions[diagram_id]
            # per_split = len(diagram_actions) // total_splits
            # start_idx = split_id * per_split
            # if split_id != total_splits - 1:
            #     end_idx = (split_id + 1) * per_split
            # else:
            #     end_idx = len(diagram_actions)
            # first_start_end = (start_idx, end_idx)
            input_list.append(
                (diagram_pairs, diagram_actions, args.seed + cpu_id,
                 per_cpu_loops, args.buffer_size, args.per_sample_loop,
                 problem_path, log_path, cpu_id, prior, args.ck_func,
                 args.tree))
    else:
        total_cpus = args.nodes * args.cpus_per_node
        input_list = []
        per_cpu_loops = args.total_samples // total_cpus
        for node_cpu_id in range(args.cpus_per_node):
            cpu_id = args.node_id * args.cpus_per_node + node_cpu_id
            input_list.append(
                ((d, pairs), d_composite_actions, args.seed + cpu_id,
                 per_cpu_loops, args.buffer_size, args.per_sample_loop,
                 problem_path, log_path, cpu_id, prior, args.ck_func,
                 args.tree))
    start_time = time.time()
    pool = multiprocessing.Pool(  # pylint: disable=consider-using-with
        processes=args.cpus_per_node,
        initargs=init_worker)
    try:
        # Use imap_unordered for possibly quicker access to results
        for _ in pool.imap_unordered(worker, input_list):
            pass  # This forces the evaluation and any potential error
        pool.close()
        pool.join()
    except KeyboardInterrupt as e:  # pylint: disable=broad-exception-caught
        print(f"An error occurred: {e}")
        pool.terminate()
        pool.join()
    print(f"--- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
    main()
