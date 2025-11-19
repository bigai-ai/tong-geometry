r"""Diagram class for drawing and forward chaining facts."""

import copy
from abc import ABC, abstractmethod
from functools import cached_property
from itertools import combinations, product
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

from tonggeometry.action import Action
from tonggeometry.constructor import (
    AllConstructors, AnyArc, BaseAcuteTriangle, BaseHarmonicQuad,
    BaseInscribedQuad, BaseInscribedTri, BaseParallelogram,
    CircumscribedCircle, IntersectCircleCircle, IntersectLineCircleOff,
    IntersectLineCircleOn, MidArc)
from tonggeometry.constructor.primitives import Circle, Point, on_same_line
from tonggeometry.inference_engine.database import Database
from tonggeometry.inference_engine.fc import one_step_fc
from tonggeometry.inference_engine.predicate import (Fact, OrderedFact,
                                                     Predicate, get_fact_dep)
from tonggeometry.inference_engine.primitives import Angle
from tonggeometry.inference_engine.primitives import Circle as LogicCircle
from tonggeometry.inference_engine.primitives import Ratio, Segment, Triangle
from tonggeometry.util import OrderedSet, isclose

# from matplotlib import font_manager
# from matplotlib.font_manager import FontProperties

# font_path = r"/home/chizhang/.local/share/fonts/Roboto-Regular.ttf"
# font_manager.fontManager.addfont(font_path)
# font = FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font.get_name()
plt.rcParams["font.family"] = "monospace"
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# rc('font', **{'family': 'serif', 'serif': ['Times']})
# rc('text', usetex=True)


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


class Node(ABC):
    """
    A representation of a single game state. MCTS works by constructing a tree
    of these Nodes. Could be e.g. a chess or in our case, the state of a
    geometry problem.
    """

    @cached_property
    @abstractmethod
    def reward(self) -> float:
        "Reward of the state."
        return .0

    @cached_property
    @abstractmethod
    def key(self) -> str:
        "Key to be used in the tree search process."
        return ""

    def __hash__(self) -> int:
        "Nodes must be hashable."
        return hash(self.key)

    def __eq__(self, node) -> bool:
        "Nodes must be comparable."
        return self.key == node.key


class ProblemNode:  # pylint: disable=too-few-public-methods
    """The intermediate state data holder in good problem tree."""

    def __init__(self):
        self.seps = {}
        self.next = {}

    def __contains__(self, key):
        return key in self.next

    def __getitem__(self, key):
        return self.next[key]

    def __setitem__(self, key, val):
        self.next[key] = val


class GoodProblemTree:  # pylint: disable=too-few-public-methods
    """The good problem tree that is basically a hierarchical hash table."""

    def __init__(self):
        self.keys = []
        self.index = 0
        self.tree = ProblemNode()

    def build(self, good_problem, fact):
        """Build the hierarchical hash tree, if in return True."""
        sep = len(good_problem[0])
        tree_node = self.tree
        is_in = True
        tree_seq = good_problem[0] + good_problem[1]
        for elem in tree_seq:
            if elem in tree_node:
                tree_node = tree_node[elem]
            else:
                is_in = False
                tree_node[elem] = ProblemNode()
                tree_node = tree_node[elem]
        if (is_in and sep not in tree_node.seps) or not is_in:
            self.keys.append(good_problem)
            tree_node.seps[sep] = []
            tree_node.seps[sep].append(fact)
        else:
            tree_node.seps[sep].append(fact)

    def retrieve(self, good_problem):
        """Retrieve the good problem's node."""
        node = self.tree
        tree_seq = good_problem[0] + good_problem[1]
        for elem in tree_seq:
            node = node[elem]
        return node

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.keys):
            result = self.keys[self.index]
            self.index += 1
            return result
        raise StopIteration

    def __len__(self):
        return len(self.keys)


class Diagram(Node):
    """The Diagram class.

    Facts cannot be obtained from analytical methods.
    Each diagram has all the facts up to the step, the action sequence applied
    to it, and the corresponding objects in each step.
    """

    def __init__(self,
                 order_flag: bool = False,
                 max_depth: int = 28,
                 max_loop: int = 1.5e6,
                 mode: int = 1):
        # mode = 0 for search time; mode = 1 for inference time
        self.order_flag = order_flag
        self.max_depth = max_depth
        self.max_loop = max_loop
        self.mode = mode

        # IMPORTANT: this string must be sorted in ASCII order
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.all_names = OrderedSet.fromkeys(self.alphabet)

        # object storage
        self.point_dict = OrderedSet()
        self.circle_dict = OrderedSet()
        self.line_to_draw = {}
        self.parent_points = None

        # action sequence
        self.actions = []
        self.to_names_seq = []

        # fact storage (for inference engine)
        self.database = Database()
        self.used_facts = {}

        # bookkeeping
        self.depth = 0
        self.running_action = None
        self.terminal_flag = False
        self.OOB_terminal = False
        self.inf_terminal = False
        self.full_dup = False
        self.over_loop = False

        # checking goodness
        self.good_facts = []
        self.good_facts_meta = []
        self.good_problems = GoodProblemTree()

        self.point_to_depth = {}

    def _copy(self) -> 'Diagram':
        """Return a copy of the diagram.

        shallow copy for changing the container, deep copy for data mutation.
        """
        instance = Diagram(order_flag=self.order_flag,
                           max_depth=self.max_depth,
                           max_loop=self.max_loop,
                           mode=self.mode)
        for attr in ("all_names", "point_dict", "circle_dict", "line_to_draw",
                     "actions", "to_names_seq", "used_facts", "depth",
                     "point_to_depth"):
            setattr(instance, attr, copy.copy(getattr(self, attr)))
        instance.database = copy.deepcopy(self.database)

        return instance

    def apply_action(
        self,
        action: Action,
        checked: bool = False,
        verbose: bool = False,
        draw_only: bool = False,
    ) -> 'Diagram':
        """Apply a new action to the diagram. Order is IMPORTANT!"""

        if self.terminal_flag:
            if verbose:
                print("TERMINAL NODE REACHED")
            return self

        # First build the family tree
        new_instance = self._copy()
        new_instance.parent_points = OrderedSet.fromkeys(
            self.point_dict.keys())

        # Run the construction and forward facts

        # Action may not come with valid to and to_names as, while
        # the action remains the same, to might be invalid in different cases.
        # But if checked, it means to doesn't need to be filtered and to_names
        # have been fixed.

        new_instance.running_action = action
        constructor = action.constructor

        # check if enough points
        if constructor.to_names_len > len(new_instance.all_names):
            new_instance.terminal_flag = True
            if verbose:
                print("NAME VAIL CHECK FAILED")
                print(new_instance.actions)
                print(f"{action} -> ({action.to_names})")
            return new_instance

        # static_check
        if action.s_check is False:
            valid, _ = constructor.static_check(new_instance,
                                                action.from_names,
                                                action.to_names,
                                                draw_only=draw_only)
            if not valid:
                new_instance.terminal_flag = True
                if verbose:
                    print("STATIC CHECK FAILED")
                    print(new_instance.actions)
                    print(f"{action} -> ({action.to_names})")
                return new_instance
        if action.output is None:
            froms = [
                new_instance.point_dict[name] for name in action.from_names
            ]
            to = constructor.compute(froms)
            action.output = to
        else:
            to = action.output

        # runtime_check
        if not checked:
            valid, _ = constructor.runtime_check(new_instance, to)
            if not valid:
                new_instance.terminal_flag = True
                if verbose:
                    print("RUNTIME CHECK FAILED")
                    print(new_instance.actions)
                    print(f"{action} -> ({action.to_names})")
                return new_instance

        to_names = (action.to_names if action.to_names else
                    new_instance.next_point_name(constructor.to_names_len))
        if action.rev_names:
            to_names = to_names[::-1]

        # construct
        from_names = action.from_names
        constructor.construct(new_instance, from_names, to_names, to)

        new_instance.to_names_seq.append(to_names)

        if verbose:
            print("FORWARDING FACTS")
            print("PREVIOUSLY: ", new_instance.actions)
            print(f"THIS: {action} -> ({to_names})")

        if not draw_only:
            new_facts = new_instance.forward_facts(
                predicates_to_add=constructor.given_facts(
                    from_names=from_names, to_names=to_names),
                verbose=verbose)
        else:
            new_facts = []

        if new_instance.terminal_flag:
            if verbose:
                print("NUMERIC ERROR ENCOUTERED")
            return new_instance

        # Finally add the action
        new_instance.actions.append(action)
        new_instance.running_action = None
        new_instance.depth += 1
        for p in to_names:
            new_instance.point_to_depth[p] = new_instance.depth
        if action.constructor_name not in [
                "AnyArc", "AnyPoint", "CenterCircle"
        ]:
            new_instance.check_good(new_facts)

        if verbose:
            print(new_instance.reward)

        return new_instance

    @classmethod
    def apply_actions(
        cls,
        actions: List[Action],
        checked: bool = False,
        verbose: bool = False,
        draw_only: bool = False,
    ) -> 'Diagram':
        """Apply a list of actions to the diagram."""
        diagram = cls()
        for action in actions:
            diagram = diagram.apply_action(action,
                                           checked=checked,
                                           verbose=verbose,
                                           draw_only=draw_only)
        return diagram

    def add_point(self, name: str, point: Point):
        """Add a new point to the diagram."""
        self.point_dict[name] = point
        self.all_names.pop(name)

    def add_circle(self, name: str, circle: Circle):
        """Add a new circle to the diagram."""
        self.circle_dict[name] = circle

    def add_line_to_draw(self, l_name: str):
        """Add a line to draw"""
        self.line_to_draw[l_name] = None

    def next_point_name(self, k=1) -> str:
        """The next point name to be used."""
        it = iter(self.all_names)
        return "".join([next(it) for _ in range(k)])

    def draw(self, saved_path: str = None) -> plt.Figure:
        """Draw the diagram and save to path if specified."""
        fig, ax = plt.subplots()

        # draw points
        for name, p in self.point_dict.items():
            ax.scatter(p.x, p.y, c="k", marker=".")
            # signx = 1 if p.x >= 0 else -1
            # signy = 1 if p.y >= 0 else -1
            ax.text(p.x + 0.2, p.y + 0.2, name, c="#EA4335")

        # draw lines
        for l_name in self.line_to_draw:
            A = self.point_dict[l_name[0]]
            B = self.point_dict[l_name[1]]
            ax.plot((A.x, B.x), (A.y, B.y), c="#4285F4")

        # draw circles
        for _, c in self.circle_dict.items():
            fig.gca().add_patch(
                plt.Circle((c.x, c.y), c.r, fill=False, color="#34A853"))

        ax.set_axis_off()
        ax.axis("equal")

        if saved_path:
            format = saved_path.split(".")[-1]
            fig.savefig(saved_path, format=format)

        return fig

    def get_mirror_action(self,
                          action: Action,
                          axis: Optional[str] = None) -> Action:
        """Get mirror action with respect to the perpendicular of AB, AC, BC."""
        constructor = action.constructor
        action_type = action.constructor_name
        from_names = action.from_names
        if action_type == "BaseAcuteTriangle":
            if axis == "AB":
                return Action(BaseAcuteTriangle, from_names="", to_names="BAC")
            if axis == "AC":
                return Action(BaseAcuteTriangle, from_names="", to_names="CBA")
            if axis == "BC":
                return Action(BaseAcuteTriangle, from_names="", to_names="ACB")
            return Action(BaseAcuteTriangle, from_names="", to_names="ABC")
        if action_type in ["MidArc", "AnyArc"]:  # when mirror, always flip
            return Action(constructor,
                          from_names=from_names[1] + from_names[0] +
                          from_names[2])
        if action_type == "IntersectCircleCircle":  # when mirror, always flip
            return Action(constructor,
                          from_names=from_names[2] + from_names[3] +
                          from_names[0] + from_names[1])
        return Action(constructor, from_names=from_names)

    # The following methods are for MCTS search

    @cached_property
    def key(self) -> str:
        return ";".join(["Null()"] + [str(a) for a in self.actions])

    @cached_property
    def is_terminal(self) -> bool:
        """Check if this branch should terminate."""
        return self.terminal_flag or self.depth >= self.max_depth or len(
            self.all_names) == 0

    @property
    def reward(self) -> int:
        # if terminal: return -1
        # otherwise
        # reward should be computed as functions of a, b, c, d where
        # a = len(self.prune(elements in the proof trace))
        # b = len(self.actions)
        # c = len(self.prune(elements in the conclusion))
        # d = b - delta (delta is the number of actions after conclusion)
        # return (a / b) * (c / d) * 10 ** (a - c)
        if self.terminal_flag:
            return -(self.max_depth - self.depth)
        if self.mode == 0:
            if len(self.good_problems) == 0:
                return 0
            return len(self.good_problems)
        if len(self.good_facts) == 0:
            return 0
        max_score = 0
        for fact in self.good_facts:
            self.trace_fact(fact)
            score = self.score_fact(fact)[0]
            max_score = max(max_score, score)
        return max_score

    def new_valid_actions(self, organize: bool = False):
        """Get all valid actions for the current diagram."""
        if self.is_terminal:
            return []

        if self.order_flag:
            keys = sorted(list(self.point_dict.keys()))
            point_dict = OrderedSet()
            for key in keys:
                point_dict[key] = self.point_dict[key]
            self.point_dict = point_dict

        valid_actions = []
        # not constructed yet
        if len(self.actions) == 0:
            valid_actions.append(Action(BaseAcuteTriangle, "", "ABC"))
            valid_actions.append(Action(BaseParallelogram, "", "ABCD"))
            valid_actions.append(Action(BaseInscribedTri, "", "ABCD"))
            valid_actions.append(Action(BaseInscribedQuad, "", "ABCDE"))
            valid_actions.append(Action(BaseHarmonicQuad, "", "ABCDE"))
            return valid_actions

        counter = OrderedSet()

        if self.actions[-1].constructor_name != "CenterCircle":
            for constructor in AllConstructors:
                valid_actions_from_constructor = constructor.valid_actions(
                    self, self.to_names_seq[-1], order=self.order_flag)
                valid_actions += valid_actions_from_constructor
                if len(valid_actions_from_constructor) > 0:
                    counter[constructor.__name__] = len(
                        valid_actions_from_constructor)
        else:
            circle = self.actions[-1].from_names
            for constructor in [
                    AnyArc, MidArc, IntersectLineCircleOn,
                    IntersectLineCircleOff, IntersectCircleCircle
            ]:
                valid_actions_from_constructor = constructor.valid_actions(
                    self, circle, pick_rep=False, from_circle=True)
                valid_actions += valid_actions_from_constructor
                if len(valid_actions_from_constructor) > 0:
                    counter[constructor.__name__] = len(
                        valid_actions_from_constructor)
        if organize:
            return valid_actions, counter
        return valid_actions

    def check_good(self, new_facts: List[Fact]):
        """good means a problem requires auxiliary construciton to solve,
        i.e., a newly created fact only depends on former geometry objects
        """
        to_names = self.to_names_seq[-1]
        for fact in new_facts:
            if self.mode == 0 and fact.type in [
                    "eqratio", "eqangle", "midp", "contri"
            ]:
                continue
            # trivial check
            if fact.type == "eqangle":
                a1, a2 = fact.objects
                # special angle: line
                if self.database.is_eqline(
                        a1.s1, a1.s2) or self.database.is_eqline(a2.s1, a2.s2):
                    continue
                # special angle: perp
                if self.database.is_perp(a1) or self.database.is_perp(a2):
                    continue
                # special trivial case: eqline
                if self.database.is_eqline(
                        a1.s1, a2.s1) or self.database.is_eqline(
                            a1.s2, a2.s2) or (
                                self.database.is_eqline(a1.s1, a2.s2)
                                and self.database.is_eqline(a1.s2, a2.s1)):
                    continue
                # special trivial case: para
                if self.database.is_para(
                        a1.s1, a2.s1) or self.database.is_para(
                            a1.s2,
                            a2.s2) or (self.database.is_para(a1.s1, a2.s2) and
                                       self.database.is_para(a1.s2, a2.s1)):
                    continue
                # special case: non optimal
                A, B, C = a1.name
                D, E, F = a2.name
                l_AB = Segment(A, B)
                l_AB_rep = self.database.inverse_eqline[l_AB]
                it = iter(self.database.lines_points[l_AB_rep])
                pA = next(it)
                if pA == B:
                    pA = next(it)
                if pA != A:
                    continue
                l_BC = Segment(B, C)
                l_BC_rep = self.database.inverse_eqline[l_BC]
                it = iter(iter(self.database.lines_points[l_BC_rep]))
                pC = next(it)
                if pC == B:
                    pC = next(it)
                if pC != C:
                    continue
                l_DE = Segment(D, E)
                l_DE_rep = self.database.inverse_eqline[l_DE]
                it = iter(self.database.lines_points[l_DE_rep])
                pD = next(it)
                if pD == E:
                    pD = next(it)
                if pD != D:
                    continue
                l_EF = Segment(E, F)
                l_EF_rep = self.database.inverse_eqline[l_EF]
                it = iter(self.database.lines_points[l_EF_rep])
                pF = next(it)
                if pF == E:
                    pF = next(it)
                if pF != F:
                    continue
            if fact.type == "eqratio":
                r1, r2 = fact.objects
                # special ratio: cong
                if self.database.is_cong(
                        r1.s1, r1.s2) or self.database.is_cong(r2.s1, r2.s2):
                    continue
                # special ratio: midp
                if r1.s1 in self.database.inverse_midp and r2.s1 in self.database.inverse_midp:
                    r1_m = self.database.inverse_midp[r1.s1]
                    r2_m = self.database.inverse_midp[r2.s1]
                    if r1.s2 in [
                            Segment(r1_m, r1.s1.p1),
                            Segment(r1_m, r1.s1.p2)
                    ] and r2.s2 in [
                            Segment(r2_m, r2.s1.p1),
                            Segment(r2_m, r2.s1.p2)
                    ]:
                        continue
                if r1.s2 in self.database.inverse_midp and r2.s2 in self.database.inverse_midp:
                    r1_m = self.database.inverse_midp[r1.s2]
                    r2_m = self.database.inverse_midp[r2.s2]
                    if r1.s1 in [
                            Segment(r1_m, r1.s2.p1),
                            Segment(r1_m, r1.s2.p2)
                    ] and r2.s1 in [
                            Segment(r2_m, r2.s2.p1),
                            Segment(r2_m, r2.s2.p2)
                    ]:
                        continue
                # special trivial case
                if self.database.is_cong(
                        r1.s1, r2.s1) or self.database.is_cong(
                            r1.s2,
                            r2.s2) or (self.database.is_cong(r1.s1, r2.s2) and
                                       self.database.is_cong(r1.s2, r2.s1)):
                    continue
            # special case: non optimal
            if fact.type == "para":
                s1, s2 = fact.objects
                l_AB = s1
                l_AB_rep = self.database.inverse_eqline[l_AB]
                it = iter(self.database.lines_points[l_AB_rep])
                pA = next(it)
                pB = next(it)
                if s1 != Segment(pA, pB):
                    continue
                l_CD = s2
                l_CD_rep = self.database.inverse_eqline[l_CD]
                it = iter(self.database.lines_points[l_CD_rep])
                pC = next(it)
                pD = next(it)
                if s2 != Segment(pC, pD):
                    continue
            if fact.type == "eqcircle" and (fact.objects[0].center
                                            or fact.objects[1].center):
                continue
            fact_points = get_fact_dep(fact)
            if self.mode == 1 and any(fact_point in to_names
                                      for fact_point in fact_points):
                continue
            context_actions = self.prune(fact_points)
            proof_points = self.used_facts[fact][0].dependency
            ca_actions = self.prune(proof_points)

            context_idx = get_idx_list(context_actions, self.actions)
            ca_idx = get_idx_list(ca_actions, self.actions)
            set_context_idx = set(context_idx)
            set_ca_idx = set(ca_idx)
            if ca_idx[-1] == len(
                    self.actions) - 1 and set_context_idx < set_ca_idx:
                if self.mode == 0:
                    aux_idx = sorted(list(set_ca_idx - set_context_idx))
                    problem = (tuple(context_idx), tuple(aux_idx))
                    self.good_problems.build(problem, fact)
                if self.mode == 1 and context_idx == list(
                        range(len(context_idx))):
                    aux = set_ca_idx - set_context_idx
                    self.good_facts.append(fact)
                    self.good_facts_meta.append((max(context_idx), min(aux)))

    def prune(self, objs: str) -> List[Action]:
        """Prune to get the minimum diagram from the objects."""

        def find_circle_proof_points(logic_c: LogicCircle, c: Circle):
            """Find related points in circle actions' preconditions."""
            found = False
            for name, cc in self.circle_dict.items():
                logic_cc = LogicCircle(name[0], [name[1]])
                if c == cc and self.database.is_eqcircle(logic_c, logic_cc):
                    found = True
                    break
            if not found:
                return ""
            if logic_c == logic_cc:
                return ""
            f = Fact("eqcircle", [logic_c, logic_cc])
            if f not in self.used_facts:
                return ""
            self.build_fact(f)
            return self.used_facts[f][0].dependency

        pruned_actions = []
        related_objs = set(objs)
        circles_needed = []
        for action, to_names in zip(self.actions[::-1],
                                    self.to_names_seq[::-1]):
            to = action.output
            if action.constructor_name == "CenterCircle":
                c = to[0]
                if c in circles_needed:
                    pruned_actions.append(
                        Action(action.constructor, action.from_names, to_names,
                               to))
                    related_objs.update(action.from_names)
            else:
                if len(related_objs.intersection(to_names)) > 0:
                    pruned_actions.append(
                        Action(action.constructor, action.from_names, to_names,
                               to))
                    related_objs.update(action.from_names)
                    if action.constructor_name == "IntersectCircleCircle":
                        O1, A, O2, B = action.from_names
                        c_OA = Circle(self.point_dict[O1], self.point_dict[A])
                        c_OB = Circle(self.point_dict[O2], self.point_dict[B])
                        circles_needed.append(c_OA)
                        circles_needed.append(c_OB)
                        logic_c_OA = LogicCircle(O1, [A])
                        logic_c_OB = LogicCircle(O2, [B])
                        related_objs.update(
                            find_circle_proof_points(logic_c_OA, c_OA))
                        related_objs.update(
                            find_circle_proof_points(logic_c_OB, c_OB))
                    elif action.constructor_name == "IntersectLineCircleOn":
                        _, A, O = action.from_names
                        c_OA = Circle(self.point_dict[O], self.point_dict[A])
                        circles_needed.append(c_OA)
                        logic_c_OA = LogicCircle(O, [A])
                        related_objs.update(
                            find_circle_proof_points(logic_c_OA, c_OA))
                    elif action.constructor_name == "IntersectLineCircleOff":
                        _, _, O, A = action.from_names
                        c_OA = Circle(self.point_dict[O], self.point_dict[A])
                        circles_needed.append(c_OA)
                        logic_c_OA = LogicCircle(O, [A])
                        related_objs.update(
                            find_circle_proof_points(logic_c_OA, c_OA))
                    elif action.constructor_name == "MidArc":
                        A, B, O = action.from_names
                        c_OA = Circle(self.point_dict[O], self.point_dict[A])
                        circles_needed.append(c_OA)
                        logic_c_OA = LogicCircle(O, [A])
                        logic_c_OB = LogicCircle(O, [B])
                        related_objs.update(
                            find_circle_proof_points(logic_c_OA, c_OA))
                        f = Fact("eqcircle", [logic_c_OA, logic_c_OB])
                        if f in self.used_facts:
                            self.build_fact(f)
                            related_objs.update(
                                self.used_facts[f][0].dependency)
                    elif action.constructor_name == "AnyArc":
                        A, B, O = action.from_names
                        c_OA = Circle(self.point_dict[O], self.point_dict[A])
                        circles_needed.append(c_OA)
                        logic_c_OA = LogicCircle(O, [A])
                        logic_c_OB = LogicCircle(O, [B])
                        related_objs.update(
                            find_circle_proof_points(logic_c_OA, c_OA))
                        f = Fact("eqcircle", [logic_c_OA, logic_c_OB])
                        if f in self.used_facts:
                            self.build_fact(f)
                            related_objs.update(
                                self.used_facts[f][0].dependency)
        return pruned_actions[::-1]

    def order(self, new_action: Action):
        """Make the action's from_names and to_names in order. Use pre apply."""
        replacement_mapping = {}
        inverse_mapping = {}
        if new_action.constructor_name == "CenterCircle":
            return replacement_mapping, inverse_mapping
        to_names_seq = new_action.to_names
        it = iter(self.all_names)
        for char in to_names_seq:
            map_char = next(it)
            replacement_mapping[char] = map_char
            inverse_mapping[map_char] = char
        return replacement_mapping, inverse_mapping

    def order_action(self,
                     action: Action,
                     mapping: dict,
                     convert=True) -> Action:
        """Order an action."""
        from_names = "".join([mapping[p_name] for p_name in action.from_names])
        if action.constructor_name == "CenterCircle" or action.to_names is None:
            to_names = None
        else:
            to_names = "".join([mapping[p_name] for p_name in action.to_names])
        new_action = Action(action.constructor, from_names, to_names,
                            action.output)
        if convert:
            new_action.convert(self)
        return new_action

    def order_fact(self, fact: Fact, mapping: dict) -> Fact:
        """Order a fact."""
        new_objs = []
        for obj in fact.objects:
            if isinstance(obj, str):
                new_objs.append(mapping[obj])
            elif isinstance(obj, Segment):
                new_objs.append(
                    Segment(*[mapping[p] for p in [obj.p1, obj.p2]]))
            elif isinstance(obj, Angle):
                new_objs.append(
                    Angle(*[mapping[p] for p in [obj.p1, obj.p2, obj.p3]]))
            elif isinstance(obj, Ratio):
                new_s1 = Segment(*[mapping[p] for p in [obj.s1.p1, obj.s1.p2]])
                new_s2 = Segment(*[mapping[p] for p in [obj.s2.p1, obj.s2.p2]])
                new_objs.append(Ratio(new_s1, new_s2))
            elif isinstance(obj, Triangle):
                new_objs.append(
                    Triangle(*[mapping[p] for p in [obj.p1, obj.p2, obj.p3]]))
            elif isinstance(obj, LogicCircle):
                new_center = mapping[
                    obj.center] if obj.center is not None else None
                new_points = [mapping[p] for p in obj.points]
                new_objs.append(LogicCircle(new_center, new_points))
        new_fact = Fact(fact.type, new_objs)
        return new_fact

    def check_num(self, fact: Fact) -> bool:
        """Check if a fact holds numerically."""
        fact_points = get_fact_dep(fact)
        if any(p not in self.point_dict for p in fact_points):
            return False
        if fact.type == "eqline":
            s1, s2 = fact.objects
            A, B = [self.point_dict[p] for p in [s1.p1, s1.p2]]
            C, D = [self.point_dict[p] for p in [s2.p1, s2.p2]]
            return on_same_line(A, B, C) and on_same_line(A, B, D)
        if fact.type == "eqcircle":
            c1, c2 = fact.objects
            if c1.center and c2.center:
                center = self.point_dict[c1.center]
                p1 = self.point_dict[c1.min_point]
                p2 = self.point_dict[c2.min_point]
                cp1 = (p1 - center).norm()
                cp2 = (p2 - center).norm()
                return isclose(cp1, cp2, 1e-5)
            if c1.center or c2.center:
                if c1.center:
                    center = self.point_dict[c1.center]
                    p_anchor = self.point_dict[c1.min_point]
                    p_others = [self.point_dict[p] for p in c2.points]
                else:
                    center = self.point_dict[c2.center]
                    p_anchor = self.point_dict[c2.min_point]
                    p_others = [self.point_dict[p] for p in c1.points]
                c_anchor = (p_anchor - center).norm()
                c_others = [(p - center).norm() for p in p_others]
                return all(
                    isclose(c_anchor, c_other, 1e-5) for c_other in c_others)
            if on_same_line(*[self.point_dict[p]
                              for p in c1.points]) or on_same_line(
                                  *[self.point_dict[p] for p in c2.points]):
                return False
            center1 = CircumscribedCircle.compute(
                [self.point_dict[p] for p in c1.points])[0]
            center2 = CircumscribedCircle.compute(
                [self.point_dict[p] for p in c2.points])[0]
            return isclose(center1.x, center2.x, 1e-5) and isclose(
                center1.y, center2.y, 1e-5)
        if fact.type == "perp":
            a = fact.objects[0]
            A, B, C = [self.point_dict[p] for p in [a.p1, a.p2, a.p3]]
            BA = A - B
            BC = C - B
            return isclose(BA.dot(BC), 0, 1e-5)
        if fact.type == "cong":
            s1, s2 = fact.objects
            A, B = [self.point_dict[p] for p in [s1.p1, s1.p2]]
            C, D = [self.point_dict[p] for p in [s2.p1, s2.p2]]
            AB = (B - A).norm()
            CD = (D - C).norm()
            return isclose(AB, CD, 1e-5)
        if fact.type == "para":
            s1, s2 = fact.objects
            A, B = [self.point_dict[p] for p in [s1.p1, s1.p2]]
            C, D = [self.point_dict[p] for p in [s2.p1, s2.p2]]
            AB = B - A
            CD = D - C
            return isclose(AB.cross(CD), 0, 1e-5)
        if fact.type == "eqangle":
            a1, a2 = fact.objects
            A, B, C = [self.point_dict[p] for p in [a1.p1, a1.p2, a1.p3]]
            D, E, F = [self.point_dict[p] for p in [a2.p1, a2.p2, a2.p3]]
            BA = A - B
            BC = C - B
            ED = D - E
            EF = F - E
            cos_ABC = BA.dot(BC) / (BA.norm() * BC.norm())
            cos_DEF = ED.dot(EF) / (ED.norm() * EF.norm())
            dir_ABC = BA.cross(BC)
            dir_DEF = ED.cross(EF)
            if isclose(dir_ABC * dir_DEF, 0, 1e-5) or dir_ABC * dir_DEF > 0:
                return isclose(cos_ABC, cos_DEF, 1e-5)
            return isclose(cos_ABC + cos_DEF, 0, 1e-5)
        if fact.type == "eqratio":
            r1, r2 = fact.objects
            s1, s2 = r1.s1, r1.s2
            s3, s4 = r2.s1, r2.s2
            A, B = [self.point_dict[p] for p in [s1.p1, s1.p2]]
            C, D = [self.point_dict[p] for p in [s2.p1, s2.p2]]
            E, F = [self.point_dict[p] for p in [s3.p1, s3.p2]]
            G, H = [self.point_dict[p] for p in [s4.p1, s4.p2]]
            AB = (B - A).norm()
            CD = (D - C).norm()
            EF = (F - E).norm()
            GH = (H - G).norm()
            return isclose(AB / CD, EF / GH, 1e-5)
        if fact.type == "midp":
            m, s = fact.objects
            M = self.point_dict[m]
            A, B = [self.point_dict[p] for p in [s.p1, s.p2]]
            M_p = (A + B) / 2
            return isclose(M.x, M_p.x, 1e-5) and isclose(M.y, M_p.y, 1e-5)
        if fact.type == "simtri":
            t1, t2 = fact.objects
            A, B, C = [self.point_dict[p] for p in [t1.p1, t1.p2, t1.p3]]
            D, E, F = [self.point_dict[p] for p in [t2.p1, t2.p2, t2.p3]]
            AB = (B - A).norm()
            BC = (C - B).norm()
            AC = (C - A).norm()
            DE = (E - D).norm()
            EF = (F - E).norm()
            DF = (F - D).norm()
            r_AB_DE = AB / DE
            r_BC_EF = BC / EF
            r_AC_DF = AC / DF
            return isclose(r_AB_DE, r_BC_EF, 1e-5) and isclose(
                r_BC_EF, r_AC_DF, 1e-5) and isclose(r_AB_DE, r_AC_DF, 1e-5)
        if fact.type == "contri":
            t1, t2 = fact.objects
            A, B, C = [self.point_dict[p] for p in [t1.p1, t1.p2, t1.p3]]
            D, E, F = [self.point_dict[p] for p in [t2.p1, t2.p2, t2.p3]]
            AB = (B - A).norm()
            BC = (C - B).norm()
            AC = (C - A).norm()
            DE = (E - D).norm()
            EF = (F - E).norm()
            DF = (F - D).norm()
            return isclose(AB, DE, 1e-5) and isclose(BC, EF, 1e-5) and isclose(
                AC, DF, 1e-5)
        return False

    ###############################
    ###### INFERENCE ENGINE
    ###############################
    def build_trivial_facts(self, new_points: Iterable[str]) -> List[Fact]:
        """Build all lines from old_points and new_points."""
        if self.order_flag:
            new_points = sorted(new_points)
        facts = []
        old_points = self.parent_points
        for old_p, new_p in product(old_points, new_points):
            s = Segment(old_p, new_p)
            f = Fact("eqline", [s, s])
            facts.append(f)
        for new_p1, new_p2 in combinations(new_points, 2):
            s = Segment(new_p1, new_p2)
            f = Fact("eqline", [s, s])
            facts.append(f)
        return facts

    def build_fact(self, fact_key: Fact):
        """Build a fact's dependency points on the forward process."""
        if self.used_facts[fact_key][1]:
            return
        fact = self.used_facts[fact_key][0]
        dependency = set()
        max_depth = 0
        for parent_key in fact.parents:
            parent = self.used_facts[parent_key][0]
            if not self.used_facts[parent_key][1]:
                self.build_fact(parent)
            dependency.update(parent.dependency)
            max_depth = max(max_depth, parent.depth)
        fact.depth = max_depth + 1
        dependency.update(get_fact_dep(fact))
        fact.dependency = "".join(dependency)
        self.used_facts[fact_key][1] = True

    def trace_fact(self, fact_key: Fact):
        """Return dependency graph."""
        fact = self.used_facts[fact_key][0]
        if len(fact.edges) > 0:
            return fact.edges
        for parent_key in fact.parents:
            edge = f"""\"{str(fact)}\" -> \"{str(parent_key)}\"""" + \
                        f"""[label=\"{fact.fn}\", dir=back]"""
            fact.edges[edge] = None
            # fact.nodes[parent_key.key] = None
        for parent_key in fact.parents:
            parent = self.used_facts[parent_key][0]
            if len(parent.edges) > 0:
                fact.edges.update(parent.edges)
                # fact.nodes.update(parent.nodes)
            else:
                self.trace_fact(parent_key)
                fact.edges.update(parent.edges)
                # fact.nodes.update(parent.nodes)
        return fact.edges

    def score_fact(self, fact_key: Fact) -> Tuple[int, int, int, str]:
        """Score a fact's difficulty."""
        fact = self.used_facts[fact_key][0]
        fact_points = get_fact_dep(fact)
        context_actions = self.prune(fact_points)
        proof_points = self.used_facts[fact][0].dependency
        ca_actions = self.prune(proof_points)
        diff = len(ca_actions) - len(context_actions)
        score = len(fact.edges)
        depth = fact.depth
        return score, diff, depth, fact_points

    def forward_facts(
        self,
        predicates_to_add: List[Predicate],
        verbose=False,
    ) -> List[Fact]:
        """Update the database IN PLACE, meaning that call `self.forward_facts`
        will update self.database. Breadth-first search.

        Returns the new facts in the forward chaining process.
        """
        facts_to_add = OrderedFact()

        if verbose:
            print("INITIALIZING")

        trivial_facts = self.build_trivial_facts(self.to_names_seq[-1])
        for fact in trivial_facts:
            facts_to_add.add(fact)

        new_facts = []
        for p in predicates_to_add:
            to_add = self.database.predicate_to_fact(p)
            facts_to_add.add(to_add)

        if verbose:
            print(facts_to_add)
            print("CHAINING STARTS")
            print("=" * 80, "\n")

        loop = 0
        while not facts_to_add.is_empty():
            new_facts_to_add = OrderedFact()
            while not facts_to_add.is_empty():
                loop += 1
                if loop > self.max_loop:
                    self.terminal_flag = True
                    self.inf_terminal = True
                    self.over_loop = True
                    return []
                if verbose:
                    print(f"LOOP {loop}")
                fact, must_keep = facts_to_add.first()
                if verbose:
                    print("TRYING: " + str(fact) + f"[{fact.fn}]" +
                          str(fact.parents))

                if fact in self.used_facts:
                    if verbose:
                        print("FACT USED")
                    continue

                filtered_fact = self.database.filter(fact)
                if must_keep or filtered_fact is not None:
                    self.used_facts[fact] = [fact, False]

                if filtered_fact is None:
                    if verbose:
                        print("FACT FILTERED")
                    continue

                if verbose:
                    print("FACT RUNNING")

                new_facts.append(fact)
                facts_from_db = self.database.add_fact(fact)
                facts_from_fc = one_step_fc(self, fact)
                if self.terminal_flag:
                    return []
                for f in facts_from_db + facts_from_fc:
                    new_facts_to_add.add(f)

                if verbose:
                    print("=" * 80, "\n")
            facts_to_add = new_facts_to_add

        for fact in new_facts:
            try:
                self.build_fact(fact)
            except Exception as e:
                print(e)
                print(str(fact) + f"[{fact.fn}]" + str(fact.parents))
                print(self.actions)
                print(self.running_action)
                # print(self.database)
                raise e

        return new_facts
