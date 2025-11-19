r"""A streamlit WebGUI for checking the correctness of the implementation."""
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import,eval-used,redefined-outer-name

import io
import re
import time
import traceback
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from graphlib import TopologicalSorter

import streamlit as st
from graphviz import Source

from tonggeometry.action import Action
from tonggeometry.constructor import *
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import (Fact, Predicate,
                                                     get_fact_dep)
from tonggeometry.inference_engine.primitives import *

CONG_SKIP_FNs = [
    "cong_and_eqline_to_midp", "eqline_and_cong_to_midp",
    "cong_and_eqline_to_cong", "eqline_and_cong_to_cong"
]
EQANGLE_SKIP_FNs = [
    "eqangle_and_eqline_to_eqline", "eqline_and_eqangle_to_eqline",
    "eqangle_and_perp_to_perp", "perp_and_eqangle_to_perp",
    "eqangle_and_para_to_para", "para_and_eqangle_to_para",
    "eqangle_and_eqline_to_para", "eqline_and_eqangle_to_para",
    "eqangle_and_eqline_to_eqangle", "eqline_and_eqangle_to_eqangle",
    "eqangle_and_eqangle_to_eqangle"
]
EQCIRCLE_SKIP_FNs = [
    "eqcircle_to_eqcircle", "eqline_and_eqcircle_to_perp",
    "eqcircle_and_eqline_to_perp", "eqcircle_and_perp_to_eqline",
    "perp_and_eqcircle_to_eqline"
]
EQRATIO_SKIP_FNs = [
    "eqratio_to_cong", "eqratio_and_cong_to_cong", "cong_and_eqratio_to_cong",
    "eqratio_and_eqline_and_eqline_to_eqratio",
    "eqline_and_eqline_and_eqratio_to_eqratio", "eqratio_and_cong_to_eqratio",
    "eqratio_and_eqratio_to_eqratio"
]
PARA_SKIP_FNs = ["para_and_eqline_to_para", "eqline_and_para_to_para"]
PERP_SKIP_FNs = [
    "perp_and_perp_to_eqangle", "eqline_and_perp_to_perp",
    "perp_and_eqline_to_perp"
]
MIDP_SKIP_FNs = ["midp_and_midp_to_eqratio"]
SKIP_FNs = (CONG_SKIP_FNs + EQANGLE_SKIP_FNs + EQCIRCLE_SKIP_FNs +
            EQRATIO_SKIP_FNs + PARA_SKIP_FNs + PERP_SKIP_FNs + MIDP_SKIP_FNs +
            ["add_fact"])
MUST_FNs = [
    "contri_to_cong", "contri_to_eqangle", "eqcircle_to_eqangle",
    "eqangle_to_eqcircle", "simtri_to_eqratio", "simtri_to_eqangle"
]


def pretty(dictionary, indent=0):
    """Pretty print a dictionary."""
    out = ""
    for key, value in dictionary.items():
        out += ' ' * indent + str(key) + ':\n'
        if isinstance(value, dict):
            out += pretty(value, indent + 2)
        else:
            out += ' ' * (indent + 2) + str(value) + '\n'
    return out


def trace_fact_with_exclude(diagram, fact_key, holder, excl: str = ""):
    """Trace facts without objects to exclude."""
    ff = diagram.used_facts[fact_key][0]
    if len(set(ff.dependency).intersection(excl)) == 0:
        for parent_key in ff.parents:
            edge = f"""\"{str(ff)}\" -> \"{str(parent_key)}\"""" + \
                        f"""[label=\"{ff.fn}\", dir=back]"""
            holder[edge] = None
    for parent_key in ff.parents:
        trace_fact_with_exclude(diagram, parent_key, holder, excl)


def write_proof(diagram, fact, new_points=""):
    """Write the proof of the fact"""

    graph = defaultdict(set)
    visited = set()

    def add_to_graph(node):
        if node in visited:
            return
        ff = diagram.used_facts[node][0]
        if len(new_points) > 0 and len(
                set(ff.dependency).intersection(new_points)) == 0:
            return
        visited.add(node)

        parents = ff.parents
        for parent in parents:
            graph[node].add(parent)
        for parent in parents:
            add_to_graph(parent)

    add_to_graph(fact)

    sorter = TopologicalSorter(graph)
    topological_order = list(sorter.static_order())
    line_count = 1
    output = ""
    f_line_count = {}
    for f in topological_order:
        ff = diagram.used_facts[f][0]
        parents = ff.parents
        if len(parents) == 0:
            continue
        all_refs = set()
        line = f"[{line_count}] {f} because "
        reasons = []
        for parent in parents:
            ref = f_line_count.get(parent, 0)
            all_refs.add(ref)
            reasons.append(f"{parent}[{ref}]")
        if ff.fn not in MUST_FNs and len(all_refs) == 1 and (
                len(parents) == 1 or ff.fn in SKIP_FNs):
            f_line_count[f] = all_refs.pop()
            continue
        line += ', '.join(reasons)
        f_line_count[f] = line_count
        line_count += 1
        output += line + f" ({ff.fn})\n"
    return output


def language_convert(cst, f_names, a_names):
    """Convert an action to natural language."""
    # pylint: disable=line-too-long,redefined-outer-name
    constructor_class = globals()[cst]
    to_names = "".join(
        [a_names.pop(0) for _ in range(constructor_class.to_names_len)])
    if cst == "BaseAcuteTriangle":
        return f"{to_names} is an acute triangle. "
    if cst == "AnyPoint":
        return f"{to_names} is a random point between {f_names}. "
    if cst == "MidPoint":
        return f"{to_names} is the mid point of {f_names}. "
    if cst == "ExtendEqual":
        return f"Extend {f_names} to {to_names} such that {f_names[1]} is the mid point of {f_names[0] + to_names}. "
    if cst == "CenterCircle":
        return f"Construct a circle centered at {f_names[0]} with {f_names[1]} on its circumference. "
    if cst == "BisectorLine":
        return f"{to_names} is on the internal bisector line of angle {f_names} and on the line of {f_names[0] + f_names[2]}. "
    if cst == "PerpendicularLine":
        return f"{to_names} is on the line of {f_names[0] + f_names[2]} and {f_names[1] + to_names} is an altitude of triangle {f_names}. "
    if cst == "InCenter":
        return f"{to_names} is the incenter of triangle {f_names}. "
    if cst == "CircumscribedCircle":
        return f"{to_names} is the center of {f_names}'s circumscribed circle. "
    if cst == "AnyArc":
        return f"{to_names} is any point between the arc of {f_names[:2]} (arc of {f_names[:2]} as shown, centered at {f_names[-1]}). "
    if cst == "MidArc":
        return f"{to_names} is the mid point of arc of {f_names[:2]} (arc of {f_names[:2]} as shown, centered at {f_names[-1]}). "
    if cst == "Perpendicular":
        return f"{to_names} is on the line of {f_names[1:]} and angle {to_names + f_names[:2]} is a right angle. "
    if cst == "Parallel":
        return f"{to_names + f_names[0]} is parallel to {f_names[1:3]} and {to_names} is on the line of {f_names[2:]}. "
    if cst == "IntersectLineLine":
        return f"{to_names} is the intersection of {f_names[:2]} and {f_names[-2:]}. "
    if cst == "IntersectLineCircleOn":
        return f"{to_names} is another intersection of line {f_names[:2]} with circle {f_names[-1] + f_names[-2]}. "
    if cst == "IntersectLineCircleOff":
        return f"{to_names} are the intersections of line {f_names[:2]} and circle {f_names[-2:]}. "
    if cst == "IntersectCircleCircle":
        return f"{to_names} are the intersections of circle {f_names[:2]} and circle {f_names[-2:]}. "
    if cst == "IsogonalConjugate":
        return f"{to_names} is the isogonal conjugate of point {f_names[-1]} with respect to triangle {f_names[:-1]}. "
    return ""


st.set_page_config(layout="wide")
css = '''
<style>
    section.main>div {
        padding-bottom: 5rem;
    }
    [data-testid="column"]>div>div>div>div>div {
        overflow: auto;
        height: 70vh;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1rem;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

if "run" not in st.session_state:
    st.session_state["run"] = False

if "d" not in st.session_state:
    st.session_state["d"] = None

if "data" not in st.session_state:
    st.session_state["data"] = []

if "error" not in st.session_state:
    st.session_state["error"] = None

default_actions = """Action(BaseAcuteTriangle, "", "CBA")
Action(PerpendicularLine, "CBA", "E")
Action(PerpendicularLine, "BCA", "F")
Action(IntersectLineLine, "BECF", "H")
Action(CircumscribedCircle, "ABC", "I")
Action(MidPoint, "BC", "M")
Action(IntersectLineCircleOn, "MAI", "D")
Action(CircumscribedCircle, "EFD", "P")
Action(Reflect, "IPD", "G")
Action(IntersectLineCircleOn, "IAI", "L")
Action(IntersectLineCircleOn, "HLI", "J")
Action(PerpendicularLine, "AHM", "N")
Action(IntersectLineLine, "NHBC", "K")"""

default_fact = """Fact("perp", [Angle(*"DGH")])"""

string = st.text_area("Convert string to action", height=100)
all_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
language = ""
if st.button("Convert"):
    converted = ""
    txt_actions = re.split(", |\n", string.strip())
    for txt_action in txt_actions:
        txt_action.strip()
        left = txt_action.index("(")
        right = txt_action.index(")")
        constructor = txt_action[:left]
        from_names = txt_action[left + 1:right]
        from_names = "".join(from_names.split(","))
        converted += f"Action({constructor}, \"{from_names}\")\n"
        language += language_convert(constructor, from_names, all_names)
    st.text(converted)
    st.markdown(language)

txt = st.text_area("Actions", height=300, value=default_actions)

draw_only = st.toggle("Draw only")

if st.button("Run"):
    st.session_state["run"] = True
    st.session_state["data"] = []
    actions = [eval(action) for action in txt.split("\n") if action.strip()]
    d = Diagram()
    for i, action in enumerate(actions):
        with io.StringIO() as buf, redirect_stdout(buf):
            try:
                start = time.time()
                if draw_only:
                    action.s_check = True
                    new_d = d.apply_action(action,
                                           verbose=True,
                                           draw_only=draw_only)
                else:
                    new_d = d.apply_action(action, verbose=True)
                timing = time.time() - start
            except Exception as e:  # pylint: disable=broad-except
                traceback.print_exc()
                buf = buf.getvalue()
                new_d = d.apply_action(action, draw_only=True)
                timing = 1e4
                d = new_d
                db = str(d.database)
                fig = d.draw()
                with open(f"./app_data/step_{i}_inf.txt",
                          "w",
                          encoding="utf-8") as f:
                    f.writelines(buf)
                with open(f"./app_data/step_{i}_db.txt", "w",
                          encoding="utf-8") as f:
                    f.writelines(db)
                fig.savefig(f"./app_data/step_{i}_fig.png")
                st.session_state["error"] = [e, buf, db, fig]
                with st.expander(f"**ERROR: {e}**"):
                    col_img, col_db, col_chain = st.columns([0.25, 0.25, 0.5])
                    with col_chain:
                        st.text(buf)
                    with col_db:
                        st.text(db)
                    with col_img:
                        st.pyplot(fig)
                break
            buf = buf.getvalue() + "\n" + str(new_d.good_facts)
        d = new_d
        db = str(d.database)
        fig = d.draw()
        with open(f"./app_data/step_{i}_inf.txt", "w", encoding="utf-8") as f:
            f.writelines(buf)
        with open(f"./app_data/step_{i}_db.txt", "w", encoding="utf-8") as f:
            f.writelines(db)
        fig.savefig(f"./app_data/step_{i}_fig.png")
        to_names = d.to_names_seq[-1]
        st.session_state["data"].append(
            (action, timing, buf, db, fig, to_names))
        with st.expander(f"**Step {i}: {action} to {to_names}: " +
                         f"{timing:.3f}s per run.**"):
            col_img, col_db, col_chain = st.columns([0.25, 0.25, 0.5])
            with col_chain:
                st.text(buf)
            with col_db:
                st.text(db)
            with col_img:
                st.pyplot(fig)
    st.session_state["d"] = d
elif st.session_state["run"] and st.session_state["data"]:
    for i, data in enumerate(st.session_state["data"]):
        action, timing, buf, db, fig, to_names = data
        with st.expander(f"**Step {i}: {action} to {to_names}: " +
                         f"{timing:.3f}s per run.**"):
            col_img, col_db, col_chain = st.columns([0.25, 0.25, 0.5])
            with col_chain:
                st.text(buf)
            with col_db:
                st.text(db)
            with col_img:
                st.pyplot(fig)
    if st.session_state["error"]:
        e, buf, db, fig = st.session_state["error"]
        with st.expander(f"ERROR: {e}"):
            col_img, col_db, col_chain = st.columns([0.25, 0.25, 0.5])
            with col_chain:
                st.text(buf)
            with col_db:
                st.text(db)
            with col_img:
                st.pyplot(fig)

if st.session_state["run"] and st.session_state["d"]:
    if st.button("Show points"):
        d = st.session_state["d"]
        st.text(pretty(d.point_dict))
    if st.button("Show good facts"):
        d = st.session_state["d"]
        st.text(d.good_facts)
    db_object = st.text_input("Object to check")
    if st.button("Inspect"):
        d = st.session_state["d"]
        st.text(pretty(getattr(d.database, db_object)))
    txt = st.text_area("New predicates")
    predicates = [
        eval(predicate) for predicate in txt.split("\n") if predicate.strip()
    ]
    if st.button("Add predicates"):
        d = deepcopy(st.session_state["d"])
        with io.StringIO() as buf, redirect_stdout(buf):
            d.forward_facts(predicates, verbose=True)
            with st.expander("**Results from interventions**"):
                buf = buf.getvalue()
                db = d.database
                col_db, col_chain = st.columns([0.5, 0.5])
                with col_chain:
                    st.text(buf)
                with col_db:
                    st.text(db)
    fact = st.text_input("Fact to trace", value=default_fact)
    exclude = st.text_input("Objects to exclude", value="")
    new_points = st.text_input("New points constructed", value="")
    if st.button("Prove"):
        d = st.session_state["d"]
        f = eval(fact)
        proven = f in d.used_facts
        st.text(f"Fact known: {proven}")
        if proven:
            st.text(write_proof(d, f, new_points))
            edges = {}
            trace_fact_with_exclude(d, f, edges, exclude)
            graph = "digraph {\n"
            graph += "\n".join(edges)
            graph += "\n}"
            # st.text(graph)
            # st.graphviz_chart(graph)
            s = Source(graph, directory="./app_data/").render(view=False)
    pfact = st.text_input("Fact to construct problem", value=default_fact)
    if st.button("Construct"):
        d = st.session_state["d"]
        f = eval(pfact)
        proven = f in d.used_facts
        st.text(f"Fact known: {proven}")
        if proven:
            fact_dep = get_fact_dep(f)
            proof_dep = d.used_facts[f][0].dependency
            pruned_context = d.prune(fact_dep)
            proof_context = d.prune(proof_dep)
            st.text(pruned_context)
            st.text(proof_context)
    pfact = st.text_input("Fact to score", value=default_fact)
    if st.button("Score"):
        d = st.session_state["d"]
        f = eval(pfact)
        proven = f in d.used_facts
        d.trace_fact(f)
        score, diff = d.score_fact(f)[:2]
        st.text(f"Fact score: {score}; Fact diff: {diff}")
