r"""The rule module. Rule module implements all forward chaining rules. When
writing rules, be careful with possible recursion during fact dependency build.
The ultimate reason for recursion is that the current trigger rule uses linkers
(especially for eqline and eqcircle), whose linked facts are in the new facts'
parents. The derived facts could possibly be the same as those linked facts in
the parent list and appear earlier in the new_facts list to build than the
linked facts.
Make sure
1. The new fact would not be in its parents.
2. When using linkers, the linked facts are checked in eqcls and inverse eqcls.
This step is unnecessary when the linkers correspond to eqcls and inverse eqcls.
3. The new facts of recusion can't possibly precede the counterpart in the
parent list without recusion when built.
4. Any facts in parents preceding trigger types have been properly built.
"""

import importlib

MODULES = [
    "cong", "contri", "eqangle", "eqcircle", "eqline", "eqratio", "midp",
    "para", "perp", "simtri"
]
ALL_MODULES = {
    name: importlib.import_module(f".{name}", __package__)
    for name in MODULES
}
