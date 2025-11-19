r"""The forward chaining module."""

from typing import TYPE_CHECKING, List

from tonggeometry.inference_engine.handler import ORDER
from tonggeometry.inference_engine.rule import ALL_MODULES

if TYPE_CHECKING:
    from tonggeometry.diagram import Diagram
    from tonggeometry.inference_engine.predicate import Fact

ALL_RULES = {name: [] for name in ORDER}
for module in ALL_MODULES:
    for function in dir(ALL_MODULES[module]):
        name = function.split("_")[0]
        if name in ORDER:
            ALL_RULES[name].append(getattr(ALL_MODULES[module], function))


def one_step_fc(diagram: 'Diagram', fact: 'Fact') -> List['Fact']:
    """One-step forward chainer

    Deduce all the new facts that can be used with the database. One-step at a
    time.

    Transitive facts in the same class are not necessary as the database handles
    them when the fact is added.
    """
    facts = []
    for rule in ALL_RULES[fact.type]:
        if diagram.terminal_flag:
            return []
        rule_new_facts = rule(diagram, fact)
        facts += rule_new_facts
    return facts
