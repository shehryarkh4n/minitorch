from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    eps_plus = list(vals)
    eps_plus[arg] += epsilon

    eps_minus = list(vals)
    eps_minus[arg] -= epsilon
    return (f(*eps_plus) - f(*eps_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    topo: Iterable[Variable] = []
    visited: Iterable[int] = set()

    def dfs(variable: Variable):
        if variable.unique_id in visited:
            return

        visited.add(variable.unique_id)

        for parent in variable.parents:
            dfs(parent)

        if not variable.is_constant():
            topo.append(variable)

    dfs(variable)
    return topo


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    queue = topological_sort(variable)
    derivs = {variable.unique_id: deriv}

    # We also need a quick way to map id → object for accumulate_derivative calls
    id_to_var = {var.unique_id: var for var in queue}

    for var in reversed(queue):
        d_output = derivs[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(d_output)
            continue

        for parent, d_parent in var.chain_rule(d_output):
            pid = parent.unique_id
            derivs[pid] = derivs.get(pid, 0.0) + d_parent
            id_to_var[pid] = parent  # ensure mapping exists
    return


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
