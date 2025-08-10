"""Interface over the DAG builder of skdag."""

from typing import Literal, NamedTuple
from sklearn.base import TransformerMixin
from functools import reduce
import skdag


class DAGComponent(NamedTuple):
    """A component for the."""

    component_id: str  # the identifier also used for the visualization
    component: TransformerMixin | Literal["passthrough"]


class DAGBuilder:
    """A DAG builder suited for a handling with Transformers and Dataframes."""
    def __init__(self) -> None:
        """Initiate an empty pipeline."""
        self._dag_builder = skdag.DAGBuilder(infer_dataframe=True)

    def add_node(self, component: DAGComponent, deps: list[DAGComponent]=[]):
        """Add a component to the DAG pipeline."""
        self._dag_builder = self._dag_builder.add_step(
            component.component_id,
            component.component,
            deps=[dc.component_id for dc in deps],
        )
        return self

    def add_linearly_chained_nodes(
        self, components: list[DAGComponent], start_anchor: list[DAGComponent]
        = []
    ):
        """Add a chain of components."""

        def update(
            acc_dag_builder: skdag.DAGBuilder,
            deps: list[DAGComponent],
            new_component: DAGComponent,
        ):
            if not deps:
                return acc_dag_builder.add_step(
                    new_component.component_id, new_component.component
                ), [new_component]
            else:
                return acc_dag_builder.add_step(
                    new_component.component_id,
                    new_component.component,
                    deps=[d.component_id for d in deps],
                ), [new_component]

        self._dag_builder, _ = reduce(
            lambda acc, component: update(
                *acc, component
            ),
            components,
            (self._dag_builder, start_anchor),
        )
        return self

    def make_dag(self):
        """"Return the dag."""
        return self._dag_builder.make_dag()
