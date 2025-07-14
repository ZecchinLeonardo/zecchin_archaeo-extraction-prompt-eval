from collections.abc import Callable
from dash.development.base_component import Component
from dash import Dash


DashComponent = tuple[list[Component], Callable[[Dash], None]]
"""A couple of elements with
- a list of children to be added in a div or a layout
- a callable function which initiate the callbacks and any route creation
"""
