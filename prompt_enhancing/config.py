"""Config functions for mkdocs.

See this:
https://daizutabi.github.io/mkapi/usage/config/#configuration-script
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig

    from mkapi.plugins import MkApiPlugin

def before_on_config(config: MkDocsConfig, plugin: MkApiPlugin) -> None:
    """Called before `on_config` event of MkAPI plugin."""

def after_on_config(config: MkDocsConfig, plugin: MkApiPlugin) -> None:
    """Called after `on_config` event of MkAPI plugin."""

def page_title(name: str, depth: int) -> str:
    """Return a page title."""
    return name.split(".")[-1]  # Remove prefix. Default behavior.

def section_title(name: str, depth: int) -> str:
    """Return a section title."""
    return name.split(".")[-1]  # Remove prefix. Default behavior.

def toc_title(name: str, depth: int) -> str:
    """Return a toc title."""
    return name.split(".")[-1]  # Remove prefix. Default behavior.
