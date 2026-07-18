"""Compatibility imports for the validated RuralScape Studio domain types."""

from . import domain as _domain
from .domain import *  # noqa: F403
from .domain import SerializableDataclass, utc_now

__all__ = _domain.__all__
