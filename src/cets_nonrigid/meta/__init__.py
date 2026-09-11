"""Per-series acquisition metadata: discovery from files next to the source,
precedence/conflict resolution against CLI values, provenance reporting."""

from cets_nonrigid.meta.model import MetaConflict, MetaConflictError, Provenance, SeriesMeta
from cets_nonrigid.meta.resolve import Discovered, resolve_series

__all__ = [
    "Discovered",
    "MetaConflict",
    "MetaConflictError",
    "Provenance",
    "SeriesMeta",
    "resolve_series",
]
