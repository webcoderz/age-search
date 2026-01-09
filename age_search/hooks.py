from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type

from sqlalchemy import event
from sqlalchemy.orm import Session

from .mixins_graph import GraphNodeMixin


@dataclass(frozen=True)
class GraphSyncOptions:
    enabled: bool = True
    detach_delete: bool = True
    # If True, graph_upsert runs even if session is flushing bulk changes
    allow_in_flush: bool = False


def install_graph_sync(model: Type[Any], *, options: GraphSyncOptions = GraphSyncOptions()) -> None:
    """
    Install SQLAlchemy ORM event hooks for a model that includes GraphNodeMixin.
    This keeps AGE vertices in sync with relational rows.

    Behavior:
      - after_insert / after_update: graph_upsert()
      - after_delete: graph_delete(detach=...)
    """

    if not issubclass(model, GraphNodeMixin):
        raise TypeError("install_graph_sync expects a model subclassing GraphNodeMixin")

    @event.listens_for(model, "after_insert", propagate=True)
    def _after_insert(mapper, connection, target):  # noqa: ANN001
        if not options.enabled:
            return
        sess: Optional[Session] = Session.object_session(target)
        if sess is None:
            return
        if sess._flushing and not options.allow_in_flush:  # type: ignore[attr-defined]
            return
        target.graph_upsert(sess)

    @event.listens_for(model, "after_update", propagate=True)
    def _after_update(mapper, connection, target):  # noqa: ANN001
        if not options.enabled:
            return
        sess: Optional[Session] = Session.object_session(target)
        if sess is None:
            return
        if sess._flushing and not options.allow_in_flush:  # type: ignore[attr-defined]
            return
        target.graph_upsert(sess)

    @event.listens_for(model, "after_delete", propagate=True)
    def _after_delete(mapper, connection, target):  # noqa: ANN001
        if not options.enabled:
            return
        sess: Optional[Session] = Session.object_session(target)
        if sess is None:
            return
        if sess._flushing and not options.allow_in_flush:  # type: ignore[attr-defined]
            return
        target.graph_delete(sess, detach=options.detach_delete)
