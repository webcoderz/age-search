from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from .cypher import cypher_json


@dataclass
class UnionFind:
    parent: dict[int, int]
    size: dict[int, int]

    @classmethod
    def from_nodes(cls, nodes: Iterable[int]) -> "UnionFind":
        parent = {}
        size = {}
        for n in nodes:
            i = int(n)
            parent[i] = i
            size[i] = 1
        return cls(parent=parent, size=size)

    def find(self, x: int) -> int:
        x = int(x)
        # path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return
        # union by size
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def connected_components(nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    """
    Simple community detection baseline: connected components (undirected).
    """
    node_list = [int(n) for n in nodes]
    uf = UnionFind.from_nodes(node_list)
    for a, b in edges:
        if int(a) not in uf.parent:
            uf.parent[int(a)] = int(a)
            uf.size[int(a)] = 1
        if int(b) not in uf.parent:
            uf.parent[int(b)] = int(b)
            uf.size[int(b)] = 1
        uf.union(int(a), int(b))

    groups: dict[int, list[int]] = {}
    for n in uf.parent.keys():
        r = uf.find(n)
        groups.setdefault(r, []).append(n)

    # stable-ish output: larger communities first, then by root id
    out = list(groups.values())
    out.sort(key=lambda g: (-len(g), min(g)))
    for g in out:
        g.sort()
    return out


def graph_edge_list_ids(
    session: Session,
    *,
    graph_name: str,
    label: str,
    edge: str,
    direction: str = "both",  # out|in|both
    limit: int = 200000,
) -> list[tuple[int, int]]:
    """
    Fetch an edge list as (src_id, dst_id) pairs from AGE.

    Assumes nodes store an integer `id` property (same convention used across this package).
    """
    if direction == "out":
        pat = f"(a:{label})-[:{edge}]->(b:{label})"
    elif direction == "in":
        pat = f"(a:{label})<-[:{edge}]-(b:{label})"
    else:
        pat = f"(a:{label})-[:{edge}]-(b:{label})"

    cy = f"""
    MATCH {pat}
    RETURN [a.id, b.id]
    LIMIT {int(limit)}
    """
    rows = cypher_json(session, cy, graph_name=graph_name)
    out: list[tuple[int, int]] = []
    for r in rows:
        if not r or not isinstance(r, list) or len(r) < 2:
            continue
        out.append((int(r[0]), int(r[1])))
    return out


def graph_connected_components(
    session: Session,
    *,
    graph_name: str,
    label: str,
    edge: str,
    direction: str = "both",
    limit_edges: int = 200000,
    nodes: Optional[Iterable[int]] = None,
) -> list[list[int]]:
    """
    Connected-components "communities" for a subgraph in AGE.

    - If `nodes` is None, node set is inferred from edges.
    """
    edges = graph_edge_list_ids(
        session,
        graph_name=graph_name,
        label=label,
        edge=edge,
        direction=direction,
        limit=limit_edges,
    )
    if nodes is None:
        node_set: set[int] = set()
        for a, b in edges:
            node_set.add(int(a))
            node_set.add(int(b))
        nodes = node_set
    return connected_components(nodes, edges)

