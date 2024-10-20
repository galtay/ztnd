from pathlib import Path

import networkx as nx

from ztnd.generations import load_completions
from ztnd.graphs import (
    build_token_graph,
    build_token_pos_graph,
)

cache_base = Path("cache") / "2024-10-14-18-20-08"
completions_path = cache_base / "completions.json"
completions = load_completions(completions_path)

for graph_type in ["token", "token_pos"]:
    if graph_type == "token":
        graph = build_token_graph(completions, add_token_ids=False)
    elif graph_type == "token_pos":
        graph = build_token_pos_graph(completions, add_token_ids=False)
    else:
        raise ValueError()

    nx.write_gexf(graph, cache_base / graph_type / "graph.gexf")

