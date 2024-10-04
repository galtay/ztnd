import json
from pathlib import Path

import networkx as nx

from ztnd.generations import load_ztnd_choices
from ztnd.graphs import build_graph_from_ztnd_choices

cache_path = Path("cache") / "2024-10-03-21-41-11"
ztnd_choices = load_ztnd_choices(cache_path / "ztnd_choices.json")
graph = build_graph_from_ztnd_choices(ztnd_choices)
data = nx.node_link_data(graph)
out_path = cache_path / "node_link_data.json"
with out_path.open("w") as fp:
    fp.write(json.dumps(data, indent=4))
