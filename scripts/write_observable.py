import json
from pathlib import Path

import networkx as nx
import numpy as np

from ztnd.generations import load_completions
from ztnd.graphs import build_token_graph, build_token_pos_tree

cache_base = Path("cache") / "2024-10-14-18-20-08"
completions_path = cache_base / "completions.json"
completions = load_completions(completions_path)

graph = build_token_pos_tree(completions, add_token_ids=False)
start_node_id = 0
pos = nx.bfs_layout(graph, start_node_id)

# update the positions so that,
# xpos goes from 0 -> num_tokens-1
# ypos have unit distance and are centered on 0
xpos = {}
for k, v in pos.items():
    xpos[k] = float(graph.nodes[k]["token_index"])

max_token_index = max([data["token_index"] for node, data in graph.nodes(data=True)])
min_token_index = min([data["token_index"] for node, data in graph.nodes(data=True)])

ypos = {}
for token_index in range(min_token_index, max_token_index + 1):
    node_data = [
        (node, data) for node, data in graph.nodes(data=True) if data["token_index"] == token_index
    ]
    node_yvals = sorted(
        [(node, float(pos[node][1])) for node, data in node_data], key=lambda x: x[1]
    )
    nyvals = len(node_yvals)

    if nyvals % 2 == 0:
        new_yvals = np.linspace(-(nyvals // 2) + 0.5, (nyvals // 2) - 0.5, nyvals)
    else:
        new_yvals = np.linspace(-(nyvals // 2), nyvals // 2, nyvals)

    for node_yval, new_yval in zip(node_yvals, new_yvals):
        node_id, yval = node_yval
        ypos[node_id] = float(new_yval)

nx.set_node_attributes(graph, xpos, "xpos")
nx.set_node_attributes(graph, ypos, "ypos")

data = nx.node_link_data(graph, edges="edges")
out_path = Path("token_pos_tree_observable.json")
with out_path.open("w") as fp:
    fp.write(json.dumps(data, indent=4))
