import json
from pathlib import Path
import networkx as nx
import numpy as np

cache_path = Path("cache") / "2024-10-03-21-41-11"
with open(cache_path / "node_link_data.json") as fp:
    nld = json.load(fp)
graph = nx.node_link_graph(nld)


# calculate bfs layout using networkx
start_node_id = "Once|0"
pos = nx.bfs_layout(graph, start_node_id)


# update the positions so that,
# xpos goes from 0 -> num_tokens-1
# ypos have unit distance and are centered on 0
xpos = {}
for k, v in pos.items():
    xpos[k] = float(graph.nodes[k]["ipos"])


max_ipos = max([data["ipos"] for node, data in graph.nodes(data=True)])
min_ipos = min([data["ipos"] for node, data in graph.nodes(data=True)])

ypos = {}
for ipos in range(min_ipos, max_ipos + 1):
    node_data = [
        (node, data) for node, data in graph.nodes(data=True) if data["ipos"] == ipos
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

data = nx.node_link_data(graph)
out_path = Path("graph_bfs_node_link_data.json")
with out_path.open("w") as fp:
    fp.write(json.dumps(data, indent=4))
