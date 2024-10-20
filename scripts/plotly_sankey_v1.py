from pathlib import Path
import json
import numpy as np

import networkx as nx
import plotly.graph_objects as go

from ztnd.generations import load_completions
from ztnd.graphs import (
    build_token_graph,
    build_token_pos_tree,
)


def get_weight_range(graph) -> tuple[float, float]:
    weights = [edge[2].get("weight", 1) for edge in graph.edges(data=True)]
    min_weight = min(weights)
    max_weight = max(weights)
    return min_weight, max_weight


def get_edge_traces(graph, pos):

    min_weight, max_weight = get_weight_range(graph)
    weight_range = max_weight - min_weight

    edge_traces = []

    for n0id, n1id, eprops in graph.edges(data=True):
        x0, y0 = pos[n0id]
        x1, y1 = pos[n1id]
        weight = eprops["weight"]

        if weight_range == 0:
            normalized_weight = 5
        else:
            normalized_weight = 1 + 40 * (weight - min_weight) / weight_range

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=normalized_weight, color="#888"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    return edge_traces

def get_node_trace(graph, pos):

    # Create node trace
    node_x = [pos[node][0] for node in graph.nodes()]
    node_y = [pos[node][1] for node in graph.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            size=10,
            color=[],
        ),
    )

    node_text = []
    for node_id, node_meta in graph.nodes(data=True):
        node_text.append(node_meta["label"])

    node_trace.text = node_text

    return node_trace



cache_base = Path("cache") / "2024-10-14-18-20-08"
completions_path = cache_base / "completions.json"
completions = load_completions(completions_path)

#graph = build_token_graph(completions, graph_type = "token", add_token_ids=False)
#graph = build_token_graph(completions, graph_type = "token_pos", add_token_ids=False)
graph = build_token_pos_tree(completions, add_token_ids=False)

start_node_id = 0
pos = nx.bfs_layout(graph, start_node_id)

label = [meta["label"] for node_id, meta in graph.nodes(data=True)]
source = [s for s,t,e in graph.edges(data=True)]
target = [t for s,t,e in graph.edges(data=True)]
value = [e["weight"] for s,t,e in graph.edges(data=True)]
xpos = np.array([float(pos[node_id][0]) for node_id, meta in graph.nodes(data=True)])
ypos = np.array([float(pos[node_id][1]) for node_id, meta in graph.nodes(data=True)])


xpos = (xpos - xpos.min()) / (xpos.max()-xpos.min())
ypos = (ypos - ypos.min()) / (ypos.max()-ypos.min()) 



fig = go.Figure(data=[go.Sankey(
    arrangement="freeform",
    node = dict(
        pad = 1,
        thickness = 10,
        line = dict(color = "black", width = 0.5),
        label = label,
        x = xpos,
        y = ypos,
        color = "blue"
    ),
    link = dict(
        source = source,
        target = target,
        value = value,
    ))])

fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
fig.show()

