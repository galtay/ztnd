from pathlib import Path
import json
import numpy as np

import networkx as nx
import plotly.graph_objects as go


def read_graph(nld_path: Path) -> nx.Graph:
    with open(nld_path) as fp:
        nld = json.load(fp)
    graph = nx.node_link_graph(nld, edges="edges")
    return graph

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
            normalized_weight = 1 + 9 * (weight - min_weight) / weight_range

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
    for node in graph.nodes():
        node_text.append(f"{node}")

    node_trace.text = node_text

    return node_trace


#nld_path = Path("cache/2024-10-14-15-30-20/token/node_link_data.json")
nld_path = Path("cache/2024-10-14-15-30-20/token_pos/node_link_data.json")
graph = read_graph(nld_path)


if "token_pos" in str(nld_path):
    start_node_id = "Once|0"
else:
    start_node_id = "Once"

pos = nx.bfs_layout(graph, start_node_id)
#pos = nx.spring_layout(graph)
edge_traces = get_edge_traces(graph, pos)
node_trace = get_node_trace(graph, pos)
fig = go.Figure(
    data=edge_traces + [node_trace],
    layout=go.Layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ),
)

fig.show()