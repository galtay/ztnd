from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import rich
import typer

from ztnd.generations import (
    create_completions,
    save_completions,
    load_completions,
)
from ztnd.graphs import (
    build_token_graph,
    build_token_pos_graph,
)

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


class LogLevel(str, Enum):
    debug = "debug"
    info = "info"
    warn = "warn"
    error = "error"
    critical = "critical"

class GraphType(str, Enum):
    token = "token"
    token_pos = "token_pos"


DEFAULT_PROMPT = """Write a short story starting with "Once upon a time"."""


@app.command()
def generate_completions(
    prompt: str = DEFAULT_PROMPT,
    model: str = "gpt-4o-mini",
    logprobs: bool = True,
    top_logprobs: int = 0,
    max_completion_tokens: int = 64,
    n_choices_per_call: int = 20,
    seed: int = 9237,
    temperature: float = 0.4,
    n_api_calls: int = 10,
    log_level: LogLevel = LogLevel.info,
):

    logging.basicConfig(level=getattr(logging, log_level.upper()))
    rich.print(f"{prompt=}")

    messages = [{"role": "user", "content": prompt}]
    completions = create_completions(
        messages,
        model=model,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        max_completion_tokens=max_completion_tokens,
        n_choices_per_call=n_choices_per_call,
        seed=seed,
        temperature=temperature,
        n_api_calls=n_api_calls,
    )

    cache_path = Path("cache") / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cache_path.mkdir(exist_ok=True, parents=True)
    save_completions(completions, cache_path / "completions.json")


@app.command()
def generate_graph(
    cache_path: Path,
    graph_type: GraphType,
    log_level: LogLevel = LogLevel.info,
):

    completions = load_completions(cache_path / "completions.json")
    rich.print(completions[0].choices[0].message)
    if graph_type == "token":
        graph = build_token_graph(completions)
    elif graph_type == "token_pos":
        graph = build_token_pos_graph(completions)
    else:
        raise ValueError()

    data = nx.node_link_data(graph, edges="edges")
    out_path = cache_path / graph_type
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "node_link_data.json").open("w") as fp:
        fp.write(json.dumps(data, indent=4))



@app.command()
def make_bfs_layout(
    nld_path: Path,
    start_node_id: str,
    log_level: LogLevel = LogLevel.info,
):

    with open(nld_path) as fp:
        nld = json.load(fp)
    graph = nx.node_link_graph(nld, edges="edges")
    pos = nx.bfs_layout(graph, start_node_id)

    xpos = {}
    ypos = {}
    for k, v in pos.items():
        xpos[k] = float(v[0])
        ypos[k] = float(v[1])

    nx.set_node_attributes(graph, xpos, "xpos")
    nx.set_node_attributes(graph, ypos, "ypos")

    data = nx.node_link_data(graph, edges="edges")
    out_path = nld_path.parent / "bfs_node_link_data.json"
    with out_path.open("w") as fp:
        fp.write(json.dumps(data, indent=4))



@app.command()
def make_bfs_layout_v2(
    nld_path: Path,
    start_node_id: str,
    log_level: LogLevel = LogLevel.info,
):

    with open(nld_path) as fp:
        nld = json.load(fp)
    graph = nx.node_link_graph(nld, edges="edges")
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
    out_path = nld_path.parent / "bfs_node_link_data.json"
    with out_path.open("w") as fp:
        fp.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    app()
