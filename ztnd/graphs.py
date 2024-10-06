from collections import defaultdict
import networkx as nx

from ztnd.generations import ZtndChoice


def build_graph_from_ztnd_choices(ztnd_choices: list[ZtndChoice]) -> nx.DiGraph:
    """Build a graph from the generated tokens."""
    graph = nx.DiGraph()

    # get choice_ids for nodes
    choice_ids = defaultdict(list)
    for ztnd_choice in ztnd_choices:
        for ztnd_token in ztnd_choice.tokens:
            node_id = ztnd_token.get_node_id()
            choice_ids[node_id].append(ztnd_choice.choice_id)

    # create nodes
    for ztnd_choice in ztnd_choices:
        for ztnd_token in ztnd_choice.tokens:
            node_id = ztnd_token.get_node_id()
            graph.add_nodes_from(
                [
                    (
                        node_id,
                        {
                            "label": ztnd_token.text,
                            "ipos": ztnd_token.ipos,
                            "choice_ids": choice_ids[node_id],
                        },
                    )
                ]
            )

    for ztnd_choice in ztnd_choices:
        for ii in range(len(ztnd_choice) - 1):
            node_id_lo, node_id_hi = (
                ztnd_choice.tokens[ii].get_node_id(),
                ztnd_choice.tokens[ii + 1].get_node_id(),
            )
            if graph.has_edge(node_id_lo, node_id_hi):
                graph[node_id_lo][node_id_hi]["weight"] += 1
            else:
                graph.add_edge(
                    node_id_lo,
                    node_id_hi,
                    weight=1,
                )

    return graph
