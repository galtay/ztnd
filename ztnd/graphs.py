import networkx as nx

from ztnd.generations import ZtndChoice


def build_graph_from_ztnd_choices(ztnd_choices: list[ZtndChoice]) -> nx.DiGraph:
    """Build a graph from the generated tokens."""
    G = nx.DiGraph()

    for ztnd_choice in ztnd_choices:
        for ztnd_token in ztnd_choice.tokens:
            node = ztnd_token.get_node()
            G.add_nodes_from(
                [
                    (
                        node,
                        {
                            "label": ztnd_token.text,
                            "ipos": ztnd_token.ipos,
                            "id": f"{ztnd_token.text}|{ztnd_token.ipos}",
                        },
                    )
                ]
            )

    for ztnd_choice in ztnd_choices:
        for ii in range(len(ztnd_choice) - 1):
            node_lo, node_hi = (
                ztnd_choice.tokens[ii].get_node(),
                ztnd_choice.tokens[ii + 1].get_node(),
            )
            if G.has_edge(node_lo, node_hi):
                G[node_lo][node_hi]["weight"] += 1
                G[node_lo][node_hi]["choice_ids"].append(ztnd_choice.choice_id)
            else:
                G.add_edge(
                    node_lo,
                    node_hi,
                    weight=1,
                    choice_ids=[ztnd_choice.choice_id],
                )

    return G
