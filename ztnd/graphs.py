from collections import defaultdict

import networkx as nx
from openai.types.chat.chat_completion import ChatCompletion
import rich


def build_graph_from_completions(completions: list[ChatCompletion]) -> nx.DiGraph:
    """Build a graph from completions."""
    graph = nx.DiGraph()

    # get choice_ids for nodes
    #=======================================================
    choice_ids = defaultdict(list)
    for completion in completions:
        for choice in completion.choices:
            if choice.logprobs is None:
                raise ValueError("choice.logprobs is None")
            if choice.logprobs.content is None:
                raise ValueError("choice.logprobs.content is None")

            completion_id=completion.id
            choice_index=choice.index
            choice_id=f"{completion.id}-{choice.index}"
            for ipos, clt in enumerate(choice.logprobs.content):
                node_id = f"{clt.token}|{ipos}"
                choice_ids[node_id].append(choice_id)

    # create nodes
    #=======================================================
    for completion in completions:
        for choice in completion.choices:
            if choice.logprobs is None:
                raise ValueError("choice.logprobs is None")
            if choice.logprobs.content is None:
                raise ValueError("choice.logprobs.content is None")

            completion_id=completion.id
            choice_index=choice.index
            choice_id=f"{completion.id}-{choice.index}"
            for ipos, clt in enumerate(choice.logprobs.content):
                node_id = f"{clt.token}|{ipos}"
                node_meta = {
                    "text": clt.token,
                    "ipos": ipos,
                    "choice_ids": choice_ids[node_id],
                }
                graph.add_nodes_from([(node_id, node_meta)])

    # create edges
    #=======================================================
    for completion in completions:
        for choice in completion.choices:
            if choice.logprobs is None:
                raise ValueError("choice.logprobs is None")
            if choice.logprobs.content is None:
                raise ValueError("choice.logprobs.content is None")

            for ii in range(len(choice.logprobs.content) - 1):
                clt_lo = choice.logprobs.content[ii]
                clt_hi = choice.logprobs.content[ii+1]

                node_id_lo = f"{clt_lo.token}|{ii}"
                node_id_hi = f"{clt_hi.token}|{ii+1}"

                if graph.has_edge(node_id_lo, node_id_hi):
                    graph[node_id_lo][node_id_hi]["weight"] += 1
                else:
                    graph.add_edge(
                        node_id_lo,
                        node_id_hi,
                        weight=1,
                    )

    return graph

