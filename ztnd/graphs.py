from collections import defaultdict
from collections import Counter
from typing import Iterable

import networkx as nx
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionTokenLogprob
from pydantic import BaseModel
import rich


class ZtndChoice(BaseModel):
    completion_id: str
    choice_index: int
    choice: Choice

    def get_id(self):
        return f"{self.completion_id}-{self.choice_index}"


def iter_choice(completion: ChatCompletion) -> Iterable[ZtndChoice]:
    for choice in completion.choices:
        yield ZtndChoice(
            completion_id = completion.id,
            choice_index = choice.index,
            choice = choice,
        )


class ZtndToken(BaseModel):
    completion_id: str
    choice_index: int
    token_index: int
    cctl: ChatCompletionTokenLogprob

    def get_id(self):
        return f"{self.completion_id}-{self.choice_index}-{self.token_index}"


def iter_token(zchoice: ZtndChoice) -> Iterable[ZtndToken]:
    if zchoice.choice.logprobs is None:
        raise ValueError("choice.logprobs is None")
    if zchoice.choice.logprobs.content is None:
        raise ValueError("choice.logprobs.content is None")
    for ii, cctl in enumerate(zchoice.choice.logprobs.content):
        yield ZtndToken(
            completion_id = zchoice.completion_id,
            choice_index = zchoice.choice_index,
            token_index = ii,
            cctl = cctl,
        )


def add_visual_space(text: str) -> str:
#    return text.replace(" ", "\u2420") # SP symbol for space
    return text.replace(" ", "\u2423") # open box


def build_token_graph(
    completions: list[ChatCompletion],
    graph_type: str,
    add_token_ids: bool = False,
) -> nx.DiGraph:
    """
    """

    def get_node_id_token(zt: ZtndToken):
        return add_visual_space(zt.cctl.token)

    def get_node_id_token_pos(zt: ZtndToken):
        return "{}|{}".format(add_visual_space(zt.cctl.token), zt.token_index)

    if graph_type == "token":
        get_node_id = get_node_id_token
    elif graph_type == "token_pos":
        get_node_id = get_node_id_token_pos
    else:
        raise ValueError()


    graph = nx.DiGraph()

    # create root node
    #---------------------------------------------------
    root_node_id = "ROOT"
    root_node_meta = {
        "id": root_node_id,
        "label": root_node_id,
    }
    graph.add_nodes_from([(root_node_id, root_node_meta)])

    # get all token_ids at a node
    #---------------------------------------------------
    node_id_to_token_ids = defaultdict(list)
    for completion in completions:
        for zc in iter_choice(completion):
            for zt in iter_token(zc):
                node_id = get_node_id(zt)
                zt_id = zt.get_id()
                node_id_to_token_ids[node_id].append(zt_id)


    # create token nodes
    #---------------------------------------------------
    for completion in completions:
        for zc in iter_choice(completion):
            for zt in iter_token(zc):
                node_id = get_node_id(zt)
                label = node_id
                node_meta = {
                    "id": node_id,
                    "label": label,
                }
                if add_token_ids:
                    node_meta["token_ids"] = node_id_to_token_ids[node_id],
                graph.add_nodes_from([(node_id, node_meta)])


    # create edges
    #---------------------------------------------------
    for completion in completions:
        for zc in iter_choice(completion):
            zts = list(iter_token(zc))

            node_id_lo = root_node_id
            node_id_hi = get_node_id(zts[0])
            if graph.has_edge(node_id_lo, node_id_hi):
                graph[node_id_lo][node_id_hi]["weight"] += 1
            else:
                graph.add_edge(
                    node_id_lo,
                    node_id_hi,
                    weight=1,
                )

            for zt_lo, zt_hi in zip(zts[:-1], zts[1:]):
                node_id_lo = get_node_id(zt_lo)
                node_id_hi = get_node_id(zt_hi)

                if graph.has_edge(node_id_lo, node_id_hi):
                    graph[node_id_lo][node_id_hi]["weight"] += 1
                else:
                    graph.add_edge(
                        node_id_lo,
                        node_id_hi,
                        weight=1,
                    )

    graph = nx.convert_node_labels_to_integers(graph)

    return graph


def build_token_pos_tree(
    completions: list[ChatCompletion],
    add_token_ids: bool = False,
) -> nx.DiGraph:
    """
    """

    def get_token(zt: ZtndToken) -> str:
        return add_visual_space(zt.cctl.token)

    def get_token_pos(zt: ZtndToken) -> str:
        return "{}|{}".format(add_visual_space(zt.cctl.token), zt.token_index)

    graph = nx.DiGraph()

    # create root node
    #---------------------------------------------------
    root_node_id = "ROOT|-1|0"
    root_node_meta = {"id": root_node_id, "label": root_node_id}
    graph.add_nodes_from([(root_node_id, root_node_meta)])

    # create tree
    #---------------------------------------------------

    # as soon as the next token is not in the tree, diverged = True

    prefix_counter = Counter()
    prefix_jj_counter = Counter()
    for icomp, completion in enumerate(completions):
        for zc in iter_choice(completion):

            diverged = False
            zts = list(iter_token(zc))

            # is there an edge from root to first token?

            node_id_prefix_lo = get_token_pos(zts[0])
            node_id_lo = "{}|0".format(node_id_prefix_lo)
            if graph.has_edge(root_node_id, node_id_lo):
                graph[root_node_id][node_id_lo]["weight"] += 1
            else:
                diverged = True
                graph.add_edge(
                    root_node_id,
                    node_id_lo,
                    weight=1,
                )

            for ii, (zt_lo, zt_hi) in enumerate(zip(zts[:-1], zts[1:])):

                node_id_prefix_hi = get_token_pos(zt_hi)
                prefix_counter[node_id_prefix_hi] += 1
                jj2 = prefix_jj_counter[node_id_prefix_hi]

                if diverged:

                    prefix_jj_counter[node_id_prefix_hi] += 1
                    jj2 = prefix_jj_counter[node_id_prefix_hi]
                    node_id_hi = f"{node_id_prefix_hi}|{jj2}"
                    graph.add_edge(
                        node_id_lo,
                        node_id_hi,
                        weight=1,
                    )

                else:

                    nbrs = graph[node_id_lo]
                    mtch_nbrs = [nbr for nbr in nbrs if nbr.startswith(node_id_prefix_hi)]
                    if len(mtch_nbrs) == 0:
                        diverged = True
                        prefix_jj_counter[node_id_prefix_hi] += 1
                        jj2 = prefix_jj_counter[node_id_prefix_hi]
                        node_id_hi = f"{node_id_prefix_hi}|{jj2}"
                        graph.add_edge(
                            node_id_lo,
                            node_id_hi,
                            weight=1,
                        )
                        prefix_jj_counter[node_id_prefix_hi] += 1
                    elif len(mtch_nbrs) == 1:
                        node_id_hi = mtch_nbrs[0]
                        graph[node_id_lo][node_id_hi]["weight"] += 1
                    else:
                        raise ValueError()

                node_id_lo = node_id_hi

    graph = nx.convert_node_labels_to_integers(graph, label_attribute="id")

    label = {}
    for node_id, node_meta in graph.nodes(data=True):
        label[node_id] = node_meta["id"].split("|")[0]
    nx.set_node_attributes(graph, label, "label")

    token_index = {}
    for node_id, node_meta in graph.nodes(data=True):
        token_index[node_id] = int(node_meta["id"].split("|")[1])
    nx.set_node_attributes(graph, token_index, "token_index")


    return graph



if __name__ == "__main__":

    from generations import load_completions
    cpath = "../scripts/cache/2024-10-14-18-20-08/completions.json"
    completions = load_completions(cpath)
    for completion in completions:
        for zc in iter_choice(completion):
            zts = list(iter_token(zc))

    token_graph = build_token_graph(completions, add_token_ids=False)
    token_pos_graph = build_token_pos_graph(completions, add_token_ids=False)
