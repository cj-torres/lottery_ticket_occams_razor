import networkx as nx
from collections import Counter
from itertools import product, combinations
import cProfile

def connect_layers(layer1: list[int], layer2: list[int], graph: nx.Graph):
    for node1 in layer1:
        for node2 in layer2:
            graph.add_edge(node1, node2)


def dnn_architecture_to_graph(architecture: list[int]):
    nodes = []
    node_num = 0
    network = nx.Graph()
    for layer in architecture:
        layer_nodes = []
        for _ in range(layer):
            layer_nodes += [node_num]
            network.add_node(node_num)
            node_num += 1
        nodes.append(layer_nodes)

    for i, layer1 in enumerate(nodes[:-1]):
        layer2 = nodes[i+1]
        network.add_edges_from(product(layer1, layer2))

    return network


def generate_end_connected_subgraphs(graph: nx.Graph, last_nodes: list[int]):
    '''
    :param graph: networkx graph
    :param last_nodes: set of nodes in networkx graph of which at least one must be included in the subgraph
    :yields: 'relevant' subrgraphs, subgraphs with components connected to last_nodes
    '''
    assert all([last_node in graph.nodes for last_node in last_nodes]), "last_nodes must be nodes in graph"
    # return the empty graph
    yield graph.subgraph([])
    for edge_num in range(1, len(graph.edges)+1):
        for edge_combo in combinations(graph.edges, edge_num):
            subgraph = graph.edge_subgraph(edge_combo)
            if nx.is_connected(subgraph) and any([last_node in subgraph.nodes for last_node in last_nodes]):
                yield subgraph
            #elif nx.is_connected(subgraph):
            #    yield graph.subgraph([])
            #else:
            #    relevant_subgraph_nodes = set()
            #    for last_node in last_nodes:
            #        relevant_subgraph_nodes.add(nx.node_connected_component(subgraph, last_node))
            #    yield subgraph.subgraph(relevant_subgraph_nodes)


def generate_input_sensitive_end_connected_subgraphs(graph: nx.Graph, input_nodes: list[int], output_nodes: list[int]):
    '''
    :param graph:
    :param input_nodes:
    :param output_nodes:
    :return:
    '''
    input_insensitive_network = graph.subgraph(set(graph.nodes) - set(input_nodes))

nw = dnn_architecture_to_graph([4,4,4,1])





network4x4x1 = nx.Graph()
layer1 = [1,2,3,4]
layer2 = [5,6,7,8]
layer3 = [9,10,11,12]
layer4 = [13]
network4x4x1.add_nodes_from(layer1+layer2+layer3+layer4)
connect_layers(layer1, layer2, network4x4x1)
connect_layers(layer2, layer3, network4x4x1)
connect_layers(layer3, layer4, network4x4x1)




