import networkx as nx
import matplotlib.pyplot as plt
from final_project_part1 import DirectedWeightedGraph


def visualize_graph_nodes(graph):
    G = nx.DiGraph()

    # Add nodes to the graph
    for node in graph.adj:
        G.add_node(node)

    # Add edges with weights to the graph
    for node1 in graph.adj:
        for node2 in graph.adj[node1]:
            weight = graph.w(node1, node2)
            G.add_edge(node1, node2, weight=weight)

    # Draw the graph
    pos = nx.spring_layout(G)  # You can change the layout algorithm as needed
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def visualize_graph_with_coordinates(graph, node_coordinates):
    G = nx.DiGraph()

    # Add nodes to the graph with geographical positions
    for node, (lat, lon) in node_coordinates.items():
        G.add_node(node, pos=(lon, lat))  # Note: The order of coordinates is (longitude, latitude)

    # Add edges with weights to the graph
    for node1 in graph.adj:
        for node2 in graph.adj[node1]:
            weight = graph.w(node1, node2)
            G.add_edge(node1, node2, weight=weight)

    # Draw the graph with geographical positions
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create an instance of DirectedWeightedGraph
    graph = DirectedWeightedGraph()

    # Add nodes and edges to the graph (you may need to modify this based on your actual data)
    graph.add_node(1)
    graph.add_node(2)
    graph.add_edge(1, 2, 3.5)

    # Sample node_coordinates (replace with your actual coordinates)
    node_coordinates = {
        1: (51.5074, -0.1278),  # London coordinates
        2: (48.8566, 2.3522),  # Paris coordinates
    }

    # Visualize the graph
    visualize_graph_nodes(graph)
    visualize_graph_with_coordinates(graph, node_coordinates)
