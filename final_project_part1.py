import min_heap2 as min_heap
import random


class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)


def dijkstra(G, source):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    # Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(source, 0)

    # Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist


def bellman_ford(G, source):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    nodes = list(G.adj.keys())

    # Initialize distances
    for node in nodes:
        dist[node] = float("inf")
    dist[source] = 0

    # Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist


def total_dist(dist):
    total = 0
    for key in dist.keys():
        total += dist[key]
    return total


def create_random_complete_graph(n, upper):
    G = DirectedWeightedGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, random.randint(1, upper))
    return G


# Assumes G represents its nodes as integers 0,1,...,(n-1)
def mystery(G):
    n = G.number_of_nodes()
    d = init_d(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d


# Dijkstra's Approximation
def dijkstra_approx(G, source, k):
    pred = {}  # Predecessor dictionary
    dist = {}  # Distance dictionary
    Q = min_heap.MinHeap([])  # Priority queue using a min heap
    nodes = list(G.adj.keys())

    # Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))  # Insert each node with distance set to infinity
        dist[node] = float("inf")  # Set initial distances to infinity
    Q.decrease_key(source, 0)  # Set the distance of the source node to 0 in the priority queue

    while not Q.is_empty():
        current_element = Q.extract_min()  # Extract the node with the minimum distance from the priority queue
        current_node = current_element.value
        dist[current_node] = current_element.key  # Update the distance of the current node
        for neighbour in G.adj[current_node]:
            # Relaxation step: Update distance if a shorter path is found
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node  # Update predecessor for the neighbor
    return dist


# Bellman-Ford Approximation
def bellman_ford_approx(G, source, k):
    pred = {}  # Predecessor dictionary
    dist = {}  # Distance dictionary
    nodes = list(G.adj.keys())

    # Initialize distances
    for node in nodes:
        dist[node] = float("inf")  # Set initial distances to infinity
    dist[source] = 0  # Set the distance of the source node to 0

    for _ in range(k):
        for node in nodes:
            for neighbour in G.adj[node]:
                # Relaxation step: Update distance if a shorter path is found
                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node  # Update predecessor for the neighbor
    return dist
