import time
import matplotlib.pyplot as plt
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


# Approximation Algorithms

# Dijkstra Approximation Algorithm
def dijkstra_approx(G, source, k):
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
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour] and dist[current_node] < k:
                Q.decrease_key(neighbour, dist[current_node] + G.w(current_node, neighbour))
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node
    return dist


# Bellman-Ford Approximation Algorithm
def bellman_ford_approx(G, source, k):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    nodes = list(G.adj.keys())

    # Initialize distances
    for node in nodes:
        dist[node] = float("inf")
    dist[source] = 0

    # Meat of the algorithm
    for _ in range(min(k, G.number_of_nodes() - 1)):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist


# Experiment 1: Varying Graph Size
def experiment_graph_size():
    sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    execution_times_dijkstra = []
    execution_times_bellman_ford = []

    for size in sizes:
        G = create_random_complete_graph(size, 100)
        source = random.choice(list(G.adj.keys()))

        start_dijkstra = time.time()
        dist_dijkstra = dijkstra_approx(G, source, k=10)
        end_dijkstra = time.time()
        execution_times_dijkstra.append(end_dijkstra - start_dijkstra)

        start_bellman_ford = time.time()
        dist_bellman_ford = bellman_ford_approx(G, source, k=10)
        end_bellman_ford = time.time()
        execution_times_bellman_ford.append(end_bellman_ford - start_bellman_ford)

    plt.plot(sizes, execution_times_dijkstra, label='Dijkstra Approximation')
    plt.plot(sizes, execution_times_bellman_ford, label='Bellman-Ford Approximation')
    plt.xlabel('Graph Size')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.title('Experiment 1: Execution Time vs. Graph Size')
    plt.show()


# Experiment 2: Varying Graph Density
def experiment_graph_density():
    densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    execution_times_dijkstra = []
    execution_times_bellman_ford = []

    for density in densities:
        G = create_random_complete_graph(50, 10)  # Keep size constant
        # Adjust edges to achieve desired density
        num_edges = int(density * G.number_of_nodes() * (G.number_of_nodes() - 1))
        edges = random.sample([(i, j) for i in G.adj.keys() for j in G.adj.keys() if i != j], num_edges)
        G.adj = {node: [] for node in G.adj}
        G.weights = {}

        for edge in edges:
            G.add_edge(edge[0], edge[1], random.randint(1, 10))

        source = random.choice(list(G.adj.keys()))

        start_time_dijkstra = time.time()
        dist_dijkstra = dijkstra_approx(G, source, k=10)
        end_time_dijkstra = time.time()
        execution_times_dijkstra.append(end_time_dijkstra - start_time_dijkstra)

        start_time_belmann_ford = time.time()
        dist_bellman_ford = bellman_ford_approx(G, source, k=10)
        end_time_bellman_ford = time.time()
        execution_times_bellman_ford.append(end_time_bellman_ford - start_time_belmann_ford)

    plt.plot(densities, execution_times_dijkstra, label='Dijkstra Approximation')
    plt.plot(densities, execution_times_bellman_ford, label='Bellman-Ford Approximation')
    plt.xlabel('Graph Density')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.title('Experiment 2: Execution Time vs. Graph Density')
    plt.show()


# Experiment 3: Impact of Relaxation Limit (k)
def experiment_relaxation_limit():
    limits = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    execution_times_dijkstra = []
    execution_times_bellman_ford = []

    for limit in limits:
        G = create_random_complete_graph(50, 10)
        source = random.choice(list(G.adj.keys()))

        start_time_dijkstra = time.time()
        dist_dijkstra = dijkstra_approx(G, source, k=limit)
        end_time_dijkstra = time.time()
        execution_times_dijkstra.append(end_time_dijkstra - start_time_dijkstra)

        start_time_belmann_ford = time.time()
        dist_bellman_ford = bellman_ford_approx(G, source, k=limit)
        end_time_belmann_ford = time.time()
        execution_times_bellman_ford.append(end_time_belmann_ford - start_time_belmann_ford)

    plt.plot(limits, execution_times_dijkstra, label='Dijkstra Approximation')
    plt.plot(limits, execution_times_bellman_ford, label='Bellman-Ford Approximation')
    plt.xlabel('Relaxation Limit (k)')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.title('Experiment 3: Execution Time vs. K Value')
    plt.show()


# Run experiments
# experiment_graph_size()
# experiment_graph_density()
# experiment_relaxation_limit()
