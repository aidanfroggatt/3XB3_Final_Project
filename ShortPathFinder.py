import heapq
from abc import ABC, abstractmethod


class SPAlgorithm(ABC):
    @abstractmethod
    def calc_sp(self, graph, source, dest):
        pass


class dijkstra(SPAlgorithm):
    def calc_sp(self, graph, source, dest):
        # Dijkstra's algorithm to find the shortest path from source to dest
        visited = set()
        distances = {node: float('inf') for node in self.graph.adj}
        distances[source] = 0
        priority_queue = [(0, source)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor in self.graph.adj[current_node]:
                weight = self.graph.w(current_node, neighbor)
                distance = distances[current_node] + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances[dest]


class BellmanFord(SPAlgorithm):
    def calc_sp(self, graph, source, dest):
        # Bellman-Ford algorithm to find the shortest path from source to dest
        pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
        dist = {}  # Distance dictionary
        nodes = list(graph.adj.keys())

        # Initialize distances
        for node in nodes:
            dist[node] = float("inf")
        dist[source] = 0

        # Meat of the algorithm
        for _ in range(graph.number_of_nodes()):
            for node in nodes:
                for neighbour in graph.adj[node]:
                    if dist[neighbour] > dist[node] + graph.w(node, neighbour):
                        dist[neighbour] = dist[node] + graph.w(node, neighbour)
                        pred[neighbour] = node
        return dist[dest]


class AStar(SPAlgorithm):
    def calc_sp(self, graph, source, dest):
        # A* algorithm to find the shortest path from source to dest using both actual distance and heuristic
        visited = set()
        distances = {node: float('inf') for node in self.graph.adj}
        distances[source] = 0
        priority_queue = [(0 + self.heuristic(source, dest), 0, source)]

        while priority_queue:
            _, current_distance, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == dest:
                return current_distance

            for neighbor in self.graph.adj[current_node]:
                weight = self.graph.w(current_node, neighbor)
                distance = distances[current_node] + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    priority_queue.append((distance + self.heuristic(neighbor, dest), distance, neighbor))
                    heapq.heapify(priority_queue)

        return float('inf')  # Goal not reachable


class Graph(ABC):
    @abstractmethod
    def get_adj_nodes(self, node):
        pass

    @abstractmethod
    def add_node(self, node):
        pass

    @abstractmethod
    def add_edge(self, start, end, w):
        pass

    @abstractmethod
    def get_num_nodes(self):
        pass

    @abstractmethod
    def w(self, node):
        pass


class WeightedGraph(Graph):
    def __init__(self):
        self.adj = {}
        self.num_nodes = 0

    def get_adj_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []
        self.num_nodes += 1

    def add_edge(self, start, end, w):
        self.adj[start].append((end, w))

    def get_num_nodes(self):
        return self.num_nodes

    def w(self, node1, node2):
        for node in self.adj[node1]:
            if node[0] == node2:
                return node[1]
        return float('inf')


class Heuristic(WeightedGraph):
    def __init__(self):
        super().__init__()

    def add_node(self, node):
        self.adj[node] = []
        self.num_nodes += 1

    def add_edge(self, start, end, w):
        self.adj[start].append((end, w))

    def get_num_nodes(self):
        return self.num_nodes

    def w(self, node1, node2):
        for node in self.adj[node1]:
            if node[0] == node2:
                return node[1]
        return float('inf')

    def heuristic(self):
        pass

    def get_heuristic(self):
        return self.heuristic