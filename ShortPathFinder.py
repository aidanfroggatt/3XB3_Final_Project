import heapq
import math
from abc import ABC, abstractmethod


#  Create a class ShortPathFinder that implements the following methods:
#     - calc_short_path(source: int, dest: int) -> float
#     - set_graph(graph: Graph)
#     - set_algorithm(algorithm: SPAlgorithm)

class ShortPathFinder:
    def __init__(self):
        self.graph = None
        self.algorithm = None

    def set_graph(self, graph):
        self.graph = graph

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def calc_short_path(self, source, dest):
        return self.algorithm.calc_sp(self.graph, source, dest)


# Create an abstract class SPAlgorithm that implements the following methods:
#     - calc_sp(graph: Graph, source: int, dest: int) -> float
class SPAlgorithm(ABC):
    @abstractmethod
    def calc_sp(self, graph, source, dest):
        pass


# Create an implementation of SPAlgorithm for each of the following algorithms:
#    - Dijkstra's algorithm
#    - Bellman-Ford algorithm
#    - A* algorithm

class Dijkstra(SPAlgorithm):
    def __init__(self):
        super().__init__()

    def calc_sp(self, graph, source, dest):
        # Dijkstra's algorithm to find the shortest path from start to goal
        visited = set()
        distances = {node: float('inf') for node in graph.adj}
        distances[source] = 0
        priority_queue = [(0, source)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor in graph.adj[current_node]:
                weight = graph.w(current_node, neighbor)
                distance = distances[current_node] + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances[dest]


class BellmanFord(SPAlgorithm):
    def __init__(self):
        super().__init__()

    def calc_sp(self, graph, source, dest):
        # Bellman-Ford algorithm to find the shortest path from start to goal
        distances = {node: float('inf') for node in graph.adj}
        distances[source] = 0

        for i in range(graph.get_num_nodes() - 1):
            for node in graph.adj:
                for neighbor in graph.adj[node]:
                    weight = graph.w(node, neighbor)
                    distance = distances[node] + weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance

        return distances[dest]


class AStar(SPAlgorithm):
    def __init__(self):
        super().__init__()

    def calc_sp(self, graph, source, dest):
        # A* algorithm to find the shortest path from start to goal using both actual distance and heuristic
        visited = set()
        distances = {node: float('inf') for node in graph.adj}
        distances[source] = 0
        priority_queue = [(0 + graph.get_heuristic(source, dest), 0, source)]

        while priority_queue:
            _, current_distance, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == dest:
                return current_distance

            for neighbor in graph.adj[current_node]:
                weight = graph.w(current_node, neighbor)
                distance = distances[current_node] + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    priority_queue.append((distance + graph.get_heuristic(neighbor, dest), distance, neighbor))
                    heapq.heapify(priority_queue)

        return float('inf')  # Goal not reachable


# Create an abstract class Graph that implements the following methods:
#   - get_adj_nodes(node: int) -> List[int]
#   - add_node(node: int)
#   - add_edge(start: int, end: int, w: float)
#   - get_num_nodes() -> int
#   - w(node: int): float

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


# Create an implementation of Graph for each of the following graphs:
#   - WeightedGraph
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


# Create an extension of WeightedGraph for each of the following graphs:
#   - Heuristic
class Heuristic(WeightedGraph):
    # The heuristic graph is a weighted graph with an additional heuristic function
    def __init__(self):
        super().__init__()
        self.heuristic = None

    # Create a function heuristic() -> Dict[int, float] that sets a dictionary of heuristic values for each node
    # Heuristic values are calculated using the Haversine formula and a goal node
    def heuristic(self, goal):
        self.heuristic = {}

        # create a nested function to calculate the distance between two nodes based on longitude and latitude
        def calculate_distance(node1, node2):
            # Calculate distance between two nodes using Haversine formula
            lat1, lon1 = self.node_coordinates[node1]
            lat2, lon2 = self.node_coordinates[node2]
            R = 6371
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
                dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c
            return distance

        # calculate the distance between each node and the goal node
        for node in self.adj:
            self.heuristic[node] = calculate_distance(node, goal)
        return self.heuristic

    # Create a function get_heuristic() -> Dict[int, float] that returns the dictionary of heuristic values
    def get_heuristic(self):
        return self.heuristic
