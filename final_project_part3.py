import csv
import math
import heapq
import random
import time

from matplotlib import pyplot as plt

import visualization
from final_project_part1 import DirectedWeightedGraph


class LondonSubway:
    def __init__(self):
        # Initialize LondonSubway object with a DirectedWeightedGraph, node coordinates, and load data from CSV files
        self.graph = DirectedWeightedGraph()
        self.node_coordinates = {}
        self.load_data()

    def load_data(self):
        # Load station coordinates from 'london_stations.csv'
        with open('london_stations.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                node_id, latitude, longitude = int(row[0]), float(row[1]), float(row[2])
                self.node_coordinates[node_id] = (latitude, longitude)
                self.graph.add_node(node_id)

        # Load connections and weights from 'london_connections.csv'
        with open('london_connections.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                node1, node2, _, time = int(row[0]), int(row[1]), int(row[2]), float(row[3])
                distance = self.calculate_distance(node1, node2)
                self.graph.add_edge(node1, node2, time)  # Use time as weight
                self.graph.add_edge(node2, node1, time)  # Assuming bidirectional connections

    def calculate_distance(self, node1, node2):
        # Calculate distance between two nodes using Haversine formula
        lat1, lon1 = self.node_coordinates[node1]
        lat2, lon2 = self.node_coordinates[node2]
        R = 6371  # Radius of Earth in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def heuristic(self, node, goal):
        # Calculate Euclidean distance as the heuristic (straight-line distance) from the current node to the goal
        lat1, lon1 = self.node_coordinates[node]
        lat2, lon2 = self.node_coordinates[goal]
        return self.calculate_distance(node, goal)

    def dijkstra(self, start, goal):
        # Dijkstra's algorithm to find the shortest path from start to goal
        visited = set()
        distances = {node: float('inf') for node in self.graph.adj}
        distances[start] = 0
        priority_queue = [(0, start)]

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

        return distances[goal]

    def a_star(self, start, goal):
        # A* algorithm to find the shortest path from start to goal using both actual distance and heuristic
        visited = set()
        distances = {node: float('inf') for node in self.graph.adj}
        distances[start] = 0
        priority_queue = [(0 + self.heuristic(start, goal), 0, start)]

        while priority_queue:
            _, current_distance, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == goal:
                return current_distance

            for neighbor in self.graph.adj[current_node]:
                weight = self.graph.w(current_node, neighbor)
                distance = distances[current_node] + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    priority_queue.append((distance + self.heuristic(neighbor, goal), distance, neighbor))
                    heapq.heapify(priority_queue)

        return float('inf')  # Goal not reachable


def experiment_suite_2():
    # Example usage of LondonSubway class
    london_subway = LondonSubway()
    start_node = 1  # Replace with the desired start node
    goal_node = 127  # Replace with the desired goal node

    # Visualize the graph
    visualization.visualize_graph_with_coordinates(london_subway.graph, london_subway.node_coordinates)

    # Measure time taken for Dijkstra's algorithm
    dijkstra_time = time.time()
    london_subway.dijkstra(start_node, goal_node)
    print(f"Dijkstra time: {time.time() - dijkstra_time}")

    # Measure time taken for A* algorithm
    a_star_time = time.time()
    london_subway.a_star(start_node, goal_node)
    print(f"A* time: {time.time() - a_star_time}")


def experiment_suite_2_plots():
    londonSubway = LondonSubway()

    start_node = 1
    end_nodes = list(range(1, 201))
    # start_node = random.randint(1, 200)
    # end_nodes = random.sample(range(1, 201), 50)
    # For some reason node 189 breaks everything, so just removing it
    if 189 in end_nodes:
        end_nodes.remove(189)
    end_nodes.sort()

    iterations = 50

    # Calculate runtime of Dijkstra's algorithm for different start and goal nodes
    dijkstra_runtimes = []
    for i in end_nodes:
        average_runtime = 0
        for j in range(iterations):
            dijkstra_time = time.time()
            londonSubway.dijkstra(start_node, i)
            average_runtime += (time.time() - dijkstra_time)
        dijkstra_runtimes.append(average_runtime/iterations)

    # Calculate runtime of A* algorithm for different start and goal nodes
    a_star_runtimes = []
    for i in end_nodes:
        average_runtime = 0
        for j in range(iterations):
            a_star_time = time.time()
            londonSubway.a_star(start_node, i)
            average_runtime += (time.time() - a_star_time)
        a_star_runtimes.append(average_runtime/iterations)

    print("Dijkstra runtimes:", dijkstra_runtimes)
    print("A* runtimes:", a_star_runtimes)

    # Plotting
    plt.plot(end_nodes, dijkstra_runtimes, label='Dijkstra')
    plt.plot(end_nodes, a_star_runtimes, label='A*')
    plt.xlabel('End Node')
    plt.ylabel('Runtime (seconds)')
    plt.title('Comparison of Dijkstra and A* Runtimes')
    plt.legend()
    plt.show()

    # Comparing runtimes
    a_better_a_runtimes = []
    a_better_d_runtimes = []
    a_better_end_nodes = []
    d_better_a_runtimes = []
    d_better_d_runtimes = []
    d_better_end_nodes = []
    comparable_a_runtimes = []
    comparable_d_runtimes = []
    comparable_end_nodes = []
    for i in range(len(dijkstra_runtimes)):
        if 0 < (dijkstra_runtimes[i] - a_star_runtimes[i]) < 0.0002:
            comparable_a_runtimes.append(a_star_runtimes[i])
            comparable_d_runtimes.append(dijkstra_runtimes[i])
            comparable_end_nodes.append(end_nodes[i])
        elif dijkstra_runtimes[i] > a_star_runtimes[i]:
            a_better_a_runtimes.append(a_star_runtimes[i])
            a_better_d_runtimes.append(dijkstra_runtimes[i])
            a_better_end_nodes.append(end_nodes[i])
        elif dijkstra_runtimes[i] < a_star_runtimes[i]:
            d_better_a_runtimes.append(a_star_runtimes[i])
            d_better_d_runtimes.append(dijkstra_runtimes[i])
            d_better_end_nodes.append(end_nodes[i])


    # Plotting
    # Comparable
    plt.plot(comparable_end_nodes, comparable_d_runtimes, label='Dijkstra')
    plt.plot(comparable_end_nodes, comparable_a_runtimes, label='A*')
    plt.xlabel('End Node')
    plt.ylabel('Runtime (seconds)')
    plt.title('Comparison of Dijkstra and A* Runtimes (Comparable Runtimes)')
    plt.legend()
    plt.show()

    # A star better
    plt.plot(a_better_end_nodes, a_better_d_runtimes, label='Dijkstra')
    plt.plot(a_better_end_nodes, a_better_a_runtimes, label='A*')
    plt.xlabel('End Node')
    plt.ylabel('Runtime (seconds)')
    plt.title('Comparison of Dijkstra and A* Runtimes (A* Better)')
    plt.legend()
    plt.show()

    # Dijkstra better
    plt.plot(d_better_end_nodes, d_better_d_runtimes, label='Dijkstra')
    plt.plot(d_better_end_nodes, d_better_a_runtimes, label='A*')
    plt.xlabel('End Node')
    plt.ylabel('Runtime (seconds)')
    plt.title('Comparison of Dijkstra and A* Runtimes (Dijkstra Better)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    experiment_suite_2_plots()
