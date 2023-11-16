import min_heap2 as min_heap
from final_project_part1 import DirectedWeightedGraph


# A* Implementation
def a_star(G, s, d, h):
    pred = {}  # Predecessor dictionary
    dist = {}  # Distance dictionary
    Q = min_heap.MinHeap([])
    nodes = list(G.adj.keys())

    # Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(s, 0)

    # Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key

        if current_node == d:
            break  # Stop the algorithm once the destination is reached

        for neighbour in G.adj[current_node]:
            # Include the heuristic function in the distance calculation
            new_distance = dist[current_node] + G.w(current_node, neighbour) + h[neighbour]
            if new_distance < dist[neighbour]:
                Q.decrease_key(neighbour, new_distance)
                dist[neighbour] = new_distance
                pred[neighbour] = current_node

    # Reconstruct the path
    path = []
    current = d
    while current is not None:
        path.insert(0, current)
        current = pred.get(current, None)

    return pred, path
