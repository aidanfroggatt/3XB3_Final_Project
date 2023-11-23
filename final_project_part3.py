from final_project_part1 import DirectedWeightedGraph


def print_all_csv_data():
    london_connections = open("Data/london_connections.csv", "r")

    for row in london_connections:
        print(row)

    london_connections.close()

    london_stations = open("Data/london_stations.csv", "r")

    for row in london_stations:
        print(row)

    london_stations.close()


def make_underground_graph():
    london_connections = open("Data/london_connections.csv", "r")
    london_stations = open("Data/london_stations.csv", "r")

    G = DirectedWeightedGraph()

    for row in london_stations:
        if row[0] != "id" and row[0] != "\"":
            G.add_node(row[0])

    for row in london_connections:
        if row[0] != "\"" and row[0] != "," and row[1] != ",":
            weight = 1
            G.add_edge(row[0], row[1], weight)

    london_connections.close()
    london_stations.close()

    return G


G = make_underground_graph()
print(G.adj)