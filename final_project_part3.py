import csv
from final_project_part1 import DirectedWeightedGraph


# TODO: Convert connections and stations csv files into a graph
# create a function to convert the csv files into a graph
def create_london_graph():
    # create an empty graph
    G = DirectedWeightedGraph()

    # create list of values to skip adding to the graph
    skip = ["station1", "station2", "weight", "id"]

    # open london_stations.csv and iterate over all rows
    with open("london_stations.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] not in skip:
                # add the station to the graph
                G.add_node(row[0])
    # open london_connections.csv and iterate over all rows
    with open("london_connections.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] not in skip and row[1] not in skip:
                # temp weight
                weight = 1
                # add the connection to the graph
                G.add_edge(row[0], row[1], weight)

    # return the graph
    return G


# Test graph creation
# print(create_london_graph().adj)


# TODO: Create a heuristic function for A* algorithm
# create a heuristic function for A* algorithm
def heuristic_function(station1, station2):
    # temp distance
    distance = 1
    # Get longitude and latitude of the stations from london_stations.csv
    long_lat1 = station_long_lat(station1)
    long_lat2 = station_long_lat(station2)
    # Calculate the distance between the two stations

    return distance


# Helper function for the heuristic function to get the longitude and latitude of a station
def station_long_lat(station):
    # Get longitude and latitude of station from london_stations.csv
    for row in csv.reader(open("london_stations.csv")):
        if row[0] == station:
            station_latitude = row[1]
            station_longitude = row[2]
    return station_latitude, station_longitude


# Test heuristic function
print(heuristic_function("1", "2"))

# TODO: Run A* and Dijkstra's algorithm on the graph, comparing the runtimes
