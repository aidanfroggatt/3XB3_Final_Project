import csv
from final_project_part1 import DirectedWeightedGraph
from math import radians, sin, cos, sqrt, atan2


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
    distance = float("inf")

    # Get longitude and latitude of the stations from london_stations.csv
    long_lat1 = station_long_lat(station1)
    long_lat2 = station_long_lat(station2)

    # Calculate the distance between the two stations
    distance = haversine_distance(float(long_lat1[0]), float(long_lat1[1]), float(long_lat2[0]), float(long_lat2[1]))

    return distance


# Helper function for the heuristic function
# Gathers the longitude and latitude of a station
def station_long_lat(station):
    # Get longitude and latitude of station from london_stations.csv
    for row in csv.reader(open("london_stations.csv")):
        if row[0] == station:
            station_latitude = row[1]
            station_longitude = row[2]
    return station_latitude, station_longitude


# Helper function for the heuristic function
# Calculates the distance between two stations
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of the Earth in kilometers (you can change it to miles by using 3958.8)
    R = 6371.0

    # Calculate the distance
    distance = R * c

    return distance


# Test heuristic function
# print(heuristic_function("1", "2"))


# TODO: Run A* and Dijkstra's algorithm on the graph, comparing the runtimes
