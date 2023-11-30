def bsp_value(L, m):
    # Sort the list of stations in ascending order
    n = len(L)
    L.sort()

    # Define a helper function to check if a given distance is feasible
    def check(distance):
        count = 0
        last_station = float('-inf')

        # Iterate through the stations to count the number of removed stations
        for station in L:
            if station - last_station >= distance:
                last_station = station
            else:
                count += 1

        # Check if the count of removed stations is within the allowed limit
        return count <= m

    # Binary search for the maximum possible distance between adjacent stations
    low, high = 0, L[-1] - L[0]

    while low < high:
        mid = (low + high + 1) // 2
        if check(mid):
            low = mid
        else:
            high = mid - 1

    # Return the maximum possible distance between adjacent stations
    return low


def bsp_solution(L, m):
    # Sort the list of stations in ascending order
    n = len(L)
    L.sort()

    result = []
    last_station = float('-inf')

    # Iterate through the stations to construct the solution based on the computed distance
    for station in L:
        if station - last_station >= bsp_value(L, m):
            result.append(station)
            last_station = station

    # Return the final list of stations satisfying the distance condition
    return result


def test_bsp():
    L = [2, 4, 6, 7, 10, 14]
    m = 2
    print(bsp_value(L, m))  # Output: 4
    print(bsp_solution(L, m))  # Output: [2, 6, 10, 14]


if __name__ == '__main__':
    test_bsp()