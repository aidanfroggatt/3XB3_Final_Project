def bsp_value(L, m):
    n = len(L)
    # The following initializes the dynamic programming table to store the maximum possible minimum distance
    dynamic_programmer = [[float('inf')] * (m + 1) for _ in range(n + 1)] 
    L = [0] + L  # To make indexing match the station numbers

    # This initializes the base case for the dynamic programming table
    for i in range(n + 1):
        dynamic_programmer[i][0] = float('inf')

    # This finds the maximum minimum distance
    for i in range(1, n + 1):
        for j in range(1, min(m, i) + 1):
            for k in range(i):
                dynamic_programmer[i][j] = min(dynamic_programmer[i][j], max(dynamic_programmer[k][j - 1], L[i] - L[k + 1]))

    return dynamic_programmer[n][m]

def bsp_solution(L, m):
    n = len(L)
    # Initializing a dynamic programming table to store the maximum possible minimum distance
    dynamic_programmer = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    L = [0] + L  # To make indexing match the station numbers
    chosen = [[-1] * (m + 1) for _ in range(n + 1)]

    # Initializing a base case for the dynamic programming table
    for i in range(n + 1):
        dynamic_programmer[i][0] = float('inf')

    # This presents a dynamic programming approach for computing maximum possible minimum distance and keeping track of which station is chosen
    for i in range(1, n + 1):
        for j in range(1, min(m, i) + 1):
            for k in range(i):
                distance = max(dynamic_programmer[k][j - 1], L[i] - L[k + 1])
                if distance < dynamic_programmer[i][j]:
                    dynamic_programmer[i][j] = distance
                    chosen[i][j] = k

    # The following rebuilds the solution from the chosen stations
    result = []
    idx = n
    for i in range(m, 0, -1):
        result.append(L[chosen[idx][i] + 1])
        idx = chosen[idx][i]

    result.sort()
    return result

# Test cases
L = [2, 4, 6, 7, 10, 14]
m = 2
print("bsp_value([2, 4, 6, 7, 10, 14], 2) =", bsp_value(L, m))
print("bsp_solution([2, 4, 6, 7, 10, 14], 2) =", bsp_solution(L, m))
