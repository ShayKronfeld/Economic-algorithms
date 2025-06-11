from scipy.optimize import linprog
import numpy as np

def find_decomposition(budget, preferences):
    n = len(preferences)
    m = len(budget)
    C = sum(budget)

    var_indices = {}
    idx = 0
    for i in range(n):
        for j in preferences[i]:
            var_indices[(i, j)] = idx
            idx += 1
    num_vars = idx

    c = np.zeros(num_vars)
    A_eq = []
    b_eq = []

    for i in range(n):
        row = np.zeros(num_vars)
        for j in preferences[i]:
            row[var_indices[(i, j)]] = 1
        A_eq.append(row)
        b_eq.append(C / n)

    for j in range(m):
        row = np.zeros(num_vars)
        for i in range(n):
            if j in preferences[i]:
                row[var_indices[(i, j)]] = 1
        A_eq.append(row)
        b_eq.append(budget[j])

    bounds = [(0, None)] * num_vars
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        x = res.x
        decomposition = [dict() for _ in range(n)]

        for (i, j), k in var_indices.items():
            if x[k] > 1e-6:
                decomposition[i][j] = round(x[k], 2)

        # --- Print formatted table with totals ---
        print("\nBudget Decomposition:")
        header = ["Player \\ Topic"] + [f"{j}" for j in range(m)] + ["Total"]
        print("{:<17}".format(header[0]) + "".join(f"{col:^10}" for col in header[1:]))

        topic_totals = [0.0] * m
        for i in range(n):
            row_str = f"{i:<17}"
            player_total = 0.0
            for j in range(m):
                val = decomposition[i].get(j, 0.0)
                topic_totals[j] += val
                player_total += val
                row_str += f"{val:^10.2f}"
            row_str += f"{player_total:^10.2f}"
            print(row_str)

        # Add total row
        total_row = "{:<17}".format("Total") + "".join(f"{s:^10.2f}" for s in topic_totals) + f"{sum(topic_totals):^10.2f}"
        print(total_row)

        return decomposition
    else:
        print("❌ No valid decomposition found (budget is not decomposable).")
        return None



print("\n--- Test 1: Original Example ---")
budget = [400, 50, 50, 0]
preferences = [ {0,1}, {0,2}, {0,3}, {1,2}, {0} ]
find_decomposition(budget, preferences)

print("\n--- Test 2: Zero Budget ---")
budget = [0, 0, 0]
preferences = [ {0}, {1}, {2} ]
find_decomposition(budget, preferences)

print("\n--- Test 3: Player with No Preferences ---")
budget = [100, 100]
preferences = [ {0}, set(), {1} ]  # Player 1 can't donate anywhere
find_decomposition(budget, preferences)

print("\n--- Test 4: Topic with No Support ---")
budget = [100, 100, 100]
preferences = [ {0}, {1} ]  # Topic 2 has no supporters
find_decomposition(budget, preferences)

print("\n--- Test 5: Non-Decomposable Budget ---")
budget = [90, 90, 20]
preferences = [ {0,1}, {1,2}, {2} ]  # No clean division into 100 per player
find_decomposition(budget, preferences)

print("\n--- Test 6: Balanced and Feasible ---")
budget = [100, 200, 100]
preferences = [ {0,1}, {1,2}, {0,2} ]
find_decomposition(budget, preferences)

print("\n--- Test 7: Single Player and Topic ---")
budget = [100]
preferences = [ {0}]
find_decomposition(budget, preferences)