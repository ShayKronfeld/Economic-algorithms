from scipy.optimize import linprog
import numpy as np

def find_decomposition(budget, preferences):
    n = len(preferences)        # Number of players (citizens)
    m = len(budget)             # Number of topics (projects)
    C = sum(budget)             # Total available budget

    # Map each variable (i,j) → index in the linear program
    # Assign a unique index to each variable x[i,j] (where player i supports topic j) so it can be used in the 1D vector representation required by linprog
    var_indices = {}
    idx = 0
    for i in range(n):
        for j in preferences[i]:
            var_indices[(i, j)] = idx
            idx += 1
    num_vars = idx  # Total number of variables

    # Objective function: we only care about feasibility, so it's zero
    c = np.zeros(num_vars)

    # Equality constraints: A_eq x = b_eq
    A_eq = [] # A_eq is the matrix of equality constraint coefficients (each row defines weights for one constraint)
    b_eq = [] # b_eq is the vector of target values for each equality constraint (right-hand side of the equations)

    # Constraint 1: each player must contribute exactly C/n to their supported topics
    for i in range(n):
        row = np.zeros(num_vars)
        for j in preferences[i]:
            row[var_indices[(i, j)]] = 1
        A_eq.append(row)
        b_eq.append(C / n)

    # Constraint 2: each topic must receive exactly its allocated budget
    for j in range(m):
        row = np.zeros(num_vars)
        for i in range(n):
            if j in preferences[i]:
                row[var_indices[(i, j)]] = 1
        A_eq.append(row)
        b_eq.append(budget[j])

    # Creates a list of length num_vars; each entry defines the bounds for a variable (here: all variables must be ≥ 0)
    bounds = [(0, None)] * num_vars

    # Solve the linear program using scipy
    # Use the HiGHS solver: a modern, fast, and numerically stable LP solver
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        x = res.x
        decomposition = [dict() for _ in range(n)]

        # Build the output structure from non-zero variables
        for (i, j), k in var_indices.items():
            if x[k] > 1e-6:  # Treat values below this as zero
                decomposition[i][j] = round(x[k], 2)

        # --- Print formatted decomposition table with totals ---
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

        # Add total row at the bottom
        total_row = "{:<17}".format("Total") + "".join(f"{s:^10.2f}" for s in topic_totals) + f"{sum(topic_totals):^10.2f}"
        print(total_row)

        return decomposition
    else:
        print("❌ No valid decomposition found (budget is not decomposable).")
        return None


# ---------- Example Tests ----------

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
preferences = [ {0}, set(), {1} ]  # Player 1 can't contribute to any topic
find_decomposition(budget, preferences)

print("\n--- Test 4: Topic with No Support ---")
budget = [100, 100, 100]
preferences = [ {0}, {1} ]  # Topic 2 has no supporters
find_decomposition(budget, preferences)

print("\n--- Test 5: Non-Decomposable Budget ---")
budget = [90, 90, 20]
preferences = [ {0,1}, {1,2}, {2} ]  # Cannot divide budget evenly into 100 per player
find_decomposition(budget, preferences)

print("\n--- Test 6: Balanced and Feasible ---")
budget = [100, 200, 100]
preferences = [ {0,1}, {1,2}, {0,2} ]
find_decomposition(budget, preferences)

print("\n--- Test 7: Single Player and Topic ---")
budget = [100]
preferences = [ {0}]
find_decomposition(budget, preferences)
