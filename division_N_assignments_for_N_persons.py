import cvxpy as cp
import numpy as np
import itertools

# Define the utility matrix v[i][j]: disutility (negative) of player i for task j
v = np.array([
    [-700, -400, -300],  # Player 1
    [-5, -6, -4],  # Player 2
    [-6, -1, -3]   # Player 3
])

n = v.shape[0]
budget = 1000

# Try all task assignments (permutations)
assignments = list(itertools.permutations(range(n)))

proof_found = False  # Will be set to True if one feasible solution is found

for assignment in assignments:
    p = cp.Variable(n)
    constraints = []

    # Envy-free constraints: p_j - p_i <= v_i(t_i) - v_i(t_j)
    for i in range(n):
        for j in range(n):
            delta = v[i, assignment[i]] - v[i, assignment[j]]
            constraints.append(p[j] - p[i] <= delta)

    # Budget constraint
    constraints.append(cp.sum(p) == budget)

    # Solve the LP
    prob = cp.Problem(cp.Minimize(0), constraints)
    result = prob.solve()

    if prob.status == "optimal":
        print(f"\n Feasible envy-free solution FOUND for assignment: {assignment}")
        for i in range(n):
            task = assignment[i]
            print(f"Player {i+1} → Task {task+1}, Utility: {v[i, task]}, Payment: {p.value[i]:.2f}")
        print(f"Total payments: {sum(p.value):.2f}")
        proof_found = True
        break  # One solution is enough to prove feasibility

if not proof_found:
    print("No feasible assignment found — contradiction to the assumption!")
else:
    print("\n This proves that the system of constraints is feasible under at least one assignment.")
