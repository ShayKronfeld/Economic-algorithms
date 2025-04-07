import numpy as np
import cvxpy

import cvxpy
import numpy as np

def calculate_resource_prices(matrix, allocation, budgets):
    num_players = len(matrix)
    num_resources = len(matrix[0])

    resource_prices = []
    
    # Iterate through resources to calculate their prices
    for resource in range(num_resources):
        resource_price = 0
        for player in range(num_players): 
            if allocation[resource][player] > 1e-6:  # Check if resource has been allocated
                total_utility = sum(allocation[res][player] * matrix[player][res] for res in range(num_resources))
                if total_utility > 0:
                    resource_price = (budgets[player] * matrix[player][resource]) / total_utility
                    break
        resource_prices.append(resource_price)

    return resource_prices

def calculate_equilibrium(matrix, budgets):
    # Perform validation on preference matrix to ensure no negative values
    for row in matrix:
        if any(value < 0 for value in row):
            raise ValueError("the matrix cannot contain negative values.")
    
    if len(matrix) != len(budgets):
        raise ValueError("the number of players must be equal to the number of budgets.")

    num_players = len(matrix)
    num_resources = len(matrix[0])

    # Create decision variables for allocations
    allocation = cvxpy.Variable((num_resources, num_players))

    constraints = []
    
    # Ensure total allocation per resource sums to 1, and allocations are within [0, 1]
    for j in range(num_resources):
        constraints.append(cvxpy.sum(allocation[j, :]) == 1)
        for i in range(num_players):
            constraints.append(allocation[j, i] >= 0)
            constraints.append(allocation[j, i] <= 1)

    # Calculate utilities for each player
    utilities = []
    for i in range(num_players):
        player_utility = sum(allocation[j, i] * matrix[i][j] for j in range(num_resources))
        utilities.append(budgets[i] * cvxpy.log(player_utility))

    # Solve optimization problem to maximize utility
    problem = cvxpy.Problem(cvxpy.Maximize(cvxpy.sum(utilities)), constraints)
    problem.solve()

    return problem.value, allocation.value


def run_example(valuation_matrix, budgets, title=""):
    prob_value, allocation = calculate_equilibrium(valuation_matrix, budgets)
    player_utils = np.sum(allocation.T * valuation_matrix, axis=1)  # שימוש ב- .T להחלפת הממדים של הקצאה
    num_agents, num_goods = valuation_matrix.shape

    print(f"\n--- {title} ---\n")
    
    # Allocation table: players × resources (readable format)
    print("Allocation (Players × Resources):")
    for i in range(num_agents):
        print(f"  Player {i+1}: ", end="")
        for j in range(num_goods):
            print(f"{allocation[j, i]:.2f} ", end=" ")  # Update to correct indexing
        print()

    prices = calculate_resource_prices(valuation_matrix, allocation, budgets)

    print("\nResource prices:")
    for j in range(num_goods):
        print(f"  Resource {j+1}: Price = {prices[j]:.4f}")

    print("\nPlayer utilities:")
    for i in range(num_agents):
        print(f"  Player {i+1}: Utility = {player_utils[i]:.4f}")
    
    print("\n" + "-" * 25 + "\n")


def set_inputs(supply, budgets, n_players, n_items):
    if supply is None:
        supply = np.ones(n_items)
    if budgets is None:
        budgets = np.ones(n_players)
    return supply, budgets

# Run examples
if __name__ == "__main__":
    supply = np.array([1, 1, 1], dtype=float)

    run_example(
        valuation_matrix=np.array([[7, 3, 1], [3, 5, 4]], dtype=float),
        budgets=np.array([55, 45], dtype=float),
        title="Example 1"
    )

    run_example(
        valuation_matrix=np.array([[9, 1, 6], [2, 7, 4]], dtype=float),
        budgets=np.array([45, 55], dtype=float),
        title="Example 2"
    )

    run_example(
        valuation_matrix=np.array([[6, 2, 3], [4, 2, 2]], dtype=float),
        budgets=np.array([2, 2], dtype=float),
        title="Example 3"
    )

    run_example(
        valuation_matrix=np.array([[6, 4, 3], [6, 4, 3], [6, 4, 3]], dtype=float),
        budgets=np.array([15, 15, 70], dtype=float),
        title="Example 4"
    )

    run_example(
        valuation_matrix=np.array([[0, 0, 12], [6, 6, 6]], dtype=float),
        budgets=np.array([35, 65], dtype=float),
        title="Example 5"
    )
