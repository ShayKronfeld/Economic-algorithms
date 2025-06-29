import cvxpy 
import numpy 

# shay kronfeld- 322234782

def Egalitarian_division(matrix):
    values = numpy.array(matrix) # Convert input list to a NumPy array 
    num_peoples, num_resources = values.shape # Get the number of people (agents) and number of resources
    
    # Create a CVXPY variable for the allocation matrix
    x = cvxpy.Variable((num_peoples, num_resources))
    
    # Calculate the utilities for each person based on the allocation
    # cvxpy.multiply(x, values): multiplication of allocation matrix and valuation matrix
    # axis=1: sums across each row (i.e., for each person), returning a vector of total utilities
    utilities = cvxpy.sum(cvxpy.multiply(x, values), axis=1)

    # Create a variable to represent the minimum utility to be maximized
    min_utility = cvxpy.Variable()
    
    constraints = [
        cvxpy.sum(x, axis=0) == 1,  # Each resource is fully allocated
        x >= 0,                  # Cannot allocate negative amounts (for each element in the matrix) 
        x <= 1,                  # Cannot allocate more than 1 unit (for each element in the matrix)
    ]
    
    # Add constraints to ensure the minimum utility is less than or equal to each person's utility
    for i in range(num_peoples):
        constraints.append(min_utility <= utilities[i])
    
    # Define the optimization problem 
    problem = cvxpy.Problem(cvxpy.Maximize(min_utility), constraints)
    problem.solve()
    # After solving, all CVXPY variables (e.g., x, min_utility) hold their solution in `.value`

    results = []
    for i in range(num_peoples):
        results.append([x[i, j].value for j in range(num_resources)])
    return results, problem.value


def print_test_result(test_name, result, matrix):
    print(f"\n-- {test_name} --")
    if result is None:
        print("No optimal solution found.")
    else:
        allocations, optimal_value = result
        num_peoples = len(matrix)
        num_resources = len(matrix[0])
        
        print("\nValuation Matrix:")
        for row in matrix:
            print(row)

        print("\nResource Allocations:")
        for i in range(num_peoples):
            allocations_str = ', '.join(f"{allocations[i][j]:.2f} of resource #{j+1}" for j in range(num_resources))
            print(f"Agent #{i+1} receives: {allocations_str}.")
        
        print(f"\nOptimal Minimum Utility: {optimal_value:.2f}")
        
        utilities = [sum(allocations[i][j] * matrix[i][j] for j in range(num_resources)) for i in range(num_peoples)]
        print("\nUtilities:")
        for i in range(num_peoples):
            print(f"Agent #{i+1} has utility: {utilities[i]:.2f}")

def run_tests():
    # Test 1: Two agents, three resources
    matrix1 = [
        [81, 19, 1],
        [70, 1, 29]
    ]
    result1 = Egalitarian_division(matrix1)
    print_test_result("Test Case 1: Two agents, three resources", result1, matrix1)

    # Test 2: Balanced valuations
    matrix2 = [
        [40, 40, 40],
        [40, 40, 40],
        [40, 40, 40]
    ]
    result2 = Egalitarian_division(matrix2)
    print_test_result("Test Case 2: Balanced valuations", result2, matrix2)

    # Test 3: Different value preferences
    matrix3 = [
        [70, 20, 10],
        [10, 70, 20],
        [20, 10, 70]
    ]
    result3 = Egalitarian_division(matrix3)
    print_test_result("Test Case 3: Different value preferences", result3, matrix3)

    # Test 4: Four agents, four resources
    matrix4 = [
        [30, 20, 50, 10],
        [10, 40, 20, 30],
        [50, 10, 30, 40],
        [20, 50, 10, 30]
    ]
    result4 = Egalitarian_division(matrix4)
    print_test_result("Test Case 4: Four agents, four resources", result4, matrix4)

    # Test 5: Large value differences
    matrix5 = [
        [200, 1, 1],
        [1, 200, 1],
        [1, 1, 200]
    ]
    result5 = Egalitarian_division(matrix5)
    print_test_result("Test Case 5: Large value differences", result5, matrix5)

    # Test 6: Equal value preferences
    matrix6 = [
        [60, 60],
        [60, 60]
    ]
    result6 = Egalitarian_division(matrix6)
    print_test_result("Test Case 6: Equal value preferences", result6, matrix6)

    # Test 7: Two agents, two resources with unequal distribution
    matrix7 = [
        [100, 0],
        [0, 100]
    ]
    result7 = Egalitarian_division(matrix7)
    print_test_result("Test Case 7: Two agents, two resources with unequal distribution", result7, matrix7)

    # Test 8: Large number of resources
    matrix8 = [
        [10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10]
    ]
    result8 = Egalitarian_division(matrix8)
    print_test_result("Test Case 8: Large number of resources", result8, matrix8)

    # Test 9: Highly unbalanced valuations
    matrix9 = [
        [100, 5, 1],
        [1, 100, 5],
        [5, 1, 100]
    ]
    result9 = Egalitarian_division(matrix9)
    print_test_result("Test Case 9: Highly unbalanced valuations", result9, matrix9)

    # Test 10: Zero Value Allocation (No value for any agent)
    matrix10 = [
        [0, 0, 0],  
        [0, 0, 0],  
        [0, 0, 0]   
    ]
    result10 = Egalitarian_division(matrix10)
    print_test_result("Test Case 10: Zero Value Allocation (No value for any agent)", result10, matrix10)

if __name__ == "__main__":
    run_tests()
