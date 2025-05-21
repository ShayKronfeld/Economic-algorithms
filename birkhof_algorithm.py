import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_bipartite_graph(matrix, title, highlight_edges=None):
    """
    Displays a bipartite graph with edge weights.
    highlight_edges: A list of edges to highlight (e.g., the current matching).
    """
    n = matrix.shape[0]
    # Node labels now start from 1
    left_nodes = [f"L{i+1}" for i in range(n)]
    right_nodes = [f"R{j+1}" for j in range(n)]
    G = nx.Graph()
    G.add_nodes_from(left_nodes, bipartite=0)
    G.add_nodes_from(right_nodes, bipartite=1)

    edge_labels = {}
    for i in range(n):
        for j in range(n):
            weight = matrix[i, j]
            if weight > 1e-8:
                # Adjust edge creation to use 1-based indexing for labels
                G.add_edge(f"L{i+1}", f"R{j+1}", weight=weight)
                edge_labels[(f"L{i+1}", f"R{j+1}")] = f"{weight:.1f}"

    pos = {}
    # Position nodes from 1 at the top, down to N at the bottom
    # We reverse the range to get the highest index at the bottom
    pos.update((node, (0, n - 1 - i)) for i, node in enumerate(range(n))) # for left_nodes
    pos.update((node, (1, n - 1 - i)) for i, node in enumerate(range(n))) # for right_nodes

    # Map the generic node indices back to the L/R labels for positioning
    actual_pos = {}
    for i in range(n):
        actual_pos[f"L{i+1}"] = (0, n - 1 - i)
        actual_pos[f"R{i+1}"] = (1, n - 1 - i)

    plt.figure(figsize=(8, 6))
    nx.draw(G, actual_pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000)
    nx.draw_networkx_edge_labels(G, actual_pos, edge_labels=edge_labels)

    if highlight_edges:
        nx.draw_networkx_edges(G, actual_pos, edgelist=highlight_edges, edge_color='red', width=2)

    plt.title(title)
    plt.axis('off')
    plt.show()

def find_best_matching(residual):
    """
    Finds a perfect matching in the bipartite graph that maximizes the minimum weight.
    """
    n = residual.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, 2*n), bipartite=1)
    
    # Add edges with weights from the residual matrix
    for i in range(n):
        for j in range(n):
            if residual[i, j] > 1e-8:
                G.add_edge(i, n + j, weight=residual[i, j])
    
    # Find a perfect matching
    matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=range(n))
    
    # Construct the permutation matrix and find matching weights
    perm_matrix = np.zeros((n, n), dtype=np.float64)
    highlight_edges = []
    matching_weights = []
    
    for i in range(n):
        j = matching.get(i)
        if j is not None:
            j_idx = j - n  # Convert back to original index
            perm_matrix[i, j_idx] = 1
            matching_weights.append(residual[i, j_idx])
            highlight_edges.append((f"L{i+1}", f"R{j_idx+1}"))
    
    # Find the minimum weight in the matching
    min_val = min(matching_weights) if matching_weights else 0
    # Round to avoid floating point precision issues
    min_val = round(min_val, 10)
    
    return min_val, perm_matrix, highlight_edges

def birkhoff_von_neumann_decomposition(D):
    """
    Performs the Birkhoff-von Neumann decomposition on a doubly stochastic matrix D.
    Returns a list of (weight, permutation matrix) pairs.
    """
    n = D.shape[0]
    # Make a copy of D with higher precision to avoid floating-point errors
    residual = D.copy().astype(np.float64)
    decomposition = []

    # Print the initial matrix
    print("Initial matrix:")
    print(np.round(residual, 1))
    print()

    plot_bipartite_graph(residual, "Initial Graph with Weights")

    iteration = 1
    while np.any(residual > 1e-8):
        # Find best matching for current residual
        min_val, perm_matrix, highlight_edges = find_best_matching(residual)
        
        # Add this permutation with its weight to our decomposition
        decomposition.append((min_val, perm_matrix))
        
        # Print current step information
        print(f"Step {iteration}:")
        print(f"Weight: {min_val:.1f}")
        print("Permutation matrix:")
        print(perm_matrix.astype(int))
        
        # Subtract this permutation from the residual
        residual -= min_val * perm_matrix
        # Clean up small floating point errors in residual
        residual = np.round(residual, 10)
        
        # Print the updated residual
        print("Residual matrix:")
        print(np.round(residual, 1))
        print()
        
        plot_bipartite_graph(residual, f"Step {iteration}: Matching with weight {min_val:.1f}", highlight_edges)
        iteration += 1

    # Print final decomposition
    print("Final decomposition:")
    for idx, (weight, P) in enumerate(decomposition):
        print(f"P{idx+1} = {weight:.1f} * ")
        print(P.astype(int))
        print()

    return decomposition

# Example usage
if __name__ == "__main__":
    # Create a 4x4 doubly stochastic matrix
    D = np.array([
        [0.9, 0.1, 0.0, 0.0],
        [0.1, 0.8, 0.0, 0.1],
        [0.0, 0.1, 0.5, 0.4],
        [0.0, 0.0, 0.5, 0.5]
    ])

    decomposition = birkhoff_von_neumann_decomposition(D)
    
    # Verify the decomposition by summing up all weighted permutation matrices
    recomposed = np.zeros_like(D)
    for weight, P in decomposition:
        recomposed += weight * P
    
    print("Sum of weighted permutation matrices:")
    print(np.round(recomposed, 1))
    
    # Check if the recomposition matches the original matrix
    if np.allclose(recomposed, D, atol=1e-8):
        print("Decomposition is correct! The original matrix matches the sum of the weighted permutation matrices.")
    else:
        print("Decomposition error! The original matrix differs from the sum of the weighted permutation matrices.")
