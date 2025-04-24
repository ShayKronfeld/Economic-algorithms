import random
import time
import matplotlib.pyplot as plt
from typing import List, Tuple

def egalitarian_allocation(valuations: List[List[int]]) -> Tuple[List[List[int]], int]:
  
    num_players = len(valuations)
    num_items = len(valuations[0])

    best_result = {
        "min_value": float('-inf'),
        "allocation": [[] for _ in range(num_players)],
    }
    best_min_player = None

    def backtrack(item_index: int, current_allocation: List[List[int]], player_scores: List[int]):
        """
        Recursive backtracking function to explore all possible allocations.

        Args:
            item_index: The index of the current item being considered.
            current_allocation: The current allocation of items to players.
            player_scores: The current total value received by each player.
        """
        nonlocal best_min_player

        # Base case: all items have been allocated
        if item_index == num_items:
            min_value = min(player_scores)
            min_player = player_scores.index(min_value)

            # Update the best result if the current allocation is better (higher minimum value, or same minimum value with a smaller index of the weakest player)
            if (min_value > best_result["min_value"] or
                    (min_value == best_result["min_value"] and
                     (best_min_player is None or min_player < best_min_player))):
                best_result["min_value"] = min_value
                best_result["allocation"] = [list(p) for p in current_allocation]
                best_min_player = min_player
            return

        # Rule B: Optimistic Bound (Pruning based on the weakest player's potential)
        weakest_player = player_scores.index(min(player_scores))
        remaining_value = sum(valuations[weakest_player][i] for i in range(item_index, num_items))
        optimistic = player_scores[weakest_player] + remaining_value

        # If the best possible value for the weakest player in the current path is less than the current best minimum value, prune this branch
        if optimistic < best_result["min_value"]:
            return

        # Try assigning the current item to each player
        for p in range(num_players):
            current_allocation[p].append(item_index)
            player_scores[p] += valuations[p][item_index]

            # Recursively explore the next item
            backtrack(item_index + 1, current_allocation, player_scores)

            # Backtrack: undo the assignment to explore other possibilities
            current_allocation[p].pop()
            player_scores[p] -= valuations[p][item_index]

    # Start the backtracking process from the first item with an empty allocation and zero scores
    backtrack(0, [[] for _ in range(num_players)], [0] * num_players)
    return best_result["allocation"], best_result["min_value"]

def run_benchmark_side_by_side():
    item_counts = list(range(1, 11))       # Items from 1 to 10
    player_counts = [2, 3, 4]              # Player counts

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

    for i, num_players in enumerate(player_counts):
        times = []

        for num_items in item_counts:
            valuations = [
                [random.randint(1, 2**32) for _ in range(num_items)]
                for _ in range(num_players)
            ]

            start = time.time()
            allocation, min_value = egalitarian_allocation(valuations)
            end = time.time()

            times.append((end - start) * 1000)  # Convert to ms

        ax = axes[i]
        ax.plot(item_counts, times, marker='o')
        ax.set_title(f"{num_players} Players")
        ax.set_xlabel("Number of Items")
        ax.set_ylabel("Execution Time (ms)")
        ax.grid(True)

    plt.suptitle("Execution Time vs. Number of Items (By Player Count)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


run_benchmark_side_by_side()
