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
        nonlocal best_result, best_min_player

        if item_index == num_items:
            min_value = min(player_scores)
            min_player = player_scores.index(min_value)

            if (min_value > best_result["min_value"] or
                    (min_value == best_result["min_value"] and
                     (best_min_player is None or min_player < best_min_player))):
                best_result["min_value"] = min_value
                best_result["allocation"] = [list(p) for p in current_allocation]
                best_min_player = min_player
            return

        weakest_player = player_scores.index(min(player_scores))
        remaining_value = sum(valuations[weakest_player][i] for i in range(item_index, num_items))
        optimistic = player_scores[weakest_player] + remaining_value
        if optimistic < best_result["min_value"]:
            return

        for p in range(num_players):
            current_allocation[p].append(item_index)
            player_scores[p] += valuations[p][item_index]

            backtrack(item_index + 1, current_allocation, player_scores)

            current_allocation[p].pop()
            player_scores[p] -= valuations[p][item_index]

    backtrack(0, [[] for _ in range(num_players)], [0] * num_players)
    return best_result["allocation"], best_result["min_value"]



def egalitarian_allocation_with_lower_bound(valuations: List[List[int]]) -> Tuple[List[List[int]], int]:
    num_players = len(valuations)
    num_items = len(valuations[0])
    best_result = {
        "min_value": float('-inf'),
        "allocation": [[] for _ in range(num_players)],
    }
    best_min_player = None

    def backtrack_with_lower_bound(item_index: int, current_allocation: List[List[int]], player_scores: List[int], best_min_value: float):
        nonlocal best_result, best_min_player

        if item_index == num_items:
            min_value = min(player_scores)
            min_player = player_scores.index(min_value)
            if (min_value > best_result["min_value"] or
                    (min_value == best_result["min_value"] and
                     (best_min_player is None or min_player < best_min_player))):
                best_result["min_value"] = min_value
                best_result["allocation"] = [list(p) for p in current_allocation]
                best_min_player = min_player
            return

        # Heuristic lower bound pruning
        if player_scores:
            min_current_score = min(player_scores)
            remaining_items_values = [valuations[p][i] for i in range(item_index, num_items) for p in range(num_players)]
            average_remaining_value = sum(remaining_items_values) / num_players if remaining_items_values and num_players > 0 else 0
            lower_bound = min_current_score + average_remaining_value
            # If the lower bound is unlikely to lead to a significant improvement
            if lower_bound < best_min_value * 0.9: # A safety margin of 10%
                return

        # Optimistic bound (upper bound) as in the original code
        weakest_player = player_scores.index(min(player_scores)) if player_scores else 0
        remaining_value = sum(valuations[weakest_player][i] for i in range(item_index, num_items))
        optimistic = (player_scores[weakest_player] if player_scores else 0) + remaining_value
        if optimistic < best_min_value:
            return

        for p in range(num_players):
            current_allocation[p].append(item_index)
            player_scores[p] += valuations[p][item_index]

            backtrack_with_lower_bound(item_index + 1, current_allocation, player_scores, best_result["min_value"])

            current_allocation[p].pop()
            player_scores[p] -= valuations[p][item_index]

    backtrack_with_lower_bound(0, [[] for _ in range(num_players)], [0] * num_players, float('-inf'))
    return best_result["allocation"], best_result["min_value"]


def run_benchmark_comparison():
    item_counts = list(range(1, 11))
    player_counts = [2, 3, 4]

    fig, axes = plt.subplots(1, len(player_counts), figsize=(12 * len(player_counts) / 3, 5))
    if len(player_counts) == 1:
        axes = [axes]

    for i, num_players in enumerate(player_counts):
        times_original = []
        times_lower_bound_pruning = []

        for num_items in item_counts:
            valuations = [
                [random.randint(1, 2**32) for _ in range(num_items)]
                for _ in range(num_players)
            ]

            # Measuring execution time for the original algorithm
            start_original = time.time()
            egalitarian_allocation(valuations)
            end_original = time.time()
            times_original.append((end_original - start_original) * 1000)

            # Measuring execution time for the algorithm with lower bound pruning
            start_lower_bound = time.time()
            allocation_lb, min_value_lb = egalitarian_allocation_with_lower_bound(valuations)
            end_lower_bound = time.time()
            times_lower_bound_pruning.append((end_lower_bound - start_lower_bound) * 1000)

        ax = axes[i]
        ax.plot(item_counts, times_original, marker='o', label='Original Algorithm')
        ax.plot(item_counts, times_lower_bound_pruning, marker='^', label='Lower Bound Pruning')
        ax.set_title(f"{num_players} Players")
        ax.set_xlabel("Number of Items")
        ax.set_ylabel("Execution Time (ms)")
        ax.grid(True)
        ax.legend()

    plt.suptitle("Comparison of Execution Times: Original vs. Lower Bound Pruning", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Running the benchmark
run_benchmark_comparison()