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
        nonlocal best_min_player

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

        # Rule B only: Optimistic Bound
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


valuations = [
    [4, 5, 6, 7, 8],
    [8, 7, 6, 5, 4]
]

allocation, min_value = egalitarian_allocation(valuations)

for i, items in enumerate(allocation):
    total = sum(valuations[i][j] for j in items)
    print(f"Player {i} gets items {items} with value {total}")

